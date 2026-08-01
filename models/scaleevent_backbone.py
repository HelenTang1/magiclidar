"""ScaleEvent backbone adapter for the official Talk2Event codebase.

This module keeps the official Talk2Event dataset and BDETR model unchanged:
Talk2Event's 20-channel event tensor is converted to the 3-channel input used
by the ScaleEvent DINOv3 event encoder, then a lightweight FPN reconstructs the
four feature levels expected by Talk2Event.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import NestedTensor


class Talk2EventToScaleEventInput(nn.Module):
    """Convert Talk2Event's 10-bin x 2-polarity tensor to ScaleEvent RGB-like input.

    Expected input layout is ``[B, 20, H, W]`` with the flattened order
    ``[bin0_neg, bin0_pos, bin1_neg, bin1_pos, ...]``.

    The conversion matches the EventDDT ``split_10_to_3`` path:
      1. signed bin: positive - negative;
      2. temporal groups: 4 bins, 3 bins, 3 bins;
      3. per-sample 99th-percentile absolute-value normalization;
      4. map [-1, 1] with ``neutral + scale * value``;
      5. ImageNet normalization required by the DINOv3 backbone.

    No spatial resize or downsampling is performed.
    """

    def __init__(
        self,
        quantile: float = 0.99,
        neutral: float = 59.0,
        scale: float = 80.0,
    ) -> None:
        super().__init__()
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")

        self.quantile = float(quantile)
        self.neutral = float(neutral)
        self.scale = float(scale)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, events: Tensor) -> Tensor:
        if events.ndim != 4:
            raise ValueError(
                f"Expected Talk2Event events shaped [B,20,H,W], got {tuple(events.shape)}"
            )
        if events.shape[1] != 20:
            raise ValueError(
                "ScaleEvent conversion expects 20 channels "
                f"(10 bins x 2 polarities), got C={events.shape[1]}"
            )

        events = events.float()
        batch, _, height, width = events.shape
        per_bin = events.reshape(batch, 10, 2, height, width)

        # EventDDT convention: channel 1 is positive, channel 0 is negative.
        signed = per_bin[:, :, 1] - per_bin[:, :, 0]  # [B, 10, H, W]

        scaleevent_input = torch.stack(
            (
                signed[:, 0:4].sum(dim=1),
                signed[:, 4:7].sum(dim=1),
                signed[:, 7:10].sum(dim=1),
            ),
            dim=1,
        )  # [B, 3, H, W]

        # Robust per-sample normalization. The input has no gradient, but keeping
        # this operation in PyTorch avoids CPU/GPU copies inside the dataloader.
        abs_flat = scaleevent_input.abs().flatten(1)
        q = torch.quantile(abs_flat, self.quantile, dim=1).clamp_min(1e-6)
        scaleevent_input = scaleevent_input / q[:, None, None, None]
        scaleevent_input = scaleevent_input.clamp_(-1.0, 1.0)

        # Match the RGB-like voxel mapping used by the previous EventDDT
        # ScaleEvent route, then apply ImageNet normalization.
        scaleevent_input = self.neutral + self.scale * scaleevent_input
        scaleevent_input = scaleevent_input.clamp_(0.0, 255.0).div_(255.0)
        scaleevent_input = (
            scaleevent_input - self.imagenet_mean.to(scaleevent_input.dtype)
        ) / self.imagenet_std.to(scaleevent_input.dtype)
        return scaleevent_input


def _unwrap_checkpoint(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "ScaleEvent checkpoint must be a mapping, "
            f"got {type(checkpoint).__name__}"
        )

    current: Mapping[str, object] = checkpoint
    for key in ("state_dict", "model", "net"):
        value = current.get(key)
        if isinstance(value, Mapping):
            current = value
            break

    tensor_state = {key: value for key, value in current.items() if torch.is_tensor(value)}
    if not tensor_state:
        raise RuntimeError("No tensor state_dict entries found in ScaleEvent checkpoint")
    return tensor_state


def _match_encoder_state(
    raw_state: Mapping[str, Tensor],
    encoder_state: Mapping[str, Tensor],
) -> Dict[str, Tensor]:
    """Support official ScaleEvent and EventDDT checkpoint key prefixes."""

    prefixes = (
        "module.event_encoder.",
        "event_encoder.",
        "module.encoder.",
        "encoder.",
        "module.",
    )

    matched: Dict[str, Tensor] = {}
    for raw_key, value in raw_state.items():
        candidates = [raw_key]
        for prefix in prefixes:
            if raw_key.startswith(prefix):
                candidates.append(raw_key[len(prefix) :])

        for candidate in candidates:
            expected = encoder_state.get(candidate)
            if expected is not None and tuple(expected.shape) == tuple(value.shape):
                matched[candidate] = value
                break

    return matched


class ScaleEventBackbone(nn.Module):
    """ScaleEvent ViT encoder plus a four-level Talk2Event feature pyramid."""

    def __init__(self, args) -> None:
        super().__init__()

        repo_root = os.path.abspath(os.path.expanduser(args.scaleevent_repo_root))
        if not os.path.isdir(repo_root):
            raise FileNotFoundError(
                f"ScaleEvent repository does not exist: {repo_root}"
            )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            from dinov3.hub.backbones import (  # type: ignore
                dinov3_vitb16,
                dinov3_vitl16,
                dinov3_vits16,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import ScaleEvent's bundled DINOv3. "
                f"Check scaleevent_repo_root={repo_root!r}."
            ) from exc

        model_size = str(getattr(args, "scaleevent_model_size", "l")).lower()
        builders = {
            "s": dinov3_vits16,
            "b": dinov3_vitb16,
            "l": dinov3_vitl16,
        }
        if model_size not in builders:
            raise ValueError(
                f"Unsupported scaleevent_model_size={model_size!r}; choose s, b, or l"
            )

        # Do not load the image-pretrained DINOv3 weights here. The event encoder
        # is loaded from the ScaleEvent checkpoint below.
        self.encoder = builders[model_size](pretrained=False)
        self.output_key = str(
            getattr(args, "scaleevent_output_key", "x_norm_patchtokens")
        )
        self.freeze_encoder = bool(getattr(args, "scaleevent_freeze", True))
        configured_trainable_blocks = getattr(args, "scaleevent_trainable_blocks", None)
        if self.freeze_encoder:
            self.trainable_blocks = 0
        elif configured_trainable_blocks is None:
            # Preserve the old scaleevent_freeze=false behavior: fine-tune all blocks.
            self.trainable_blocks = -1
        else:
            self.trainable_blocks = int(configured_trainable_blocks)
        self.train_final_norm = bool(
            getattr(args, "scaleevent_train_final_norm", True)
        )

        checkpoint_path = os.path.abspath(
            os.path.expanduser(args.scaleevent_checkpoint)
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ScaleEvent checkpoint does not exist: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw_state = _unwrap_checkpoint(checkpoint)
        encoder_state = self.encoder.state_dict()
        matched_state = _match_encoder_state(raw_state, encoder_state)
        if not matched_state:
            sample_keys = list(raw_state.keys())[:10]
            raise RuntimeError(
                "No ScaleEvent checkpoint tensors matched the DINOv3 event encoder. "
                f"First checkpoint keys: {sample_keys}"
            )

        missing, unexpected = self.encoder.load_state_dict(matched_state, strict=False)
        strict_load = bool(getattr(args, "scaleevent_strict_load", True))
        if strict_load and (missing or unexpected):
            raise RuntimeError(
                "Strict ScaleEvent loading failed: "
                f"matched={len(matched_state)}, missing={len(missing)}, "
                f"unexpected={len(unexpected)}. "
                f"First missing keys: {missing[:10]}"
            )

        self._configure_encoder_trainability()
        trainable_encoder_params = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )
        total_encoder_params = sum(
            parameter.numel() for parameter in self.encoder.parameters()
        )
        print(
            "ScaleEvent encoder load summary: "
            f"checkpoint={checkpoint_path}, matched={len(matched_state)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"trainable_blocks={self.trainable_blocks}, "
            f"trainable_params={trainable_encoder_params:,}/{total_encoder_params:,}"
        )

        embed_dim = int(self.encoder.embed_dim)
        self.input_adapter = Talk2EventToScaleEventInput(
            quantile=float(getattr(args, "scaleevent_voxel_quantile", 0.99)),
            neutral=float(getattr(args, "scaleevent_voxel_neutral", 59.0)),
            scale=float(getattr(args, "scaleevent_voxel_scale", 80.0)),
        )

        # ScaleEvent ViT-L/16 produces one 30x40 grid for a 480x640 input.
        # Reconstruct the RVT-compatible pyramid and channel widths so the
        # official BDETR input projections can be reused from pretrain_2d.pth.
        self.up_to_stride8 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2),
            nn.GroupNorm(32, 128),
            nn.GELU(),
        )
        self.up_to_stride4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(16, 64),
            nn.GELU(),
        )
        self.to_stride16 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        self.to_stride32 = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
        )

        self.strides = [4, 8, 16, 32]
        self.num_channels = [64, 128, 256, 512]

    def _configure_encoder_trainability(self) -> None:
        """Freeze all, fine-tune a suffix, or fine-tune the full ScaleEvent ViT."""

        self.encoder.requires_grad_(False)

        if self.trainable_blocks == 0:
            return

        blocks = getattr(self.encoder, "blocks", None)
        if blocks is None:
            if self.trainable_blocks > 0:
                raise AttributeError(
                    "Partial ScaleEvent fine-tuning requires encoder.blocks, "
                    "but the selected DINOv3 encoder does not expose it."
                )
            self.encoder.requires_grad_(True)
            return

        num_blocks = len(blocks)
        if self.trainable_blocks < 0:
            self.encoder.requires_grad_(True)
            return
        if self.trainable_blocks > num_blocks:
            raise ValueError(
                "scaleevent_trainable_blocks exceeds encoder depth: "
                f"requested={self.trainable_blocks}, available={num_blocks}"
            )

        for block in blocks[-self.trainable_blocks:]:
            block.requires_grad_(True)

        if self.train_final_norm:
            for norm_name in ("norm", "norm_head"):
                norm = getattr(self.encoder, norm_name, None)
                if isinstance(norm, nn.Module):
                    norm.requires_grad_(True)

    def _set_encoder_train_mode(self, mode: bool) -> None:
        if self.trainable_blocks == 0:
            self.encoder.eval()
            return
        if self.trainable_blocks < 0:
            self.encoder.train(mode)
            return

        # Keep the frozen prefix deterministic and enable training behavior only
        # for the trainable suffix and final normalization layers.
        self.encoder.eval()
        blocks = getattr(self.encoder, "blocks")
        for block in blocks[-self.trainable_blocks:]:
            block.train(mode)
        if self.train_final_norm:
            for norm_name in ("norm", "norm_head"):
                norm = getattr(self.encoder, norm_name, None)
                if isinstance(norm, nn.Module):
                    norm.train(mode)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_encoder_train_mode(mode)
        return self

    def _encode(self, x: Tensor) -> Tensor:
        if self.trainable_blocks == 0:
            with torch.no_grad():
                outputs = self.encoder.forward_features(x)
        else:
            outputs = self.encoder.forward_features(x)

        if not isinstance(outputs, Mapping):
            raise TypeError(
                "ScaleEvent encoder.forward_features() must return a mapping, "
                f"got {type(outputs).__name__}"
            )
        if self.output_key not in outputs:
            raise KeyError(
                f"ScaleEvent output key {self.output_key!r} is missing. "
                f"Available keys: {list(outputs.keys())}"
            )

        tokens = outputs[self.output_key]
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens [B,N,C], got {tuple(tokens.shape)}"
            )
        return tokens

    def forward(self, tensor_list: NestedTensor):
        events = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None:
            raise ValueError("Talk2Event NestedTensor mask must not be None")

        scaleevent_input = self.input_adapter(events)
        batch, _, height, width = scaleevent_input.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "ScaleEvent ViT/16 requires H and W divisible by 16, "
                f"got {(height, width)}"
            )

        patch_tokens = self._encode(scaleevent_input)
        patch_h, patch_w = height // 16, width // 16
        expected_tokens = patch_h * patch_w
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "Unexpected ScaleEvent patch-token count: "
                f"got {patch_tokens.shape[1]}, expected {expected_tokens} "
                f"for input {(height, width)}"
            )

        feature = patch_tokens.transpose(1, 2).reshape(
            batch, patch_tokens.shape[-1], patch_h, patch_w
        )

        stride8 = self.up_to_stride8(feature)
        stride4 = self.up_to_stride4(stride8)
        stride16 = self.to_stride16(feature)
        stride32 = self.to_stride32(feature)
        features = (stride4, stride8, stride16, stride32)

        output = OrderedDict()
        for level, level_feature in enumerate(features):
            level_mask = F.interpolate(
                mask[:, None].float(),
                size=level_feature.shape[-2:],
                mode="nearest",
            )[:, 0].to(torch.bool)
            output[str(level)] = NestedTensor(level_feature, level_mask)
        return output
