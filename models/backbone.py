# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2021 Carnegie Mellon University. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import numpy as np

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import ipdb
st = ipdb.set_trace

import yaml
from dotmap import DotMap
from .event.maxvit_rnn import RNNDetector as MaxViTRNNDetector
from .event.utils import _get_modified_hw_multiple_of


class FrozenBatchNorm2d(torch.nn.Module):
    """Frozen BatchNorm2d."""

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool,
                 butd=False, with_learned_class_embeddings=False,
                 embeddings_path=None, device=None):
        super().__init__()
        self.butd = butd
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if butd:
            if with_learned_class_embeddings:
                self.butd_class_embeddings = nn.Embedding(1601, 768)
                saved_class_embeddings = torch.from_numpy(np.load(embeddings_path, allow_pickle=True))
                saved_class_embeddings = torch.cat([torch.zeros(1, 768), saved_class_embeddings], dim=0)
                self.butd_class_embeddings.weight.data.copy_(saved_class_embeddings)
                self.butd_class_embeddings.requires_grad = False
            else:
                self.butd_class_embeddings = nn.Embedding(1601, 32)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class EventBackbone(nn.Module):

    def __init__(self, args):
        super().__init__()

        with open(args.event_config, "r") as f:
            config = yaml.safe_load(f)

        config = DotMap(config)
        backbone_cfg = config.backbone

        partition_split_32 = backbone_cfg.partition_split_32
        assert partition_split_32 in (1, 2, 4)

        multiple_of = 32 * partition_split_32
        requested_event_hw = tuple(int(x) for x in getattr(args, "event_input_hw", (480, 640)))
        if len(requested_event_hw) != 2:
            raise ValueError(f"event_input_hw must contain (H, W), got {requested_event_hw}")

        mdl_hw = _get_modified_hw_multiple_of(hw=requested_event_hw, multiple_of=multiple_of)
        self.requested_event_hw = requested_event_hw
        self.model_event_hw = mdl_hw
        backbone_cfg.in_res_hw = mdl_hw

        attention_cfg = backbone_cfg.stage.attention
        partition_size = tuple(x // (32 * partition_split_32) for x in mdl_hw)
        assert all(x > 0 for x in partition_size), (
            f"Invalid partition_size={partition_size} for event_input_hw={mdl_hw}"
        )
        assert (mdl_hw[0] // 32) % partition_size[0] == 0, f'{mdl_hw[0]=}, {partition_size[0]=}'
        assert (mdl_hw[1] // 32) % partition_size[1] == 0, f'{mdl_hw[1]=}, {partition_size[1]=}'
        if mdl_hw == requested_event_hw:
            print(
                f'Set event input size: {requested_event_hw}; '
                f'partition sizes: {partition_size}'
            )
        else:
            print(
                f'Set event data size: {requested_event_hw}; '
                f'pad inside RVT to: {mdl_hw}; '
                f'partition sizes: {partition_size}'
            )
        attention_cfg.partition_size = partition_size

        self.event_backbone = MaxViTRNNDetector(backbone_cfg)

        checkpoint = torch.load(args.event_checkpoint, map_location=args.device)
        event_ckpt = checkpoint["state_dict"]
        prefix = "mdl.backbone.stages"
        filtered_ckpt = {
            k[len('mdl.backbone') + 1:]: v
            for k, v in event_ckpt.items()
            if k.startswith(prefix + ".")
        }
        self.event_backbone.load_state_dict(filtered_ckpt, strict=False)
        self.strides = [4, 2, 2, 2]
        self.num_channels = [64, 128, 256, 512]

    def forward(self, tensor_list: NestedTensor):
        tensors, input_mask = tensor_list.decompose()
        input_hw = tuple(int(x) for x in tensors.shape[-2:])
        if input_hw != self.requested_event_hw:
            raise ValueError(
                f"RVT received event tensor size {input_hw}, but --input_size "
                f"was set to {self.requested_event_hw}."
            )

        pad_h = self.model_event_hw[0] - input_hw[0]
        pad_w = self.model_event_hw[1] - input_hw[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"RVT model canvas {self.model_event_hw} is smaller than input {input_hw}."
            )
        if pad_h or pad_w:
            tensors = F.pad(tensors, (0, pad_w, 0, pad_h), value=0)
            input_mask = F.pad(input_mask, (0, pad_w, 0, pad_h), value=True)

        xs = self.event_backbone(tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = input_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool,
                 dilation: bool, butd=False, with_learned_class_embeddings=False,
                 embeddings_path=None, device=None):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers,
                         butd=butd,
                         with_learned_class_embeddings=with_learned_class_embeddings,
                         embeddings_path=embeddings_path, device=device)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.backbone = backbone
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.num_feature_levels > 1
    backbone = Backbone(args.backbone, train_backbone,
                        return_interm_layers, args.dilation, args.butd,
                        args.with_learned_class_embeddings,
                        args.embeddings_path, args.device)
    return Joiner(backbone, position_embedding)


def build_event_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = EventBackbone(args)
    return Joiner(backbone, position_embedding)
