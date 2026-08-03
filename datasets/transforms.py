# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MDETR 
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torch.nn.functional as torch_F
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate


def crop(image, event, target, region):

    cropped_image = F.crop(image, *region)
    cropped_event = F.crop(event, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map", "isfinal"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")
    
    if "butd_boxes" in target:
        butd_boxes = target["butd_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_butd_boxes = butd_boxes - torch.as_tensor([j, i, j, i])
        cropped_butd_boxes = torch.min(cropped_butd_boxes.reshape(-1, 2, 2), max_size)
        cropped_butd_boxes = cropped_butd_boxes.clamp(min=0)
        target["butd_boxes"] = cropped_butd_boxes.reshape(-1, 4)
        fields.append("butd_boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)
        for field in fields:
            if field in target:
                if field == 'butd_boxes':
                    target[field] = target[field]
                    continue
                target[field] = target[field][keep]

    return cropped_image, cropped_event, target


def hflip(image, event, target):
    flipped_image = F.hflip(image)
    flipped_event = F.hflip(event)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
        
    if "butd_boxes" in target:
        butd_boxes = target["butd_boxes"]
        butd_boxes = butd_boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["butd_boxes"] = butd_boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, flipped_event, target


def resize(image, event, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    # Keep event resizing consistent with the original Talk2Event code path.
    # torchvision.transforms.functional.resize enables antialiasing by default
    # in recent torchvision versions, which substantially smooths sparse event
    # tensors and changes their value distribution. Treat the temporal/event
    # channels as independent samples and use bilinear interpolation without
    # antialiasing, matching the previously documented implementation in
    # talk2event_dataset.py.
    if not torch.is_tensor(event) or event.ndim != 3:
        raise TypeError(
            f"Expected event tensor with shape [C, H, W], got {type(event)!r} "
            f"with shape {getattr(event, 'shape', None)}"
        )
    rescaled_event = torch_F.interpolate(
        event.unsqueeze(1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    if target is None:
        return rescaled_image, rescaled_event, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
        
    if "butd_boxes" in target:
        butd_boxes = target["butd_boxes"]
        scaled_butd_boxes = butd_boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["butd_boxes"] = scaled_butd_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, rescaled_event, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes
        
    def __call__(self, img: PIL.Image.Image, event, target: dict):
        init_boxes = len(target["boxes"])
        max_patience = 100
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            result_img, result_event, result_target = crop(img, event, target, region)
            if not self.respect_boxes or len(result_target["boxes"]) == init_boxes or i == max_patience - 1:
                return result_img, result_event, result_target
        return result_img, result_event, result_target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, event, target):
        if random.random() < self.p:
            return hflip(img, event, target)
        return img, event, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, event, target=None):
        size = random.choice(self.sizes)
        return resize(img, event, target, size, self.max_size)


class FixedResize(object):
    """Resize image, event tensor, and boxes to an exact ``(H, W)``."""

    def __init__(self, size_hw):
        if not isinstance(size_hw, (list, tuple)) or len(size_hw) != 2:
            raise ValueError(f"size_hw must be (H, W), got {size_hw}")
        self.size_hw = tuple(int(x) for x in size_hw)
        if any(x <= 0 for x in self.size_hw):
            raise ValueError(f"size_hw values must be positive, got {self.size_hw}")

    def __call__(self, img, event, target=None):
        height, width = self.size_hw
        # ``resize`` follows the original DETR convention for tuple sizes and
        # expects (W, H), while this public transform accepts the clearer (H, W).
        return resize(img, event, target, (width, height))


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, event, target):
        if random.random() < self.p:
            return self.transforms1(img, event, target)
        return self.transforms2(img, event, target)


class ToTensor(object):
    def __call__(self, img, event, target):
        return F.to_tensor(img), event, target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, event, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, event, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "butd_boxes" in target:
            butd_boxes = target["butd_boxes"]
            butd_boxes = box_xyxy_to_cxcywh(butd_boxes)
            butd_boxes = butd_boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["butd_boxes"] = butd_boxes
        return image, event, target


class RemoveDifficult(object):
    def __init__(self, enabled=False):
        self.remove_difficult = enabled

    def __call__(self, image, target=None):
        if target is None:
            return image, None
        target = target.copy()
        keep = ~target["iscrowd"].to(torch.bool) | (not self.remove_difficult)
        if "boxes" in target:
            target["boxes"] = target["boxes"][keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, event, target):
        for t in self.transforms:
            image, event, target = t(image, event, target)
        return image, event, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
