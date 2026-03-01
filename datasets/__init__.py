# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
from .talk2event_dataset import Talk2EventDataset

def build_dataset(dataset_file, image_set, args):
    if dataset_file == "talk2event":
        return Talk2EventDataset(args, image_set)
    raise ValueError(f'dataset {args.dataset_file} not supported')
