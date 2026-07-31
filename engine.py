# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2021 Carnegie Mellon University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR 
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# -------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Dict, Optional
from numpy import positive
from numpy import expand_dims


import torch
import utils.misc as utils
from datasets.data_prefetcher import data_prefetcher, targets_to
from datasets.visualize import visualize_inputs, visualize_coco
from utils.optim import update_ema, adjust_learning_rate
import ipdb
st = ipdb.set_trace

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    wandb=None,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    #targets keys
    #(['boxes', 'labels', 'caption', 'image_id', 'tokens_positive', 'area', 'iscrowd', 
    # 'orig_size', 'size', 'positive_map', 'dataset_name', 'sentence_id', 'original_img_id'])
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, event_samples, targets = prefetcher.next()

    num_training_steps = int(len(data_loader) * args.epochs)

    for i, _ in enumerate(metric_logger.log_every(
            range(len(data_loader)), print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        captions = [t["caption"] for t in targets]
        positive_map = torch.cat([t["positive_map"] for t in targets])
        memory_cache = None
        butd_boxes = None
        butd_masks = None
        butd_classes = None
        if args.butd:
            butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
            butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
            butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
        memory_cache = model(
            samples,
            event_samples,
            captions,
            encode_and_save=True,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        outputs = model(
            samples, event_samples, captions, encode_and_save=False,
            memory_cache=memory_cache,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks,
            targets=targets
        )

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
            if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if args.wandb and curr_step % 10000 == 0:
            wandb.log({f'train_loss_full': loss_value}, step=curr_step)
            for k, v in loss_dict_reduced_scaled.items():
                wandb.log({f'train_loss_{k}': v}, step=curr_step)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                model.parameters(), max_norm)

        optimizer.step()

        if args.large_scale:
            adjust_learning_rate(
                optimizer,
                epoch,
                curr_step,
                num_training_steps=num_training_steps,
                args=args,
            )

        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)
        
        metric_logger.update(loss=loss_value) #, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        samples, event_samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    device,
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    args,
    epoch=0,
    wandb=None
):
    # Set to eval
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    for i, (samples, event_samples, targets) in enumerate(metric_logger.log_every(data_loader, 500, 'Test:')):
        # Move variables to device
        curr_step = epoch * len(data_loader) + i
        samples = samples.to(device)
        event_samples = event_samples.to(device)
        targets = targets_to(targets, device)
        captions = [t["caption"] for t in targets]
        positive_map = torch.cat(
            [t["positive_map"] for t in targets])

        memory_cache = None
        butd_boxes = None
        butd_masks = None
        butd_classes = None
        if args.butd:
            butd_boxes = torch.stack([t['butd_boxes'] for t in targets], dim=0)
            butd_masks = torch.stack([t['butd_masks'] for t in targets], dim=0)
            butd_classes = torch.stack([t['butd_classes'] for t in targets], dim=0)
        memory_cache = model(
            samples,
            event_samples,
            captions,
            encode_and_save=True,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks
        )
        outputs = model(
            samples, event_samples, captions, encode_and_save=False,
            memory_cache=memory_cache,
            butd_boxes=butd_boxes,
            butd_classes=butd_classes,
            butd_masks=butd_masks,
            targets = targets
        )

        # Collect losses
        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }

        if args.wandb and curr_step % 1000 == 0:
            for k, v in loss_dict_reduced_scaled.items():
                wandb.log({f'eval_loss_{k}': v}, step=curr_step)

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))

        # Postprocess results (to bring them in COCO format?)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats
