# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import os

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader
)
import utils.misc as utils
import datasets
from datasets import build_dataset
import datasets.samplers as samplers
from engine import evaluate, train_one_epoch
from models import build_bdetr_model
from models.postprocessors import build_postprocessors

import ipdb
st = ipdb.set_trace
import wandb
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Talk2event', add_help=False)
    
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default='configs/pretrain.json')
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )
    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )
    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument(
        '--scaleevent_encoder_lr',
        default=1e-6,
        type=float,
        help='Learning rate for trainable ScaleEvent encoder parameters.',
    )
    parser.add_argument("--text_encoder_lr", default=6e-6, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--val_batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=True, action='store_true')

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )
    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument("--ema", default=True, action='store_true')
    parser.add_argument("--ema_decay", type=float, default=0.9998)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--debug", action="store_true", help="Runs eval on train to check overfitting instead of val")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='refexp')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--resume', default='data/pretrain_2d.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--wandb', default=True, action='store_true')    
    parser.add_argument('--run_dir', default='exp1')
    parser.add_argument('--butd', default=False)
    parser.add_argument('--custom_text', default="all objects", type=str)
    parser.add_argument('--img_path', default='img.jpg', type=str)
    parser.add_argument('--with_learned_class_embeddings', default=True, action='store_true')
    parser.add_argument('--embeddings_path', type=str, default="")
    parser.add_argument("--new_contrastive", default=True, action='store_true')
    parser.add_argument("--large_scale", default=True, action='store_true')
    parser.add_argument("--event_config", default='models/event/backbone.yaml')
    parser.add_argument("--event_checkpoint", default='data/pretrain_event.ckpt')
    parser.add_argument(
        "--input_size",
        "--event_input_hw",
        dest="event_input_hw",
        default=(480, 640),
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help=(
            "Exact Talk2Event image/event resize in (height width) order. "
            "For example: --input_size 240 320."
        ),
    )
    parser.add_argument(
        "--talk2event_src_path",
        default="/dataset/shared/magic/",
        type=str,
        help="Root directory for Talk2Event data",
    )
    parser.add_argument("--modality", default='event')
    parser.add_argument("--attribute", default='fusion')
    parser.add_argument("--moe_fusion", default=True, action='store_true')
    return parser


def parse_main_args(argv=None):
    """Load JSON values as defaults, then let explicit CLI values win.

    The previous code updated ``args`` from the JSON inside ``main``. That
    meant values such as ``--batch_size 2`` were silently overwritten by the
    dataset config. Parsing the config as defaults preserves the expected
    command-line behaviour and also makes ``--input_size`` controllable.
    """
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script',
        parents=[get_args_parser()],
        allow_abbrev=False,
    )

    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument(
        "--dataset_config",
        default=parser.get_default("dataset_config"),
    )
    config_args, _ = config_probe.parse_known_args(argv)

    if config_args.dataset_config:
        config_path = Path(config_args.dataset_config)
        if not config_path.is_file():
            parser.error(f"Dataset config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            dataset_cfg = json.load(f)
        if not isinstance(dataset_cfg, dict):
            parser.error(
                f"Dataset config must contain a JSON object: {config_path}"
            )
        parser.set_defaults(**dataset_cfg)

    args = parser.parse_args(argv)
    args.event_input_hw = tuple(int(x) for x in args.event_input_hw)
    if any(x <= 0 for x in args.event_input_hw):
        parser.error(
            f"--input_size values must be positive, got {args.event_input_hw}"
        )
    return args


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    utils.init_distributed_mode(args)
    torch.autograd.set_detect_anomaly(True)
    
    # print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    if args.wandb:
        run = wandb.init(
            project="Talk2Event",
            name=args.run_dir,
        )
    else:
        run=None

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)   

    # print('=====args.moe_fusion:=====', args.moe_fusion)

    model, criterion, weight_dict = \
        build_bdetr_model(args)
    model.to(device)

    model_ema = deepcopy(model) if args.ema else None #True
    model_without_ddp = model
    if args.distributed:#False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    # Set up optimizers. ScaleEvent's pretrained ViT encoder is separated
    # from the newly initialized feature pyramid so it can use a much smaller LR.
    use_scaleevent = str(getattr(args, "event_backbone_type", "rvt")).lower() == "scaleevent"

    def is_scaleevent_encoder_parameter(name):
        if not use_scaleevent:
            return False
        return (
            "backbone.0.encoder." in name
            or "event_backbone.0.encoder." in name
        )

    named_trainable = [
        (name, parameter)
        for name, parameter in model_without_ddp.named_parameters()
        if parameter.requires_grad
    ]
    main_params = [
        parameter
        for name, parameter in named_trainable
        if "backbone" not in name and "text_encoder" not in name
    ]
    backbone_params = [
        parameter
        for name, parameter in named_trainable
        if "backbone" in name
        and "text_encoder" not in name
        and not is_scaleevent_encoder_parameter(name)
    ]
    text_encoder_params = [
        parameter
        for name, parameter in named_trainable
        if "text_encoder" in name
    ]
    scaleevent_encoder_params = [
        parameter
        for name, parameter in named_trainable
        if is_scaleevent_encoder_parameter(name)
    ]

    param_dicts = []
    for role, params, lr in (
        ("main", main_params, args.lr),
        ("backbone", backbone_params, args.lr_backbone),
        ("text_encoder", text_encoder_params, args.text_encoder_lr),
        ("scaleevent_encoder", scaleevent_encoder_params, args.scaleevent_encoder_lr),
    ):
        if params:
            param_dicts.append({"params": params, "lr": lr, "lr_role": role})
            print(
                f"Optimizer group {role}: "
                f"parameters={sum(p.numel() for p in params):,}, lr={lr:.3e}"
            )
    if args.sgd:#use adam, 1e-4
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    
    if not args.large_scale:#not use StepLR
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train, sampler_train, data_loader_train = None, None, None
    if not args.eval:
        if args.debug:
            image_set = 'train100'
        else:
            image_set = 'train' #here we use test for verification
        dataset_train = ConcatDataset(
            [build_dataset(name, image_set=image_set, args=args) for name in args.combine_datasets]#flickr, mixed, coco
        )

        if args.distributed:
            sampler_train = samplers.DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        if args.debug:
            image_set = "train100"
        else:
            image_set = "test"
        dset = build_dataset(dset_name, image_set=image_set, args=args)
        sampler = (
            samplers.DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=None, evaluator_list=None))

    output_dir = Path(args.output_dir)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            if os.path.exists(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                print(f"Warning: NOT loading checkpoint because {args.resume} doesn't exist. Don't worry if this is the first epoch")
        
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        num_model_keys = len(model_without_ddp.state_dict())
        num_ckpt_keys = len(checkpoint['model'])
        num_loaded_keys = num_model_keys - len(missing_keys)
        print(
            f"Resume summary: loaded={num_loaded_keys}, "
            f"model_keys={num_model_keys}, ckpt_keys={num_ckpt_keys}, "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.large_scale and 'lr_scheduler' in checkpoint:
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                print(optimizer.param_groups)
                if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"], strict=False)

    if args.eval:
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        limit = 100 if args.debug else -1
        for i, item in enumerate(val_tuples):

            postprocessors = build_postprocessors(args, item.dataset_name)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                device=device,
                args=args,
                epoch=args.start_epoch,
                wandb=run
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)
        return
    
    
    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            wandb=run,
            model_ema=model_ema
        )
        if not args.large_scale:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_freq == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    "model_ema": model_ema.state_dict() if args.ema else None,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if not args.large_scale else None,
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if (epoch + 1) % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            limit = 100 if args.debug else -1
            for i, item in enumerate(val_tuples):
                postprocessors = build_postprocessors(args, item.dataset_name)
                print(f"Evaluating {item.dataset_name}")
                curr_test_stats = evaluate(
                    model=test_model,
                    criterion=criterion,
                    postprocessors=postprocessors,
                    weight_dict=weight_dict,
                    data_loader=item.dataloader,
                    device=device,
                    args=args,
                    epoch=epoch,
                    wandb=run
                )
                test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.eval_skip == 0:
            metric = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])

            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = parse_main_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
