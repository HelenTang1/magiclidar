#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/data4/yhtang/exp/EventDDT_Private/outputs_talk2event/ScaleEvent_event_only_finetune}"
PRETRAIN_2D="${2:-/data4/yhtang/exp/EventDDT_Private/pretrain_weights/talk2event/pretrain_2d.pth}"

python main.py \
  --dataset_config configs/scaleevent_event_only_finetune.json \
  --output_dir "${OUTPUT_DIR}" \
  --run_dir ScaleEvent_event_only_finetune \
  --resume "${PRETRAIN_2D}" \
  --batch_size 1
