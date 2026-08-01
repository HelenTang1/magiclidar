#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="${RVT_TRAIN_DIR:-/20TB_04/yhtang_dataset/outputs_talk2event/RVT_event_only}"

if [[ $# -ge 1 ]]; then
  CHECKPOINT="$1"
else
  echo "No checkpoint specified." >&2
  echo "Usage: bash scripts/eval_rvt_event_only.sh [checkpoint.pth] [output_dir]" >&2
  exit 1
fi

OUTPUT_DIR="${2:-${TRAIN_DIR}/evaluation}"
EVENT_CHECKPOINT="${RVT_EVENT_CHECKPOINT:-/20TB_04/yhtang_dataset/event_ddt_pretrained/talk2event/pretrain_event.ckpt}"
BATCH_SIZE="${RVT_EVAL_BATCH_SIZE:-2}"
DEVICE="${RVT_EVAL_DEVICE:-cuda}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Evaluation checkpoint not found: ${CHECKPOINT}" >&2
  echo "Usage: bash scripts/eval_rvt_event_only.sh [checkpoint.pth] [output_dir]" >&2
  exit 1
fi

if [[ ! -f "${EVENT_CHECKPOINT}" ]]; then
  echo "RVT pretraining checkpoint not found: ${EVENT_CHECKPOINT}" >&2
  exit 1
fi

python test.py \
  --dataset_config configs/rvt_event_only.json \
  --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" \
  --event_checkpoint "${EVENT_CHECKPOINT}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}"
