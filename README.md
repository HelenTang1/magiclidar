
# Talk2Event

Code for NeurIPS 2025 paper **Talk2Event**.

## Installation

### Requirements

- Linux
- Python 3.8
- PyTorch 2.0.0 + CUDA 11.7

We recommend using conda with the provided environment file:

```bash
conda env create -f environment.yml
conda activate talk2event
```

Compile custom CUDA operators:

```bash
sh utils/init.sh
```

## Data Preparation

Download dataset:

- `talk2event.zip`:  
  https://huggingface.co/datasets/dylanorange/talk2event/resolve/main/talk2event.zip

After extraction, set `talk2event_src_path` in `configs/pretrain.json` to the dataset root path:

```json
{
  "talk2event_src_path": "YOUR_DATA_PATH"
}
```

## Pretrained Weights

Place the following files under `data/`:

- `pretrain_event.ckpt`  
  https://huggingface.co/datasets/dylanorange/talk2event/resolve/main/pretrain_event.ckpt
- `pretrain_2d.pth`  
  https://huggingface.co/datasets/dylanorange/talk2event/resolve/main/pretrain_2d.pth

Evaluation checkpoints:

- `event.pth`  
  https://huggingface.co/datasets/dylanorange/talk2event/resolve/main/event.pth
- `fusion.pth`  
  https://huggingface.co/datasets/dylanorange/talk2event/resolve/main/fusion.pth

## Usage

### Training

```bash
python main.py \
  --output_dir outputs/train_fusion \
  --modality fusion \
  --attribute fusion \
  --moe_fusion
```

Common flags:

- `--resume`: resume from checkpoint
- `--modality`: `event` or `fusion`
- `--attribute`: `all` / `fusion` / other supported attribute keys

### Evaluation

Event model:

```bash
python test.py \
  --output_dir outputs/test_event \
  --resume ckpt/event.pth \
  --modality event \
  --attribute fusion \
  --moe_fusion
```

Fusion model:

```bash
python test.py \
  --output_dir outputs/test_fusion \
  --resume ckpt/fusion.pth \
  --modality fusion \
  --attribute fusion \
  --moe_fusion
```

Notes:

- `--resume` points to the checkpoint to evaluate.
- `--moe_fusion` is typically used together with `--attribute fusion`.
- Output files are saved as `{checkpoint_name}.json` and `{checkpoint_name}.pkl` in `--output_dir`.

## Visualization Toolkit

### 1. Generate `*.pkl` outputs

After running `test.py`, the output directory contains:

- `{checkpoint_name}.json`: metrics
- `{checkpoint_name}.pkl`: inference results for visualization

### 2. Visualize predictions

```bash
python vis_tools/active_window.py
```

Then select the target `*.pkl` file from the dropdown in the UI.
