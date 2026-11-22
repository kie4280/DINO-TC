# DINO-TC

Official repository for BMVC 2026 paper: **Adapting Foundational Image Models to Video Action Recognition via Learnable Context Tokens**.

This repository does not redistribute any video datasets. Download each dataset from its official source and make sure that its terms allow your intended use.

## Installation

The code is tested with Python 3.12, PyTorch 2.8, and CUDA. Create an environment and install the pinned dependencies:

```bash
conda create -n dino-tc python=3.12
conda activate dino-tc
pip install -r requirements.txt
```

The code uses `torchcodec` to read videos. Install a working FFmpeg installation if your platform requires it, and verify that PyTorch can see your GPU before running distributed training.

## Dataset preparation

Set `dataset.root` to the directory containing the dataset-specific files described below. The loaders expect the directory layouts shown here; do not point `dataset.root` at the directory containing this repository unless the dataset is stored there.

### Something-Something V2 (SSV2)

The SSV2 loader expects:

```text
${SSV2_ROOT}/
├── labels/
│   ├── labels.json
│   ├── train.json
│   └── val.json
└── videos/
    ├── <video-id>.webm
    └── ...
```

`labels.json` must map each action template to its integer class index. The annotation files must use the standard SSV2 fields, including `id` and `template`. The loader uses `val.json` for both validation and test mode.

The standard training configuration is `pilot/configs/ssv2/dino_tc_action.yaml`. Example:

```bash
python -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/ssv2/dino_tc_action.yaml \
  dataset.root=/path/to/ssv2
```

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/ssv2/dino_tc_action.yaml \
  dataset.root=/path/to/ssv2
```

### Kinetics-400

The published DINO-TC Kinetics configuration uses the OpenXLab loader, which expects:

```text
${K400_ROOT}/
├── annotations/
│   ├── train.txt
│   └── val.txt
├── train/
│   └── <filename>.mp4
└── val/
    └── <filename>.mp4
```

Each line in `train.txt` and `val.txt` must contain an MP4 filename followed by its integer class index, separated by a space. The loader resolves videos from `train/` or `val/` according to the split.

Use `pilot/configs/k400/dino_tc_action.yaml`:

```bash
python -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/k400/dino_tc_action.yaml \
  dataset.root=/path/to/k400
```

The alternative `Kinetics-400` loader expects `annotations/train.csv`, `annotations/val.csv`, and `annotations/kinetics_400_labels.csv`, together with `train/<youtube-id>.mp4` and `val/<youtube-id>.mp4`. Select it by setting `dataset.name=Kinetics-400` in a compatible configuration.

### Diving48

The Diving48 loader expects the extracted videos and split lists in this layout:

```text
${DIVING48_ROOT}/
└── diving48/
    ├── diving48_train_list_videos.txt
    ├── diving48_val_list_videos.txt
    ├── <video>.mp4
    └── ...
```

Each split-list line must contain a video filename and its integer class index separated by a space. Use `pilot/configs/diving48/dino_tc_32f.yaml`:

```bash
python -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/diving48/dino_tc_32f.yaml \
  dataset.root=/path/to/diving48
```

### FineGym

FineGym requires the videos, split files, and temporal annotation JSON:

```text
${FINEGYM_ROOT}/
├── finegym_annotation_info_v1.1.json
├── gym288_train_element_v1.1.txt
├── gym288_val_element.txt
└── videos/
    ├── gym288_train/
    │   └── <video-id>.mp4
    └── gym288_val/
        └── <video-id>.mp4
```

The split files are supplied by FineGym. The repository’s configuration uses the 288-class split (`288_train`, `288_val`, and `288_test`). Training videos are resolved from `videos/gym288_train/`; validation and test videos are resolved from `videos/gym288_val/`. Use `pilot/configs/finegym/dino_tc_32f.yaml`:

```bash
python -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/finegym/dino_tc_32f.yaml \
  dataset.root=/path/to/finegym
```

### Checking a dataset path

Before launching a long training job, run a short evaluation or debug job with the selected configuration. A missing annotation file or video usually indicates that `dataset.root` is one directory too high or too low.

```bash
python -m run eval \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/ssv2/dino_tc_action.yaml \
  dataset.root=/path/to/ssv2
```

## Pretrained weights

DINO-TC configurations use DINOv2 initialization by default. Place compatible weights under:

```text
pretrained_weights/dinov2/
├── dinov2_vits14_pretrain.pth
├── dinov2_vits14_reg4_pretrain.pth
├── dinov2_vitb14_pretrain.pth
├── dinov2_vitb14_reg4_pretrain.pth
├── dinov2_vitl14_pretrain.pth
└── dinov2_vitl14_reg4_pretrain.pth
```

The exact file selected depends on the model size and `model.register_tokens` value. The directory is intentionally ignored by Git. If `model.pretrained=true`, a missing required file raises `FileNotFoundError`.

CLIP-TC configurations additionally load OpenAI CLIP weights from `pretrained_weights/CLIP/`. These weights are also ignored by Git.

## Checkpoint loading

Loading checkpoints is still under construction and is currently TBD.

## Running experiments

The entry point is `run.py` and accepts `train`, `eval`, `test`, and visualization modes. Configurations are merged from a default config, a dataset/model config, and optional command-line overrides:

```bash
python -m run train \
  --default_config pilot/configs/_default_tc.yaml \
  --config pilot/configs/ssv2/dino_tc_action.yaml \
  dataset.root=/path/to/ssv2 \
  training.batch_size=8 \
  num_workers=8
```

Checkpoints and logs are written below `outputs/`, which is ignored by Git. Weights & Biases logging is disabled by default; add `--wandb` only after configuring your own W&B account and project settings.

## Repository notes

- Dataset videos, pretrained weights, and generated outputs are not part of the source release.
- Check the licenses and citation requirements of DINOv2, CLIP, SSV2, Kinetics-400, Diving48, FineGym, and all third-party dependencies before redistribution.
- The `pilot/configs/` directory contains the experiment settings used by the repository. Keep the configuration and checkpoint names together when publishing results so that experiments can be reproduced.

## Citation

If you use this code, please cite the accompanying BMVC paper.
