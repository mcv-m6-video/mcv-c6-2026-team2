# Week 5: Ball Action Classification

This document explains how the `Week5` folder is organized, how the training and inference pipeline works, how to run experiments, and what the main configuration options mean, highlighting our best-performing setup.

## Best Model Checkpoint

Our best model checkpoint can be found [here](https://drive.google.com/drive/folders/1gEt6xjN-Hfl7A96zR3-zubQeJDtMuOa7?usp=sharing).

> Note: The checkpoint is hosted externally due to GitHub file size limitations (>100MB).

### Best Configuration
The best results were obtained using the following setup:

- **Backbone:** R3D-18 (3D ResNet-18)
- **Clip length:** 100 frames
- **Temporal stride:** 5

This configuration improved temporal modeling by capturing longer motion patterns compared to shorter clips used in the baseline.


## Folder organization

```text
Week5/
├── config/            # Experiment JSON files and config documentation
├── data/              # Dataset split files and dataset-specific README
├── dataset/           # Clip sampling and dataset loading code
├── model/             # Model definitions and model registry
├── notebooks/         # Dataset download and plotting notebooks
├── util/              # IO, evaluation, and helper functions
├── main_classification.py
├── inference.py
├── extract_frames_snb.py
├── download_frames_snb.py
└── requirements.txt
```

## What each part does

- `main_classification.py`: main training script. It trains the selected model, saves the best checkpoint, and evaluates on the test split.
- `inference.py`: loads a trained checkpoint and runs evaluation or exports qualitative examples.
- `config/`: one JSON file per experiment. The `--model` argument must match the filename without `.json`.
- `model/__init__.py`: selects the Python model class from `model_type`.
- `model/model_classification.py`: frame-based baseline model.
- `model/model_classification_lstm.py`: baseline model with an LSTM temporal head.
- `model/model_classification_lstm_attention.py`: LSTM model with attention.
- `model/model_classification_3dcnn.py`: 3D CNN backbones such as `r3d_18`, `r2plus1d_18`, `x3d_s`, and `x3d_m`.
- `model/model_classification_3dcnn_lstm.py`: 3D CNN features followed by an LSTM head.
- `dataset/datasets.py` and `dataset/frame.py`: create train/val/test datasets, sample clips, and load frames.
- `util/eval_classification.py`: computes the final per-class AP table plus AP10 and AP12.
- `experiments/`: helper scripts for launching runs with `nohup`.

## Installation

Install dependencies from inside `Week5`:

```bash
cd Week5
pip install -r requirements.txt
```

## Data preparation

Before training, make sure:

- the SoccerNet Ball dataset has been downloaded,
- the video frames have been extracted,
- the paths inside your config file are correct,
- and the split metadata has been generated at least once.

Useful files:

- `data/soccernetball/README.md`
- `download_frames_snb.py`
- `extract_frames_snb.py`

## Training pipeline

The training flow is:

1. Run `main_classification.py --model <config_name>`.
2. The script loads `config/<config_name>.json`.
3. The config selects the model family, backbone, clip settings, paths, and hyperparameters.
4. The dataset loader either stores clip metadata once or reuses a previously stored split.
5. The selected model is trained on the training split and validated after every epoch.
6. The best checkpoint is saved to `save_dir/<config_name>/checkpoints/checkpoint_best.pt`.
7. The best checkpoint is loaded and evaluated on the test split.

## How to run

### Train and evaluate an experiment

```bash
cd Week5
python3 main_classification.py --model <config_name>
```

Example:

```bash
python3 main_classification.py --model baseline_15e
```

### Run inference from a saved checkpoint

```bash
cd Week5
python3 inference.py \
  --model baseline_15e \
  --checkpoint /path/to/checkpoint_best.pt
```

### Save qualitative clips

```bash
python3 inference.py \
  --model baseline_15e \
  --checkpoint /path/to/checkpoint_best.pt \
  --save_qualitative \
  --num_qualitative 10 \
  --save_gif
```

## Command-line options

### `main_classification.py`

- `--model`: required. Name of the config file without `.json`.
- `--seed`: random seed. Default: `1`.
- `--sweep`: yaml file for WandB sweep configuration. Default: `None`.
- `--feature_arch`: optional override for the config backbone. Supported values: `r3d_18`, `r2plus1d_18`, `x3d_s`, `x3d_m`.

### `inference.py`

- `--model`: required. Config name without `.json`.
- `--checkpoint`: required. Path to a `.pt` checkpoint.
- `--seed`: random seed.
- `--save_qualitative`: save qualitative clips with labels and predictions.
- `--num_qualitative`: number of examples to export.
- `--qualitative_dir`: output directory for exported media.
- `--save_gif`: also export a `.gif`.
- `--pred_threshold`: threshold for positive predictions.
- `--only_event_clips`: only export clips that contain ground-truth events.

## Configuration options

Each experiment is described by a JSON file inside `config/`. The most important fields are:

- `model_type`: model family. Supported values in this folder are `baseline`, `lstm`, `lstm_attn`, `3dcnn`, and `3dcnn_lstm`.
- `frame_dir`: directory containing extracted video frames.
- `save_dir`: root directory used to save checkpoints, stored splits, and losses.
- `labels_dir`: directory containing labels.
- `store_mode`: `store` to create split metadata once, `load` to reuse stored splits.
- `task`: usually `classification`.
- `batch_size`: batch size.
- `clip_len`: number of frames per clip.
- `stride`: sample one frame every `stride` frames.
- `overlap`: Percentage of overlapping frames in contiguous clips.
- `dataset`: dataset name, usually `soccernetball`.
- `epoch_num_frames`: number of frames used to define one epoch.
- `feature_arch`: backbone architecture.
- `learning_rate`: optimizer learning rate.
- `num_classes`: number of classes.
- `num_epochs`: maximum number of epochs.
- `warm_up_epochs`: number of warmup epochs before cosine decay.
- `patience`: optional early stopping patience.
- `only_test`: if `true`, skip training and only run evaluation.
- `device`: `cuda` or `cpu`.
- `num_workers`: dataloader workers.
- `pretrained`: optional flag used by some 3D backbones.
- `loss_type`: optional loss selector. This codebase uses `bce` and `focal`.


## Supported model families

### Baseline-style models

- `baseline`: frame-based encoder plus classification head.
- `lstm`: baseline features followed by an LSTM.
- `lstm_attn`: LSTM plus attention.

Typical backbones for these models are RegNetY variants such as `rny002`, `rny004`, and `rny008`.

### 3D video models

- `3dcnn`: end-to-end spatio-temporal backbones.
- `3dcnn_lstm`: 3D CNN features followed by an LSTM.

Supported 3D backbones in the current CLI are:

- `r3d_18`
- `r2plus1d_18`
- `x3d_s`
- `x3d_m`

