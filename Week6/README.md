# Week 6: Ball Spotting Classification

This document explains how the `Week6` folder is organized, how the spotting pipeline works, how to run experiments, and highlights our best-performing model.

## Best Model Checkpoint

Our best model checkpoint can be found here:  
[checkpoint_best.pt](./best_checkpoint/checkpoint_best.pt)

### Best Configuration

The best results were obtained using:

- **Model**: X3D-M + GRU (Bidirectional)
- **Task**: Frame-level action spotting
- **Key idea**: Combine strong spatiotemporal features (X3D-M) with temporal modeling (GRU)

This model achieved the highest performance:

- **AP10**: 44.34
- **AP12** : 41.5

Significantly improving over the baseline by better capturing temporal dependencies.

## Dataset: SoccerNet
We use the SoccerNet Ball Action Spotting dataset (SN-BAS-2025).

- Contains soccer match videos annotated with temporal events
- Each label includes:
    - Action class (e.g., pass, shot, cross…)
    - Timestamp (exact moment in the video)

## Folder organization

```text
Week6/
├── best_checkpoint/   # Best trained model (.pt) used for final evaluation and inference
├── config/            # JSON configuration files defining each experiment setup
├── data/              # Dataset metadata, splits, and dataset-specific instructions
├── dataset/           # Data loading, clip sampling, and frame processing utilities
├── model/             # Model architectures (baseline, RNNs, 3D CNNs, hybrids)
├── notebooks/         # Notebooks for analysis, visualization, and debugging
├── util/              # Evaluation (mAP), NMS, and general helper functions
├── main_spotting.py   # Main script for training and evaluating spotting models
├── inference.py       # Script for running inference and qualitative visualizations
├── extract_frames_snb.py   # Extract frames from raw SoccerNet videos
├── download_frames_snb.py  # Download pre-extracted frames from SoccerNet
├── README.md          # Documentation for Week 6
└── requirements.txt   # Python dependencies
```

## Installation

Install dependencies from inside `Week6`:

```bash
cd Week6
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

## Pipeline
1. Run `main_spotting.py --model <config_name>`
2. Load configuration from `config/<config_name>.json`
3. Extract clips (or reuse stored ones)
4. Train model with frame-level supervision
5. Apply Non-Maximum Suppression (NMS) during inference
6. Evaluate using mAP with temporal tolerance

## How to run
### Train and evaluate an experiment

```bash
cd Week6
python main_spotting.py --model <config_name>
```

Example:

```bash
cd Week6
python main_spotting.py --model x3d_m_gru
```

### Run inference from a saved checkpoint

```bash
cd Week6
python inference.py \
  --model x3d_m_gru \
  --checkpoint /path/to/checkpoint_best.pt
```

### Save qualitative clips

```bash
python inference.py \
  --model x3d_m_gru \
  --checkpoint /path/to/checkpoint_best.pt \
  --save_qualitative \
  --num_qualitative 10 \
  --save_gif
```

## Evaluation
We report:

* AP per class
* AP10 (excluding rare classes: free kick, goal)
* AP12 (all classes)

Evaluation uses the official SoccerNet metric with 1-second tolerance.

## Experiment Summary
We explored multiple approaches:

* Baseline (RegNet frame-based)
* LSTM / GRU temporal models
* Attention mechanisms
* Transformer-based models
* 3D CNNs (X3D)
* Hybrid models (3D CNN + RNN)

### Key Insights
* Pure frame-based models perform poorly
* Temporal modeling is crucial
* Best results come from combining:
    * Strong spatial encoder (X3D)
    * Temporal modeling (GRU)
