# Week 7: Ball Spotting Classification

This document explains how the `Week7` folder is organized, how the spotting pipeline works, how to run experiments, and highlights our best-performing model.

## Best Model Checkpoint

Our best model checkpoint can be found here:  
[checkpoint_best.pt](./best_checkpoint/checkpoint_best.pt)

### Best Configuration

The best results were obtained using:

- **Model**: T-DEED
- **Backbone**: X3D-S
- **Task**: Frame-level action spotting
- **Key idea**: Temporal detection with radi displacement (radius = 2)

This model achieved the highest performance:

- **AP10**: 46.06
- **AP12** : 41.67

Improving over Week 6 by better modeling temporal localization and detection with temporal aggregation and radius displacement.

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
├── inference.py       # Script for running inference 
├── qualitative_examples.py # Script for generating qualitative visualizations
├── extract_frames_snb.py   # Extract frames from raw SoccerNet videos
├── download_frames_snb.py  # Download pre-extracted frames from SoccerNet
├── README.md          # Documentation for Week 7
└── requirements.txt   # Python dependencies
```

## Installation

Install dependencies from inside `Week7`:

```bash
cd Week7
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
cd Week7
python main_spotting.py --model <config_name>
```

Example:

```bash
cd Week7
python main_spotting.py --model tdeed_x3d_s_radi2
```

### Run inference from a saved checkpoint

```bash
cd Week7
python inference.py \
  --model tdeed_x3d_s_radi2 \
  --checkpoint /path/to/checkpoint_best.pt
```

### Generate qualitative examples

The script `qualitative_examples.py` generates GIF visualizations of selected clips, showing predictions, ground-truth labels, and optionally a confidence chart.

You can choose clips in two different ways:

1. Select specific dataset indices

Use `--indices` to generate qualitative results for specific clips:

```bash
python qualitative_examples.py \
  --model tdeed_x3d_s_radi2 \
  --checkpoint /path/to/checkpoint_best.pt \
  --indices 10 25 42 \
  --show_chart
```

2. Randomly sample clips
Use `--num_random` to randomly select a number of clips from the dataset:

```bash
python qualitative_examples.py \
  --model tdeed_x3d_s_radi2 \
  --checkpoint /path/to/checkpoint_best.pt \
  --num_random 5 \
  --show_chart
```


## Evaluation
We report:

* AP per class
* AP10 (excluding rare classes: free kick, goal)
* AP12 (all classes)

Evaluation is performed with different temporal tolerances:

* 1 second (standard)
* 0.5 seconds (stricter)

## Experiment Summary
We explored several improvements over Week 6:

* T-DEED with RegNetY backbones (rny002, rny008) + GSF (as in the paper)
* Different temporal tolerances (0.5s and 1s)
* Soft labels with different sigmas (σ = 2, 4, 6, 12)
* Radius displacement with different radius (r = 2, 4)

All experiments except the last one were performed without displacement (r = 0).

### Key Insights
* Temporal detection models outperform frame-level approaches
* Radius displacement improves localization performance
* Soft labeling does not help much
* Best results achieved with T-DEED + X3D-S + radius = 2
