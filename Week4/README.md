# Week 4: Multi Camera Tracking

## Introduction
In the fourth week of the project, we extend the previous single-camera tracking pipeline towards cross-camera identity matching. The goal is not only to track vehicles inside each camera independently, but also to associate identities across different cameras within the same sequence.

This week is divided into two main blocks:

- Single-camera tracking generation:
    Run the detector and tracker to obtain per-camera trajectories.

- Cross-camera matching and evaluation:
    Train or load a matching model to associate vehicle identities across cameras, generate the final multi-camera tracking results, and evaluate them.

## Dataset

For this week, we work with the **AI City Challenge 2022 Track 1** dataset.

The pipeline is designed to operate on the dataset root containing multiple sequences (S01, S03, S04), each composed of several camera views.

In our experimental setup:

- Sequences S01 and S04 are used for **training**,

- Sequence S03 is used for **evaluation** (test set).

## Tasks Overview

The work for Week 4 is structured according to the following steps:

**Task 1: Single-camera tracking**: 
- Run the detector and tracker on each camera independently.
- Generate tracking files that contain the trajectories of detected vehicles.

**Task 2: Train the matching model**:
- Train or fine-tune the model used to compare vehicle identities across cameras.
- This model is later used to merge tracks belonging to the same object.

**Task 3: Cross-camera matching**:
- Use the trained matching model to associate tracks coming from different cameras.
- Generate the final multi-camera tracking output.

**Task 4: Evaluation**:
- Evaluate the predicted tracking output against the ground truth annotations.

## Repository Structure

The Week4 directory is organized as follows:

```
Week4/
├── src/
│   ├── main.py              # Entry point of Week 4
│   ├── utils/               # Shared helper functions used across tasks
│   ├── tasks/               # Main task implementations
│   └── models/              # Model implementations
├── config/                  # Config files 
├── environment.yml          # Environment yaml
├── environment_eval.yml     # Environment yaml for evaluation
└── README.md                # README for Week 3
```

## Installation
To run the code in this repository, you need to create two separate conda environments:

1. **Main environment**: used for training, tracking, and matching.
2. **Evaluation environment***: required to run the official AI City Challenge evaluation code, which depends on specific library versions.

### Main Environment
Create and activate the main environment using:
```bash
cd Week4
conda env create -f environment.yml
conda activate c6
```
This environment includes all dependencies required for:
- tracking
- training the matcher
- cross-camera matching

### Evaluation Environment
The evaluation step relies on the official AI City Challenge evaluation tools, which require a different set of dependencies. For this reason, we provide a separate environment:
```bash
cd Week4
conda env create -f environment_eval.yml
conda activate eval_mcmt
```
> **Note**: The evaluation environment is only needed when running the evaluate command.

## How to run
All Week 4 functionality is executed from:
```bash
python -m src.main
```
The `main.py` file exposes four subcommands:
- `tracking`
- `train_matcher`
- `matching`
- `evaluate` 

The general syntax is:
```bash
python -m src.main <subcommand> [arguments]
```

To inspect the available commands and parameters:
```bash
# General help
python -m src.main -h

# Tracking help
python -m src.main tracking -h

# Train matcher help
python -m src.main train_matcher -h

# Matching help
python -m src.main matching -h

# Evaluation help
python -m src.main evaluate -h
```


