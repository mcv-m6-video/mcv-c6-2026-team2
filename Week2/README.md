# Week 2: Object Detection and Tracking

## Introduction

In the second week of the project, we move from traditional background modelling to Deep Learning-based Object Detection and Multiple Object Tracking (MOT). The goal is to detect vehicles in road traffic sequences and maintain their identities across time.

This week is divided into two main blocks:

1. **Object Detection**: Evaluating off-the-shelf models and fine-tuning them to improve performance on our specific traffic dataset.

2. **Object Tracking**: Implementing tracking algorithms starting from simple geometric overlap to more robust motion-based models like Kalman Filters.

## Dataset

We continue working with the AI City Challenge sequence **S03-C010**, provided for the project.

## Tasks Overview

The work for Week 2 is structured according to the project tasks:
 
**Task 1: Object Detection**
- **Task 1.1: Off-the-shelf:**  
  Evaluate pre-trained state-of-the-art detectors trained on COCO.

- **Task 1.2: Fine-tuning:**  
  Adapt a pre-trained model to the project sequence using transfer learning.

- **Task 1.3: K-Fold Cross validation:**
  Implement robust validation strategies (Fixed and Random folds) to ensure model generalization.

**Task 2: Object Tracking**
- **Task 2.1: Tracking by overlap:**  
  Implement a baseline tracker that associates detections between frames based on the Maximum Overlap (IoU).

- **Task 2.2: Tracking with a Kalman Filter:**  
  Enhance the tracker by incorporating a linear constant velocity motion model to predict future vehicle positions.

- **Task 2.3: IDF1 & HOTA Scores:**  
  Evaluate tracking performance using advanced metrics that account for both detection accuracy and identity consistency.


## Repository Structure

The Week2 directory is organized as follows:

```
Week2/
├── src/
│   ├── main.py         # Entry point of Week 2
│   ├── utils.py        # Shared helper functions used across tasks
│   ├── task1/          # Code for Task 1
│   └── task2/          # Code for Task 2
├── config/             # Config files 
├── environment.yml     # Environment yaml
└── README.md           # README for Week 2
```

## Installation
To run the code in this repo you must first install all needed libraries using conda with the help of the ```environment.yml``` file. This code is tested under python version 3.12.

```bash
cd Week2
conda env create -f environment.yml
conda activate c6
```

## How to run

The project can be executed in two different ways:  
(1) using a configuration file, or  
(2) directly from the command line.

---

### Option 1: Using a Configuration File (Recommended for Parameter Search)

You can define all parameters in a YAML configuration file and run:

```bash
python -m src.main --config config/taskX.yaml
```

### Option 2: Direct Command Line Execution

You can also run a single configuration directly from the terminal:

```bash
python -m src.main \
  --task taskX \
  --video path/to/video.avi \
  --annotations path/to/annotations.xml
``` 
Hyperparameters can also be passed manually.

## Task 1


## Task 2
