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
  Evaluate the detector using K=4 cross-validation with two strategies: sequential folds that preserve the temporal structure of the video, and random folds that mix frames from the entire sequence.

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
│   ├── utils/          # Shared helper functions used across tasks
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
  --video_path path/to/video.avi \
  --gt_xml_path path/to/annotations.xml
``` 
Hyperparameters can also be passed manually. 

> **Note:** Task **2.3 (IDF1 & HOTA evaluation)** does not have a standalone command.  
> It runs internally when executing **Task 2.1** or **Task 2.2** with the `--eval` flag.

## Task 1.1
We chose the Faster R-CNN model implementation of Torchvision with the pretrained resnet50 weights. In this first task we reached a mAP@50 of 0.3735, with a fixed threshold of 0.5.

## Task 1.2
The backbone was frozen and only the prediction head was fine-tuned. The hyperparameters used can be found in the config file for this task./ After just 1 epoch, mAP@50 reached 0,9498, and in the end it achieved a 0,9602 in that metric.

## Task 1.3  
To evaluate the robustness of the detector, we applied **K=4 cross-validation** using two different strategies. Strategy B divides the sequence into four **sequential folds**, preserving the temporal order of the video. Strategy C instead creates **random folds**, mixing frames from the entire sequence.

Using sequential folds (Strategy B) we obtained an average **mAP of 0.924 ± 0.008**, while random folds (Strategy C) achieved **0.959 ± 0.001**. The higher performance in Strategy C is likely due to the random sampling of frames, which makes the training and validation sets more similar.

## Task 2.1 - Maximum Overlap Tracker

The Maximum Overlap tracker is a simple greedy tracking strategy that associates detections across frames based on Intersection over Union (IoU) between bounding boxes.

The algorithm maintains a list of active tracks, each storing the history of bounding boxes associated with a given object. For every new frame, detections are matched to existing tracks using the detection that maximizes the IoU with the last bounding box of the track.

This method is computationally inexpensive and serves as a strong baseline for multi-object tracking.

### Tracking Procedure

For each new frame:

1. **Detection filtering**: Detections are first filtered according to their confidence score: `score ≥ conf_threshold`. This removes low-confidence detections that are likely to correspond to noise.

2. **Duplicate removal:** Within each frame we remove duplicated detections using an IoU threshold (`filter_threshold`). When two detections overlap strongly, only the most confident one is kept.

3. **Track association:** Each active track searches for the detection with the highest IoU with its most recent bounding box.

    - If the IoU is greater than `iou_threshold`, the detection is assigned to the track.

    - Otherwise the track is considered unmatched for that frame.

4. **Track update:**

    - Matched tracks append the new bounding box to their history.

    - Unmatched tracks increase a miss counter.

5. **Track creation:** Any detection that is not assigned to an existing track initializes a new track.

6. **Track deletion:** Tracks that remain unmatched for more than `max_age` frames are removed.


### Hyperparameters

The main parameters controlling the behaviour of the tracker are:

- `iou_threshold`:	Minimum IoU required to associate a detection with an existing track
- `max_age`:	Maximum number of frames a track can survive without being matched
- `conf_threshold`:	Minimum detection confidence to be considered
- `filter_threshold`:	IoU threshold used to remove duplicate detections

These parameters control the balance between track stability and track fragmentation.

- Higher IoU thresholds produce more conservative associations but may fragment tracks.

- Higher max_age allows tracks to survive temporary occlusions but may increase identity switches.

### Strengths and Limitations

#### Advantages

- Extremely simple and fast

- Works well when detections are stable

- Easy to interpret and debug

#### Limitations

- Cannot predict object motion

- Struggles under occlusion

- Sensitive to detection noise

- Track identity can be lost when objects cross

These limitations motivate the use of motion models, which are introduced in Task 2.2.

## Task 2.2 - Kalman Filter Tracker (SORT)

In Task 2.2 we adapted an implementation from Alex Bewley of a Kalman Filter-based tracker following the SORT (Simple Online and Realtime Tracking) framework.

SORT extends the previous approach by incorporating a motion model that predicts the future position of objects. This prediction improves the robustness of data association and allows tracks to survive short detection failures.

The tracker combines three main components:

1. Kalman filter motion model

2. IoU-based data association

3. Hungarian assignment algorithm

### Kalman Filter Motion Model

Each object track is modeled using a constant velocity motion model.

The Kalman filter state vector is defined as:

```bash
[x, y, s, r, vx, vy, vs]
```

Where:

- `x, y` → center of the bounding box

- `s` → bounding box scale (area)

- `r` → aspect ratio

- `vx, vy, vs` → velocities of the corresponding quantities

At every frame the Kalman filter performs two steps:

#### Prediction

The tracker predicts the new state of each object using the motion model:

```bash
x_k = F x_{k-1}
```

This produces a predicted bounding box even if the object was not detected in the current frame.

#### Update

If a detection is associated with the track, the Kalman filter updates its estimate using the observed bounding box.

### Data Association

After predicting the position of all tracks, the algorithm associates detections with predicted tracks.

1. **IoU matrix computation:** The Intersection over Union is computed between every detection and every predicted track.

2. **Assignment with Hungarian algorithm:** The optimal assignment between detections and tracks is obtained using the Hungarian algorithm, which maximizes the total IoU.

3. **Association filtering:** Matches with IoU below iou_threshold are discarded.

### Track Lifecycle

Tracks follow the same general lifecycle as in Task 2.1:

- Matched tracks update their Kalman state

- Unmatched detections create new tracks

- Unmatched tracks increase a miss counter

- Tracks are removed when `time_since_update > max_age`

Additionally, SORT introduces a parameter `min_hits`, which controls when a track becomes confirmed and is allowed to be reported.


### Hyperparameters

The main parameters used by the Kalman tracker are:

- `iou_threshold`:	Minimum IoU required for detection-track association
- `max_age`:	Maximum number of frames a track can remain unmatched
- `min_hits`:	Minimum number of successful matches before a track is confirmed
- `conf_threshold`:	Detection confidence filtering
- `filter_threshold`:	Duplicate detection removal threshold


### Advantages of Kalman-Based Tracking

Compared to the maximum overlap tracker, the Kalman filter tracker provides several improvements:

- Motion prediction allows tracks to persist when detections are temporarily missing.

- More robust association when objects move between frames.

- Reduced track fragmentation.

Because of these properties, SORT is widely used as a strong baseline in multi-object tracking benchmarks.

## Task 2.3 - Evaluation with IDF1 and HOTA from TrackEval

To evaluate the tracking performance we use two standard metrics from TrackEval: **IDF1** and **HOTA**.

**IDF1 (Identification F1 Score)** measures how well the tracker preserves object identities over time. It is the harmonic mean of identification precision and recall, and penalizes identity switches and track fragmentation.

**HOTA (Higher Order Tracking Accuracy)** evaluates both detection quality and identity association simultaneously. It combines detection accuracy and association accuracy, providing a balanced measure of overall tracking performance.
