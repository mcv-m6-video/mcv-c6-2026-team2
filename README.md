# C6-Project: Road Traffic Monitoring and Human Action Recognition (Team 2)

[Road Traffic Monitoring Final Presentation](https://docs.google.com/presentation/d/1vMEpA82IR3UQMij1ftU_XvWKjbRAGzm5CiM_HfaHH64/edit?usp=sharing)   

## Members
- Lore Oregi Lauzirika
- Júlia Garcia Torné
- Adrián García Tapia
- Marina Rosell Murillo

## Project Structure

This repository is structured by weeks. Each directory contains its own source code, experiments, and a dedicated README with detailed technical documentation.

```
├── .gitignore
├── README.md              # Global overview 
└── WeekX/                 # Week X
    ├── README.md          # Technical details for Week X
    └── src/               # Code for Week X
```

## Week 1: Background Estimation

In Week 1 of Module C6 (Video Surveillance for Road Traffic Monitoring), we focus on background subtraction techniques for foreground object detection in traffic video sequences. 

We implement a single Gaussian background model, extend it with an adaptive version, and evaluate detections using mAP@0.5. Finally, we compare our approach with state-of-the-art methods.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 1 README](./Week1/README.md).

## Week 2: Object Detection and Tracking

In Week 2 of Module C6, we move from background estimation to deep learning-based Object Detection and Object Tracking.

We evaluate state-of-the-art detectors, both as off-the-shelf models and by fine-tuning them to our specific traffic data. Additionally, we implement tracking algorithms based on Maximum Overlap and Kalman Filters to maintain vehicle identities across the sequence.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 2 README](./Week2/README.md).

## Week 3: Tracking with Optical Flow

In Week 3 of Module C6, we integrate Optical Flow into a tracking algorithm from last week.

We evaluate Optical Flow methods, both classic and deep learning approaches. Then we integrate the best Optical Flow method into the tracking algorithm from last week, substituting Kalmann with it. This evaluation is performed in 3 different datasets for single-camera tracking.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 3 README](./Week3/README.md).

## Week 4: Multi Camera Tracking

In Week 4 of Module C6, we extend the tracking problem from a single-camera setup to a multi-camera scenario, where the goal is to maintain consistent identities of vehicles across different viewpoints and non-overlapping fields of view.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 4 README](./Week4/README.md).

## Week 5: Ball Action Classification

In Week 5 of Module C6, we begin Project 2, focusing on Ball Action Classification (BAC) using the SoccerNet dataset. The objective of this task is to build and improve a multi-label video classification model capable of recognizing different ball-related actions in soccer matches. This serves as a first step toward the final goal of action spotting.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 5 README](./Week5/README.md).

## Week 6: Ball Action Spotting

In Week 6 of Module C6, we shift from action classification to the more challenging task of Ball Action Spotting, where the goal is to precisely localize actions in time within soccer videos. We start from a provided baseline model, reproduce its results, and explore improvements by incorporating temporal information.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 6 README](./Week6/README.md).

# Week 7: Ball Action Spotting

In Week 7 of Module C6, we continue working on Ball Action Spotting by exploring improvements to the baseline model.

We experiment with different architectures that incorporate temporal aggregation and evaluate their impact on performance. Additionally, we analyze results under different temporal tolerances and compare models to better understand their behavior.

For a detailed explanation of the tasks, implementation steps, and experiments refer to the dedicated [Week 7 README](./Week7/README.md).