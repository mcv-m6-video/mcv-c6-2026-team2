# C6-Project: Road Traffic Monitoring (Team 2)

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




