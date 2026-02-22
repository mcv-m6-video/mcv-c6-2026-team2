# Week 1: Background Estimation

## Introduction

In this week, we address the problem of background subtraction for traffic surveillance. The objective is to model the background of a video sequence and detect moving objects in the scene.

We implement a single Gaussian model per pixel to separate foreground from background, following a statistical approach. The model is evaluated using mAP@0.5, computed with the Pascal VOC 11-point interpolation method, as specified in the project guidelines.

The study is extended by introducing an adaptive (recursive) Gaussian model, and both approaches are compared in terms of detection performance. Finally, we benchmark our implementation against state-of-the-art background subtraction methods. 

## Dataset

We use the AI City Challenge sequence **S03-C010**, provided for the project.

## Tasks Overview

The work for Week 1 is structured according to the project tasks:
 
- **Task 1.1: Gaussian Modelling:**  
  Implement a single Gaussian model per pixel to estimate the background and perform foreground segmentation.

- **Task 1.2: Evaluation (mAP@0.5):**  
  Convert foreground masks into object detections and evaluate results using AP@0.5 based on IoU.

- **Task 2.1: Adaptive Gaussian Modelling:**  
  Extend the background model with a recursive (adaptive) update rule and tune parameters to maximize mAP.

- **Task 2.2: Adaptive vs Non-Adaptive Comparison:**  
  Compare both approaches in terms of detection performance.

- **Task 3: Comparison with State-of-the-Art:**  
  Benchmark our implementation against existing background subtraction methods.


## Repository Structure

The Week1 directory is organized as follows:

```
Week1/
│
├── src/
│   ├── Task1/          # Code for Task 1
│   ├── Task2/          # Code for Task 2
│   └── Task3/          # Code for Task 3 
└── README.md           # README for Week 1
```

## Task 1

## Task 2

## Task 3