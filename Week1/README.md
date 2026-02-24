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
│   ├── main.py         # Entry point of Week 1
│   ├── utils.py        # Shared helper functions used across tasks
│   ├── task1/          # Code for Task 1
│   ├── task2/          # Code for Task 2
│   └── task3/          # Code for Task 3 
└── README.md           # README for Week 1
```

## How to run

The project can be executed in two different ways:  
(1) using a configuration file, or  
(2) directly from the command line.

---

### Option 1: Using a Configuration File (Recommended for Parameter Search)

You can define all parameters in a YAML configuration file and run:

```bash
python main.py --config config/taskX.yaml
```

### Option 2: Direct Command Line Execution

You can also run a single configuration directly from the terminal:

```bash
python main.py \
  --task taskX \
  --video path/to/video.avi \
  --annotations path/to/annotations.xml
``` 
Hyperparameters can also be passed manually.

## Task 1

### Single Gaussian background model

In Task 1 we implement a **single Gaussian model per pixel** to separate background from foreground. We use the **first 25%** of the video to estimate the background statistics (per-pixel **mean** and **standard deviation**), and we segment the **remaining 75%** using a fixed decision rule:

$
\text{Foreground if } |I_t(i) - \mu(i)| \ge \alpha \cdot (\sigma(i) + 2)
$

where $I_t(i)$ is the grayscale intensity at pixel $i$ and frame $t$, and $\mu(i), \sigma(i)$ are the background mean and standard deviation learned during training. The `+2` term prevents overly small $\sigma$ values from producing unstable thresholds.

### Foreground mask post-processing and detections

The raw foreground mask is refined with several steps to improve detection quality:

- **Shadow removal (HSV, Cucchiara-style test):**  
  We remove pixels classified as shadows using a chromaticity/brightness consistency check in HSV space. A pixel is considered shadow if its brightness decreases within a fixed range while hue and saturation remain similar to the background. We use fixed parameters:
  - brightness ratio bounds: `alpha=0.4`, `beta=0.9`
  - saturation threshold: `tau_s=50`
  - hue threshold: `tau_h=20`

- **Morphological filtering:**  
  We apply **opening** (remove small noise) and **closing** (fill small holes) with tunable kernel sizes, followed by a small **dilation** to connect fragmented blobs.

- **Bounding box extraction:**  
  We run connected components on the refined mask and convert components into bounding boxes, filtering by **minimum area**, fill ratio, and aspect ratio. We additionally remove nested boxes and merge overlapping boxes to reduce duplicates.

### Hyperparameter search and best configuration

We explored different values for the main segmentation and post-processing parameters:

- `alpha`: foreground threshold multiplier in the Gaussian decision rule. Larger values make the method more conservative (fewer foreground pixels).
- `min_area`: minimum connected-component area required to be considered an object (filters small noisy blobs).
- `open_size`: kernel size for morphological opening (noise removal).
- `close_size`: kernel size for morphological closing (hole filling / blob consolidation).

The best-performing configuration (`AP50 = 0.4150`) was:

- `alpha = 3`
- `min_area = 1500`
- `open_size = 3`
- `close_size = 7`

## Task 2

### Adaptive Gaussian Modelling

In Task 2 we extend the approach from **Task 1** by introducing an **adaptive (recursive) background model**.  
All components from Task 1 are kept unchanged:

- Foreground decision rule
- Shadow removal (HSV-based)
- Morphological filtering (opening, closing, dilation)
- Bounding box extraction and post-processing
- COCO evaluation (AP50)

The only modification concerns the **background modelling step**.

### Recursive Update Rule

After the initial training phase (first 25% of frames), the background model is updated during the remaining 75% of the sequence **only for pixels classified as background**:

$
\mu_i = \rho I_i + (1 - \rho)\mu_i
$

$
\sigma_i^2 = \rho (I_i - \mu_i)^2 + (1 - \rho)\sigma_i^2
$

where:

- $ \rho $ is the **learning rate**
- $ I_i $ is the current pixel intensity
- $ \mu_i, \sigma_i^2 $ are the background mean and variance

### Role of ρ (rho)

The parameter **ρ (rho)** controls how fast the background adapts:

- **Low ρ (e.g., 0.001)** → Slow adaptation  
  - More stable background  
  - Robust to short-term noise  
  - May fail under gradual illumination changes  

- **High ρ (e.g., 0.1)** → Fast adaptation  
  - Quickly adapts to lighting changes  
  - Risk of absorbing slow-moving objects into the background  

Thus, ρ defines the trade-off between **stability** and **adaptability**.

### Hyperparameter Selection

We used the best threshold found in Task 1:

- `alpha = 3`

Then, we evaluated different values of `rho` while keeping all other parameters fixed.

The best-performing configuration (`AP50 = 0.6419`) was:

- `alpha = 3`
- `rho = 0.01`

This configuration achieved the best balance between adaptation to scene changes and preservation of moving objects as foreground.

## Task 3