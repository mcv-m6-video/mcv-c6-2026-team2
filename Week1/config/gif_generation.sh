#!/bin/bash

src .venv/bin/activate

# Masks
echo Generating raw mask gif
python -m src.utils src/task1/results/videos/raw_masks.mp4 src/task1/results/videos/raw_masks.gif --start_frame 0 --end_frame 90 --opt
python -m src.utils src/task1/results/videos/shadowless_masks.mp4 src/task1/results/videos/shadowless_masks.gif --start_frame 0 --end_frame 90 --opt
python -m src.utils src/task1/results/videos/morpho_masks.mp4 src/task1/results/videos/morpho_masks.gif --start_frame 0 --end_frame 90 --opt
python -m src.utils src/task1/results/videos/best_masks.mp4 src/task1/results/videos/best_masks.gif --start_frame 0 --end_frame 90 --opt

# Ok Detections
echo Generating Ok detections gifs...
echo Task 1...
python -m src.utils src/task1/results/videos/detections_task1_a3.0_ma1500_os3_cs9_ap50_0.4369.mp4 src/task1/results/videos/task1_ok_detections.gif --start_frame 0 --end_frame 90 --opt
echo Task 2...
python -m src.utils src/task2/results/videos/detections_task2_a3_r0.01_ap50_0.6470.mp4 src/task2/results/videos/task2_ok_detections.gif --start_frame 0 --end_frame 90 --opt

# Small light variance
echo Generating Small light variance gifs...
echo Task 1...
python -m src.utils src/task1/results/videos/detections_task1_a3.0_ma1500_os3_cs9_ap50_0.4369.mp4 src/task1/results/videos/task1_small_light_variance.gif --start_frame 600 --end_frame 690 --opt
echo Task 2...
python -m src.utils src/task2/results/videos/detections_task2_a3_r0.01_ap50_0.6470.mp4 src/task2/results/videos/task2_small_light_variance.gif --start_frame 600 --end_frame 690 --opt

# Huge light variance
echo Generating Huge light variance gifs...
echo Task 1...
python -m src.utils src/task1/results/videos/detections_task1_a3.0_ma1500_os3_cs9_ap50_0.4369.mp4 src/task1/results/videos/task1_huge_light_variance.gif --start_frame 1180 --end_frame 1303 --opt
echo Task 2...
python -m src.utils src/task2/results/videos/detections_task2_a3_r0.01_ap50_0.6470.mp4 src/task2/results/videos/task2_huge_light_variance.gif --start_frame 1180 --end_frame 1303 --opt
