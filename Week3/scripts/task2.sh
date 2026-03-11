#!/bin/bash

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run -m src.main --config configs/task21.yaml task21
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run -m src.main --config configs/task22.yaml task21
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run -m src.main --dataset_path datasets/AI_CITY_CHALLENGE_2022_TRAIN/train/S01 --output_path results/task21 task21 --obj_detector_path checkpoints/fasterrcnn_faster-rcnn_best.pth --make_video
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 uv run -m src.main --dataset_path datasets/AI_CITY_CHALLENGE_2022_TRAIN/train/S03 --output_path results/task22 task21 --obj_detector_path checkpoints/fasterrcnn_faster-rcnn_best.pth --make_video