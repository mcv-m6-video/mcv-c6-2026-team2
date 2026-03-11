#!/bin/bash

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run -m src.main --dataset_path datasets/AI_CITY_CHALLENGE_2022_TRAIN/train/S03 --output_path results/task22 task21 --obj_detector_path checkpoints/fasterrcnn_faster-rcnn_best.pth --make_video