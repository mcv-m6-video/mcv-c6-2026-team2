#!/bin/bash

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run -m src.main task12 --obj_detector_path checkpoints/fasterrcnn_faster-rcnn_best.pth --of_method memfof --make_video --start_frame 200 --end_frame 270 --iou_threshold 0.2 --dup_iou_threshold 0.5 --max_age 20 --min_hits 3 --conf_threshold 0.7