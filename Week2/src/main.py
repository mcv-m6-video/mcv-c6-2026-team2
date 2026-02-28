import argparse
import os
import sys
import numpy as np
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.task2.task21 import run_task21

def main():
    parser = argparse.ArgumentParser(description="Week 2: Tracking Pipeline - Team 02")
    
    parser.add_argument('--task', type=str, required=True, choices=['2.1'], help="Task to run")
    
    parser.add_argument('--det_path', type=str, default="Data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    parser.add_argument('--gt_xml', type=str, default="Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument('--video_path', type=str, default="Data/AICity_data/train/S03/c010/vdo.avi")
    
    parser.add_argument('--make_video', action='store_true', help="Generate visualization video")
    parser.add_argument('--eval', action='store_true', help="Run TrackEval after tracking")
    parser.add_argument('--iou_thr', type=float, default=0.4, help="IoU threshold for association")

    args = parser.parse_args()

    output_folder = f"results/task{args.task.replace('.', '')}"
    output_txt = os.path.join(output_folder, "data/s03c010.txt")
    trackeval_path = "src/task2/TrackEval" if args.eval else None

    if args.task == '2.1':
        print(f"\nRunning Task 2.1")
        run_task21(det_path=args.det_path,
            output_txt_path=output_txt,
            video_path=args.video_path if args.make_video else None,
            xml_gt_path=args.gt_xml if args.eval else None,
            iou_threshold=args.iou_thr,
            trackeval_path=trackeval_path,
            make_video=args.make_video
        )

if __name__ == "__main__":
    main()