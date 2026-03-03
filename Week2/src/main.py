import argparse
import os
import sys
import numpy as np
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.task2.task21 import run_task21
from src.task1.evaluate import main as run_task11
from src.task1.finetune import main as run_task12

def main():
    parser = argparse.ArgumentParser(description="Week 2: Tracking Pipeline - Team 02")
    
    parser.add_argument('--task', type=str, required=True, choices=['1.1', '1.2', '2.1'], help="Task to run")
    
    parser.add_argument('--det_path', type=str, default="Data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")
    parser.add_argument('--gt_xml', type=str, default="Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument('--video_path', type=str, default="Data/AICity_data/train/S03/c010/vdo.avi")
    
    parser.add_argument('--make_video', action='store_true', help="Generate visualization video")
    parser.add_argument('--eval', action='store_true', help="Run TrackEval after tracking")
    parser.add_argument('--iou_thr', type=float, default=0.4, help="IoU threshold for association")

    # Task 1.1 args
    parser.add_argument("--model_name", type=str, default="faster-rcnn")
    parser.add_argument("--data_path", type=str, default="Data/AICity_data/train/S03/c010/images")
    parser.add_argument("--annotations_path", type=str, default="Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_file", type=str, default="Data/AICity_data/train/S03/c010/det/det_fasterrcnn.txt")
    parser.add_argument("--log_wandb", action="store_true")

    # Task 1.2 args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--unfreeze_depth', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2)

    args = parser.parse_args()

    output_folder = f"results/task{args.task.replace('.', '')}"
    output_txt = os.path.join(output_folder, "data/s03c010.txt")
    trackeval_path = "src/task2/TrackEval" if args.eval else None

    if args.task == "1.1":
        print("\nRunning Task 1.1")
        run_task11(args)
    
    if args.task == "1.2":
        print("\nRunning Task 1.2")
        run_task12(args)

    if args.task == '2.1':
        print("\nRunning Task 2.1")
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