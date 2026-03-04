from src.utils.visualization import create_video, gif_selector
from src.task1.finetune import main as run_task12
from src.task1.evaluate import main as run_task11
from src.task2.kalman_filter import main as run_task22
from src.task2.task21 import run_task21
from src.task2.utils import convert_xml_to_mot
import argparse
import os
import sys
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.task2.task21 import run_task21
from src.task1.evaluate import main as run_task11
from src.task1.finetune import main as run_task12
from src.task1.kfoldcrossval import main as  run_task13
from src.utils.visualization import create_video, gif_selector

def main():
    parser = argparse.ArgumentParser(description="Week 2: Tracking Pipeline - Team 02")
    
    parser.add_argument('--task', type=str, required=True, choices=['1.1', '1.2', '1.3', '2.1', 'vidgen', 'gifgen', 'xml2txt'], help="Task to run")
    
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument('--det_path', type=str, default="Data/AICity_data/train/S03/c010/det/det_fasterrcnn.txt")
    parser.add_argument('--gt_xml_path', type=str, default="/DATA/home/jgarcia/SpectralSegmentation/mcv-c6-2026-team2/Week 1/Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument('--video_path', type=str, default="/DATA/home/jgarcia/SpectralSegmentation/mcv-c6-2026-team2/Week 1/Data/AICity_data/train/S03/c010/vdo.avi")
    
    parser.add_argument('--make_video', action='store_true', help="Generate visualization video")
    parser.add_argument('--eval', action='store_true', help="Run TrackEval after tracking")
    parser.add_argument('--iou_thr', type=float, default=0.4, help="Minimum IoU required to associate a detection with an existing track.")
    parser.add_argument('--max_age', type=int, default=10, help="Maximum number of consecutive frames a track is kept alive without being matched.")
    parser.add_argument('--conf_thr', type=float, default=0.6, help="Minimum detection confidence required to consider a bounding box for tracking.")
    parser.add_argument('--filter_thr', type=float, default=0.5, help="IoU threshold used to remove duplicate detections within the same frame (NMS-like filtering).")

    # Task 1.1 args
    parser.add_argument("--model_name", type=str, default="faster-rcnn")
    parser.add_argument("--data_path", type=str,
                        default="Data/AICity_data/train/S03/c010/images")
    parser.add_argument("--annotations_path", type=str,
                        default="Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_file", type=str,
                        default="Data/AICity_data/train/S03/c010/det/det_fasterrcnn.txt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--split", type=str, choices=["train", "eval", "all"], default="all")

    # Task 1.2 args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--unfreeze_depth', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2)

    # Task 1.3 args
    parser.add_argument('--strategy', type=str, default="c", choices=["b", "c"], help="Cross-validation strategy to use for evaluation (kfold or holdout)")

    # Task 2.2 args (Kalman Filter)
    parser.add_argument('--preprocess', action='store_true',
                        help="Whether to preprocess detections by filtering duplicates before tracking.")

    # Video and gif generation
    parser.add_argument('--gt_det_path', type=str, default="Data/AICity_data/train/S03/c010/det/det_gt.txt")
    parser.add_argument('--video_output', type=str, default="results/video.mp4")
    parser.add_argument('--video_tracking', action="store_true")
    parser.add_argument('--video_max_frames', type=int, default=500)
    parser.add_argument('--gif_start_frame', type=int, default=0)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # 4. Overwrite the parser defaults with YAML values
        # This keeps the command line flags as the highest priority
        parser.set_defaults(**yaml_config)

        # 5. Re-parse to finalize the values
        args = parser.parse_args()

    output_folder = f"results/task{args.task.replace('.', '')}/fasterrcnn/"
    output_txt = os.path.join(output_folder, "data/s03c010.txt")
    trackeval_path = "src/task2/TrackEval" if args.eval else None

    if args.task == "1.1":
        print("\nRunning Task 1.1")
        run_task11(args)

    if args.task == "1.2":
        print("\nRunning Task 1.2")
        run_task12(args)
    
    if args.task == "1.3":
        print("\nRunning Task 1.3")
        run_task13(args)

    if args.task == '2.1':
        print("\nRunning Task 2.1")
        run_task21(det_path=args.det_path,
                   output_txt_path=output_txt,
                   video_path=args.video_path if args.make_video else None,
                   xml_gt_path=args.gt_xml_path if args.eval else None,
                   iou_threshold=args.iou_thr,
                   trackeval_path=trackeval_path,
                   make_video=args.make_video,
                   max_age=args.max_age,
                   conf_threshold=args.conf_thr,
                   filter_threshold=args.filter_thr
                   )

    if args.task == "2.2":
        print("\nRunning Task 2.2")
        run_task22(det_path=args.det_path,
                   output_txt_path=output_txt,
                   video_path=args.video_path if args.make_video else None,
                   xml_gt_path=args.gt_xml_path if args.eval else None,
                   iou_threshold=args.iou_thr,
                   trackeval_path=trackeval_path,
                   make_video=args.make_video,
                   max_age=args.max_age,
                   conf_threshold=args.conf_thr,
                   filter_threshold=args.filter_thr,
                   preprocess=args.preprocess,
                   )

    if args.task == "vidgen":
        print("\nGenerating Video")
        create_video(
            video_path=args.video_path,
            results_path=args.det_path,
            output_video_path=args.video_output,
            tracking=args.video_tracking,
            max_frames=args.video_max_frames,
            gt_path=args.gt_det_path
        )

    if args.task == "gifgen":
        print("\nGenerating gif")
        gif_selector(
            video_path=args.video_path,
            output_gif=args.video_output,
            start_frame=args.gif_start_frame,
            end_frame=args.gif_start_frame + args.video_max_frames
        )
    
    if args.task == "xml2txt":
        print("\nConverting XML to TXT format")
        convert_xml_to_mot(
            xml_path=args.gt_xml_path,
            output_txt_path=args.det_path
        )



if __name__ == "__main__":
    main()
