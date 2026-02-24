import argparse
from task1.task1 import run_task1
from task2.task2 import run_task2
from utils import create_detection_gif, create_detection_video
import yaml
import os

def main():

    parser = argparse.ArgumentParser(description="MCV Week1 Project")

    parser.add_argument("--task", type=str, default=None)

    parser.add_argument("--video", type=str, default=None,
                        help="Path to video")

    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to XML annotations")
    
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--alpha", nargs='+', type=float, default=[3.0])
    parser.add_argument("--min_area", type=int, default=[1500])
    parser.add_argument("--open_size", type=int, default=[3])
    parser.add_argument("--close_size", type=int, default=[7])
    parser.add_argument("--rho", nargs='+', type=float, default=[0.01])

    parser.add_argument("--gif", action="store_true", help="Generate detection GIF at the end")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
            
        # 4. Overwrite the parser defaults with YAML values
        # This keeps the command line flags as the highest priority
        parser.set_defaults(**yaml_config)
        
        # 5. Re-parse to finalize the values
        args = parser.parse_args()
    
    assert args.task
    assert args.video
    assert args.annotations

    best_alpha = None
    best_rho = None

    if args.task == "task1":
        all_boxes, gt_dict, train_end, best_alpha, best_min_area, best_open_size, best_close_size, ap50 = run_task1(args)
        desc = f"_a{best_alpha}_ma{best_min_area}_os{best_open_size}_cs{best_close_size}"

    elif args.task == "task2":
        all_boxes, gt_dict, train_end, best_alpha, best_rho, ap50 = run_task2(args)
        desc = f"_a{best_alpha}_r{best_rho}"
    else:
        raise ValueError("Unknown task")

    if args.gif:
        prefix = f"detections_{args.task}_"
        prefix = prefix + desc
        file_name = prefix + f"_ap50_{ap50:.04f}.mp4"
        output_folder = os.path.join(args.task, 'results', 'videos')
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(args.task, 'results', 'videos', file_name)
        
        create_detection_video(
            args.video,
            all_boxes,
            gt_dict,
            train_end,
            output_path=output_path
        )

if __name__ == "__main__":
    main()