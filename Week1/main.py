import argparse
from task1.task1 import run_task1
from task2.task2 import run_task2
from utils import create_detection_gif
import yaml

def main():

    parser = argparse.ArgumentParser(description="MCV Week1 Project")

    parser.add_argument("--task", type=str, default=None)

    parser.add_argument("--video", type=str, default=None,
                        help="Path to video")

    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to XML annotations")
    
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--train_ratio", type=float, default=0.25)

    parser.add_argument("--alpha", nargs='+', type=float, default=[3.0])
    parser.add_argument("--min_area", type=int, default=1500)
    parser.add_argument("--open_size", type=int, default=3)
    parser.add_argument("--close_size", type=int, default=7)
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

    if args.task == "task1":
        test_frames, all_boxes, gt_dict, train_end = run_task1(args)

    elif args.task == "task2":
        test_frames, all_boxes, gt_dict, train_end, alpha, rho = run_task2(args)

    else:
        raise ValueError("Unknown task")

    if args.gif:
        create_detection_gif(
            test_frames=test_frames, 
            all_pred_boxes=all_boxes, 
            gt_per_frame=gt_dict, 
            train_end=train_end,
            output_path=f"{args.task}/outputs/detections_alpha_{alpha}_{rho}.gif",
            max_frames=100
        )

if __name__ == "__main__":
    main()