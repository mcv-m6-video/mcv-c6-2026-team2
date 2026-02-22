import argparse
from task1.task1 import run_task1
from utils import create_detection_gif

def main():

    parser = argparse.ArgumentParser(description="MCV Week1 Project")

    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--video", type=str, required=True,
                        help="Path to video")

    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to XML annotations")
    
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--train_ratio", type=float, default=0.25)

    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--min_area", type=int, default=1500)
    parser.add_argument("--open_size", type=int, default=3)
    parser.add_argument("--close_size", type=int, default=7)

    parser.add_argument("--gif", action="store_true", help="Generate detection GIF at the end")

    args = parser.parse_args()

    if args.task == "task1":
        test_frames, all_boxes, gt_dict, train_end = run_task1(args)

    elif args.task == "task2":
        print("Task2 not implemented yet")

    else:
        raise ValueError("Unknown task")

    if args.gif:
        create_detection_gif(
            test_frames=test_frames, 
            all_pred_boxes=all_boxes, 
            gt_per_frame=gt_dict, 
            train_end=train_end,
            output_path=f"{args.task}/outputs/detections_alpha_{args.alpha}.gif",
            max_frames=100
        )

if __name__ == "__main__":
    main()