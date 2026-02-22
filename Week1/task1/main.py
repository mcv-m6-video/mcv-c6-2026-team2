import argparse
from task1 import run_task1

def main():

    parser = argparse.ArgumentParser(description="MCV Week1 Project")

    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--video", type=str, required=True,
                        help="Path to video")

    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to XML annotations")

    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--min_area", type=int, default=1500)
    parser.add_argument("--open_size", type=int, default=3)
    parser.add_argument("--close_size", type=int, default=7)

    args = parser.parse_args()

    if args.task == "task1":
        run_task1(args)

    elif args.task == "task2":
        print("Task2 not implemented yet")

    else:
        raise ValueError("Unknown task")

if __name__ == "__main__":
    main()