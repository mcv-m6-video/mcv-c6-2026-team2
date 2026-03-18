
import argparse

from .models.matcher import DEFAULT_SIMILARITY_THRESHOLD
from .tasks import evaluate, match, track, train_match, evaluate_matcher


def args_parser():
    dataset_parser = argparse.ArgumentParser(add_help=False)

    dataset_parser.add_argument(
        "--dataset_root", type=str, default="datasets/AI_CITY_CHALLENGE_2022_TRAIN")
    dataset_parser.add_argument("--detections_root", type=str, default="data")
    dataset_parser.add_argument("--seq", type=str, default="S01")

    det_model_parser = argparse.ArgumentParser(add_help=False)
    det_model_parser.add_argument(
        "--det_checkpoint", type=str, default="checkpoints/fasterrcnn_faster-rcnn_best.pth")

    of_model_parser = argparse.ArgumentParser(add_help=False)
    of_model_parser.add_argument(
        "--of_checkpoint", type=str, default="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")

    match_model_parser = argparse.ArgumentParser(add_help=False)
    match_model_parser.add_argument(
        "--match_checkpoint", type=str, default="checkpoints/best_matcher.pth")

    eval_subparser = argparse.ArgumentParser(add_help=False)
    eval_subparser.add_argument("--seq", type=str, default="S03")
    eval_subparser.add_argument(
        "--eval_output_dir", type=str, default="eval_output")

    main_parser = argparse.ArgumentParser(parents=[dataset_parser])
    subparsers = main_parser.add_subparsers(required=True)

    tracking_subparser = subparsers.add_parser("tracking", parents=[det_model_parser]) #, of_model_parser])
    tracking_subparser.add_argument("--iou_threshold", type=float, default=0.4)
    tracking_subparser.add_argument("--dup_iou_threshold", type=float, default=0.5)
    tracking_subparser.add_argument("--max_age", type=int, default=10)
    tracking_subparser.add_argument("--conf_threshold", type=float, default=0.6)
    tracking_subparser.add_argument("--output_path", type=str, default="data")
    tracking_subparser.set_defaults(func=track)

    # train matching task
    train_matcher_subparser = subparsers.add_parser(
        "train_matcher", parents=[match_model_parser])
    
    train_matcher_subparser.set_defaults(func=train_match)

    # matching task
    matching_subparser = subparsers.add_parser(
        "matching", parents=[match_model_parser])
    matching_subparser.add_argument(
        "--match_threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    matching_subparser.add_argument(
        "--tracking_file", type=str, default="mtsc/mtsc_deepsort_mask_rcnn.txt")
    matching_subparser.add_argument(
        "--output_folder", type=str, default="results")
    matching_subparser.set_defaults(func=match)

    # evaluation task
    evaluation_subparser = subparsers.add_parser(
        "evaluate", parents=[eval_subparser])
    evaluation_subparser.add_argument(
        "--gt", type=str, default="data/gt/gt_S03.txt")
    evaluation_subparser.add_argument("--pred", required=True, type=str)
    evaluation_subparser.add_argument("--dstype", type=str, default="train")
    evaluation_subparser.add_argument(
        "--roidir", type=str, default="datasets/AI_CITY_CHALLENGE_2022_TRAIN")
    evaluation_subparser.add_argument("-m", "--mread", action="store_true")
    evaluation_subparser.add_argument(
        "--render_videos",
        action="store_true",
        help="Render annotated per-camera videos for the evaluated predictions.",
    )
    evaluation_subparser.set_defaults(func=evaluate)

    # evaluation of the matcher independently using the GT
    evaluation_matcher_subparser = subparsers.add_parser(
        "evaluate_matcher", parents=[match_model_parser, eval_subparser])
    evaluation_matcher_subparser.add_argument(
        "--threshold_start", type=float, default=0.60)
    evaluation_matcher_subparser.add_argument(
        "--threshold_end", type=float, default=0.90)
    evaluation_matcher_subparser.add_argument(
        "--threshold_step", type=float, default=0.05)
    evaluation_matcher_subparser.add_argument(
        "--render_threshold", type=float, default=None)
    evaluation_matcher_subparser.add_argument(
        "--train_sequences",
        type=str,
        default="",
        help="Comma-separated sequences used as the training pool when reconstructing train/val identity splits.",
    )
    evaluation_matcher_subparser.add_argument(
        "--val_sequences",
        type=str,
        default="",
        help="Comma-separated dedicated validation sequences, if your training used them.",
    )
    evaluation_matcher_subparser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation identity ratio used during training when no dedicated validation sequences were provided.",
    )
    evaluation_matcher_subparser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used to reconstruct the train/val identity split.",
    )
    evaluation_matcher_subparser.add_argument(
        "--split_subset",
        type=str,
        choices=["all", "train", "val"],
        default="all",
        help="Subset of the training-time split to evaluate: full sequences, only train identities, or only val identities.",
    )
    evaluation_matcher_subparser.set_defaults(func=evaluate_matcher)

    args = main_parser.parse_args()

    return args


def main(args):
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
