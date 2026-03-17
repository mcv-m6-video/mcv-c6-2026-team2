
import argparse

from .tasks import evaluate, match, track, train_match, evaluate_matcher


def args_parser():
    dataset_parser = argparse.ArgumentParser(add_help=False)

    dataset_parser.add_argument(
        "--dataset_root", type=str, default="datasets/AI_CITY_CHALLENGE_2022_TRAIN")
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

    tracking_subparser = subparsers.add_parser(
        "tracking", parents=[det_model_parser, of_model_parser])
    tracking_subparser.set_defaults(func=track)

    train_matcher_subparser = subparsers.add_parser(
        "train_matcher", parents=[match_model_parser])
    train_matcher_subparser.set_defaults(func=train_match)

    matching_subparser = subparsers.add_parser("matching", parents=[match_model_parser])
    matching_subparser.add_argument("--tracking_file", type=str, default="mtsc/mtsc_deepsort_mask_rcnn.txt")
    matching_subparser.add_argument("--output_folder", type=str, default="results")
    matching_subparser.set_defaults(func=match)

    evaluation_subparser = subparsers.add_parser(
        "evaluate", parents=[eval_subparser])
    evaluation_subparser.add_argument("--gt", required=True, type=str)
    evaluation_subparser.add_argument("--pred", required=True, type=str)
    evaluation_subparser.add_argument("--dstype", type=str, default="train")
    evaluation_subparser.add_argument(
        "--roidir", type=str, default="datasets/AI_CITY_CHALLENGE_2022_TRAIN")
    evaluation_subparser.add_argument("-m", "--mread", action="store_true")
    evaluation_subparser.set_defaults(func=evaluate)

    evaluation_subparser = subparsers.add_parser(
        "evaluate_matcher", parents=[match_model_parser, eval_subparser])
    evaluation_subparser.set_defaults(func=evaluate_matcher)

    args = main_parser.parse_args()

    return args


def main(args):
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
