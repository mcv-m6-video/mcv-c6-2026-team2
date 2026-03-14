
import argparse

from .tasks import evaluate, match, track, train_match


def args_parser():
    dataset_parser = argparse.ArgumentParser(add_help=False)

    dataset_parser.add_argument("--dataset_root", type=str, default="datasets/AI_CITY_CHALLENGE_2022_TRAIN")
    dataset_parser.add_argument("--seq", type=str, default="S01")

    det_model_parser = argparse.ArgumentParser(add_help=False)
    det_model_parser.add_argument("--det_checkpoint", type=str, default="checkpoints/fasterrcnn_faster-rcnn_best.pth")

    of_model_parser = argparse.ArgumentParser(add_help=False)
    of_model_parser.add_argument("--of_checkpoint", type=str, default="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")

    match_model_parser = argparse.ArgumentParser(add_help=False)
    match_model_parser.add_argument("--match_checkpoint", type=str, default="checkpoints/best_matcher.pt")

    main_parser = argparse.ArgumentParser(parents=[dataset_parser])
    subparsers = main_parser.add_subparsers(required=True)

    tracking_subparser = subparsers.add_parser("tracking", parents=[det_model_parser, of_model_parser])
    tracking_subparser.set_defaults(func=track)

    train_matcher_subparser = subparsers.add_parser("train_matcher", parents=[match_model_parser])
    train_matcher_subparser.set_defaults(func=train_match)

    matching_subparser = subparsers.add_parser("matching", parents=[match_model_parser])
    matching_subparser.set_defaults(func=match)

    evaluation_subparser = subparsers.add_parser("evaluate")
    evaluation_subparser.set_defaults(func=evaluate)

    args = main_parser.parse_args()

    return args


def main(args):
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)