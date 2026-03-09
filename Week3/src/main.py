import argparse

import yaml

from . import t11, t12, t21, t22


def parse_config(parser: argparse.ArgumentParser, config_file: str):
    """
    Overwrites the argument with those of the given config_file if any.
    """
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)
    parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    return args


def args_parser():
    """
    Parses the arguments of the script.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # General
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file with all the arguments configuration.",
    )
    parser.add_argument("--dataset_path", type=str, default="datasets/KITTI")
    parser.add_argument("--output_path", type=str, default="results/task11")

    # Task 1.1
    parser_11 = subparsers.add_parser("task11")
    parser_11.add_argument("--image_id", type=int, default=45)
    parser_11.add_argument("--num_iters", type=float, default=1)

    ## Pyflow
    parser_11.add_argument("--pf_alpha", type=float, default=0.012)
    parser_11.add_argument("--pf_ratio", type=float, default=0.75)
    parser_11.add_argument("--pf_minWidth", type=int, default=20)
    parser_11.add_argument("--pf_nOuterFPIters", type=int, default=7)
    parser_11.add_argument("--pf_nInnerFPIters", type=int, default=1)
    parser_11.add_argument("--pf_nSORIters", type=int, default=30)
    parser_11.add_argument(
        "--pf_colType",
        type=int,
        default=0,
        help="Color type. 0(default): RGB. 1: GRAY.",
    )

    ## Farneback
    parser_11.add_argument("--fb_pyrScale", type=float, default=0.5)
    parser_11.add_argument("--fb_levels", type=int, default=3)
    parser_11.add_argument("--fb_winSize", type=int, default=15)
    parser_11.add_argument("--fb_iters", type=int, default=3)
    parser_11.add_argument("--fb_polyN", type=int, default=5)
    parser_11.add_argument("--fb_polySigma", type=int, default=1.2)

    ## Perceiver
    parser_11.add_argument("--perc_path", type=str, default="deepmind/optical-flow-perceiver")

    ## MEMFOF
    parser_11.add_argument("--mf_path", type=str, default="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")

    parser_11.set_defaults(func=t11)

    # Task 1.2
    parser_12 = subparsers.add_parser("task12")
    parser_12.add_argument("--pingo", type=str, default="pingo")
    parser_12.set_defaults(func=t12)

    # Task 2.1
    parser_21 = subparsers.add_parser("task21")
    parser_21.add_argument("--tracker_name", type=str, default="kalmannof")
    parser_21.add_argument("--make_video", type=bool, action="store_true")
    parser_21.set_defaults(func=t21)

    # Task 2.2
    parser_22 = subparsers.add_parser("task22")
    parser_22.set_defaults(func=t22)

    args = parser.parse_args()

    if args.config:
        args = parse_config(parser, args.config)

    return args


def main(args):
    """
    Entry point of every task
    """
    args.func(args)


if __name__ == "__main__":
    args = args_parser()
    main(args)
