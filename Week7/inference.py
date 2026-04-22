#!/usr/bin/env python3
"""
Inference script for action spotting models.
Only runs inference/evaluation.
"""

import argparse
import torch
import os
import numpy as np
import random
from tabulate import tabulate
import sys

from util.io import load_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model import get_model
from main_spotting import compute_ap10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nms_window', type=int, default=5)
    return parser.parse_args()

def update_args(args, config):
    args.model_type        = config['model_type']
    args.frame_dir         = config['frame_dir']
    args.save_dir          = config['save_dir'] + '/' + args.model
    args.store_dir         = config['save_dir'] + '/' + "splits"
    args.labels_dir        = config['labels_dir']
    args.store_mode        = config['store_mode']
    args.task              = config['task']
    args.batch_size        = config['batch_size']
    args.clip_len          = config['clip_len']
    args.dataset           = config['dataset']
    args.epoch_num_frames  = config['epoch_num_frames']
    args.feature_arch      = config['feature_arch']
    args.learning_rate     = config['learning_rate']
    args.num_classes       = config['num_classes']
    args.num_epochs        = config['num_epochs']
    args.warm_up_epochs    = config['warm_up_epochs']
    args.only_test         = config['only_test']
    args.device            = config['device']
    args.num_workers       = config['num_workers']
    args.patience          = config['patience']

    # Optional for T-DEED-like model
    args.temporal_arch       = config.get('temporal_arch', 'ed_sgp_mixer')
    args.radi_displacement   = config.get('radi_displacement', 0)
    args.n_layers            = config.get('n_layers', 2)
    args.sgp_ks              = config.get('sgp_ks', 9)
    args.sgp_r               = config.get('sgp_r', 4)
    args.crop_dim            = config.get('crop_dim', -1)

    # Optional for soft labels
    args.soft_labels         = config.get('soft_labels', False)
    args.soft_sigma          = config.get('soft_sigma', 2.0)

    return args


def main(args):
    print("Setting seed to:", args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = load_json("config/" + args.model + ".json")
    args = update_args(args, config)
    args.num_workers = min(args.num_workers, 2)

    classes, _, _, test_data = get_datasets(args)

    if args.store_mode == "store":
        sys.exit("Datasets stored! Re-run with store_mode=load.")

    print("Datasets loaded correctly.")

    model = get_model(args=args)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=model.device)
    model.load(checkpoint)

    print("START INFERENCE")

    map_score, ap_score, _ = evaluate(
        model,
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nms_window=args.nms_window,
    )

    table = [[name, f"{ap_score[i] * 100:.2f}"] for i, name in enumerate(classes.keys())]
    print(tabulate(table, ["Class", "AP"], tablefmt="grid"))

    ap10 = compute_ap10(classes, ap_score)
    print(tabulate(
        [["AP10", f"{ap10 * 100:.2f}"], ["AP12", f"{map_score * 100:.2f}"]],
        ["Metric", "Value"],
        tablefmt="grid"
    ))

    print("CORRECTLY FINISHED INFERENCE")


if __name__ == "__main__":
    main(get_args())