#!/usr/bin/env python3
"""
File containing the inference script for classification models.
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from tabulate import tabulate
import sys

# Local imports
from util.io import load_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model import get_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Name of the config/model experiment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pt)')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    args.model_type = config.get('model_type', 'baseline')
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']

    # Optional
    args.patience = config.get('patience', None)

    return args


def main(args):
    # Set seed
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # For inference, safer memory usage
    args.num_workers = min(args.num_workers, 2)

    # Get datasets
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    # Build model
    model = get_model(args=args)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    print(f'Loading checkpoint from: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=model.device)
    model.load(checkpoint)

    # Run inference
    print('START INFERENCE')
    ap_score = evaluate(model, test_data)

    # Per-class AP table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i] * 100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    # Calculate AP12 and AP10
    ap12 = np.mean(ap_score)

    exclude = {"FREE KICK", "GOAL"}
    ap10_scores = [
        ap_score[i] for i, class_name in enumerate(classes.keys())
        if class_name not in exclude
    ]
    ap10 = np.mean(ap10_scores)

    avg_table = [
        ["AP10", f"{ap10 * 100:.2f}"],
        ["AP12", f"{ap12 * 100:.2f}"]
    ]

    headers = ["Metric", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    print('CORRECTLY FINISHED INFERENCE')


if __name__ == '__main__':
    main(get_args())