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
import cv2
import imageio.v2 as imageio

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
    parser.add_argument('--save_qualitative', action='store_true',
                    help='Save qualitative clip examples as mp4/gif')
    parser.add_argument('--num_qualitative', type=int, default=10,
                        help='Number of qualitative examples to save')
    parser.add_argument('--qualitative_dir', type=str, default='qualitative_examples',
                        help='Output directory for qualitative examples')
    parser.add_argument('--save_gif', action='store_true',
                        help='Also save gif version')
    parser.add_argument('--pred_threshold', type=float, default=0.5,
                        help='Threshold for positive predicted labels')
    parser.add_argument('--only_event_clips', action='store_true',
                        help='Save only clips that contain GT events')
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

def invert_class_dict(classes):
    """
    classes: dict(name -> idx starting at 1)
    returns: dict(idx starting at 0 -> name)
    """
    return {v - 1: k for k, v in classes.items()}


def tensor_frame_to_bgr(frame_tensor):
    """
    frame_tensor: torch.Tensor [C, H, W]
    returns: np.ndarray [H, W, 3] in BGR
    """
    frame = frame_tensor.detach().cpu().numpy()
    frame = np.transpose(frame, (1, 2, 0))  # C,H,W -> H,W,C
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def get_gt_text(gt_vector, idx_to_class):
    gt_vector = np.asarray(gt_vector)
    positives = [idx_to_class[i] for i, v in enumerate(gt_vector) if v == 1]
    if len(positives) == 0:
        return "NO EVENT"
    return ", ".join(positives)


def get_pred_texts(pred_scores, idx_to_class, threshold=0.5):
    pred_scores = np.asarray(pred_scores)

    positive_items = [
        (idx_to_class[i], float(s))
        for i, s in enumerate(pred_scores)
        if s >= threshold
    ]

    if len(positive_items) == 0:
        pred_text = "NONE"
        best_idx = int(np.argmax(pred_scores))
        best_text = f"{idx_to_class[best_idx]} ({pred_scores[best_idx]:.2f})"
    else:
        pred_text = ", ".join([f"{name} ({score:.2f})" for name, score in positive_items])
        best_idx = int(np.argmax(pred_scores))
        best_text = f"{idx_to_class[best_idx]} ({pred_scores[best_idx]:.2f})"

    return pred_text, best_text

def render_clip_with_labels(frames, gt_text, pred_text, threshold, out_path, fps=6, save_gif=False):
    """
    frames: torch.Tensor [T, C, H, W]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    T, C, H, W = frames.shape
    header_h = 80
    out_h = H + header_h
    out_w = W

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (out_w, out_h)
    )

    gif_frames = []

    for t in range(T):
        img = tensor_frame_to_bgr(frames[t])

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[header_h:, :, :] = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        cv2.putText(canvas, f"Frame {t+1}/{T}", (10, 25),
                    font, font_scale, (255, 255, 255), thickness)

        cv2.putText(canvas, f"GT: {gt_text}", (10, 50),
                    font, font_scale, (0, 255, 0), thickness)

        cv2.putText(canvas, f"Pred: {pred_text}", (10, 75),
                    font, font_scale, (0, 215, 255), thickness)

        writer.write(canvas)

        if save_gif:
            gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    writer.release()

    if save_gif:
        gif_path = os.path.splitext(out_path)[0] + '.gif'
        imageio.mimsave(gif_path, gif_frames, fps=fps, loop=0)

def save_qualitative_examples(model, dataset, classes, out_dir,
                              num_examples=10, pred_threshold=0.5,
                              only_event_clips=False, save_gif=False):
    idx_to_class = invert_class_dict(classes)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    attempts = 0
    max_attempts = max(num_examples * 10, 50)

    print('START SAVING QUALITATIVE EXAMPLES')

    while saved < num_examples and attempts < max_attempts:
        sample = dataset[0]   # da igual el índice: tu dataset devuelve uno aleatorio
        attempts += 1

        frames = sample['frame']              # [T, C, H, W]
        gt = sample['label']                  # [C]
        contains_event = sample['contains_event']

        if only_event_clips and contains_event == 0:
            continue

        pred_scores = model.predict(frames)[0]   # [C]

        gt_text = get_gt_text(gt, idx_to_class)
        pred_text, best_text = get_pred_texts(
            pred_scores, idx_to_class, threshold=pred_threshold
        )

        safe_gt = gt_text.replace(" ", "_").replace(",", "").replace("/", "-")
        safe_pred = best_text.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")

        out_path = os.path.join(
            out_dir,
            f'clip_{saved:03d}_gt_{safe_gt}_pred_{safe_pred}.mp4'
        )

        render_clip_with_labels(
            frames=frames,
            gt_text=gt_text,
            pred_text=pred_text,
            threshold=pred_threshold,
            out_path=out_path,
            fps=6,
            save_gif=save_gif
        )

        print(f'Saved {saved + 1}/{num_examples}: {out_path}')
        saved += 1

    print('FINISHED SAVING QUALITATIVE EXAMPLES')

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

    if args.save_qualitative:
        save_qualitative_examples(
            model=model,
            dataset=test_data,
            classes=classes,
            out_dir=args.qualitative_dir,
            num_examples=args.num_qualitative,
            pred_threshold=args.pred_threshold,
            only_event_clips=args.only_event_clips,
            save_gif=args.save_gif
        )

    print('CORRECTLY FINISHED INFERENCE')

if __name__ == '__main__':
    main(get_args())