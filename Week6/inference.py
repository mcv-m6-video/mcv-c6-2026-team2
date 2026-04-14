#!/usr/bin/env python3
"""
Inference script for action spotting models.

- Computes metrics exactly as evaluate() does.
- Optionally saves qualitative GIFs per dataset clip.
- Qualitative overlays use GLOBAL post-NMS predictions (same level as evaluation),
  filtered to the current clip.

Overlay format:
    f: 18/50
    GT: G@13, FK@31
    PR: G@16(score), FK@44(score)

Text is automatically wrapped into multiple lines so it fits.
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
from util.eval_spotting import evaluate, collect_predictions_and_targets
from dataset.datasets import get_datasets
from dataset.frame import FPS_SN
from model import get_model
from main_spotting import compute_ap10


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True,
                        help='Name of the config/model experiment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pt)')
    parser.add_argument('--seed', type=int, default=1)

    # Qualitative
    parser.add_argument('--save_qualitative', action='store_true',
                        help='Save qualitative GIFs using global post-NMS predictions')
    parser.add_argument('--num_qualitative', type=int, default=10,
                        help='Number of clip GIFs to save')
    parser.add_argument('--qualitative_dir', type=str, default='qualitative_examples',
                        help='Output directory for qualitative GIFs')
    parser.add_argument('--only_event_clips', action='store_true',
                        help='If set, save only clips containing GT events')
    parser.add_argument('--nms_window', type=int, default=5,
                        help='NMS window used for evaluation/global predictions')
    parser.add_argument('--gif_fps', type=int, default=6,
                        help='FPS for saved GIFs')

    return parser.parse_args()


def update_args(args, config):
    args.model_type = config['model_type']
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
    args.patience = config['patience']
    return args


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


def sanitize_text(text):
    return (
        str(text).replace(" ", "_")
                 .replace(",", "")
                 .replace("/", "-")
                 .replace("(", "")
                 .replace(")", "")
                 .replace(":", "")
    )


def build_class_abbrev(classes):
    """
    classes: dict(name -> idx starting at 1)

    Returns:
        idx_to_abbrev_pred: dict(idx starting at 0 -> abbrev) for predictions without background
        idx_to_abbrev_gt: dict(idx starting at 1 -> abbrev) for GT labels with background=0 separately
    """
    custom = {
        "PASS": "P",
        "DRIVE": "D",
        "HEADER": "H",
        "HIGH PASS": "HP",
        "OUT": "O",
        "CROSS": "C",
        "THROW IN": "TI",
        "SHOT": "S",
        "BALL PLAYER BLOCK": "BPB",
        "PLAYER SUCCESSFUL TACKLE": "PST",
        "FREE KICK": "FK",
        "GOAL": "G"
    }

    idx_to_abbrev_pred = {}
    idx_to_abbrev_gt = {}

    used = set()

    for class_name, idx1 in classes.items():
        if class_name in custom:
            abbr = custom[class_name]
        else:
            words = class_name.replace("-", " ").split()
            if len(words) == 1:
                abbr = words[0][:3].upper()
            else:
                abbr = "".join(w[0].upper() for w in words[:3])

        original = abbr
        k = 2
        while abbr in used:
            abbr = f"{original}{k}"
            k += 1
        used.add(abbr)

        idx_to_abbrev_gt[idx1] = abbr
        idx_to_abbrev_pred[idx1 - 1] = abbr

    return idx_to_abbrev_pred, idx_to_abbrev_gt


def extract_gt_events_from_clip_labels(gt_labels, idx_to_abbrev_gt):
    """
    gt_labels: [T] values in {0..C}
    Returns:
        [{"abbr": "...", "t": local_t}, ...]
    """
    gt_labels = np.asarray(gt_labels)
    events = []

    for t, label in enumerate(gt_labels):
        label = int(label)
        if label != 0:
            events.append({
                "abbr": idx_to_abbrev_gt[label],
                "t": t,
            })

    return events


def extract_pred_events_from_global_scores(scores_nms_video, clip_start, clip_len, idx_to_abbrev_pred):
    """
    scores_nms_video: [T_video, C], post-NMS global predictions
    Returns:
        [{"abbr": "...", "t": local_t, "score": float}, ...]
    """
    events = []

    video_len = scores_nms_video.shape[0]
    num_classes = scores_nms_video.shape[1]

    for local_t in range(clip_len):
        global_t = clip_start + local_t
        if global_t < 0 or global_t >= video_len:
            continue

        for c in range(num_classes):
            score = float(scores_nms_video[global_t, c])
            if score >= 0:
                events.append({
                    "abbr": idx_to_abbrev_pred[c],
                    "t": local_t,
                    "score": score,
                })

    return events

def summarize_gt_events(gt_events):
    """
    gt_events: [{"abbr": "...", "t": ...}, ...]
    """
    if len(gt_events) == 0:
        return "NONE"
    return ", ".join([f"{e['abbr']}@{e['t']}" for e in gt_events])


def summarize_pred_events(pred_events):
    """
    pred_events: [{"abbr": "...", "t": ..., "score": ...}, ...]
    """
    if len(pred_events) == 0:
        return "NONE"

    pred_events = sorted(pred_events, key=lambda e: e["score"], reverse=True)

    return ", ".join([
        f"{e['abbr']}@{e['t']}({e['score']:.2f})"
        for e in pred_events
    ])

def clip_has_gt(gt_events):
    return len(gt_events) > 0


def clip_has_pred(pred_events):
    return len(pred_events) > 0

def wrap_text_for_width(text, max_width_px, font, font_scale, thickness):
    """
    Greedy word wrap based on rendered pixel width.
    """
    if text == "":
        return [""]

    words = text.split(" ")
    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = current + " " + word
        (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if w <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def render_spotting_clip_gif(
    frames,
    out_path,
    gt_summary,
    pred_summary,
    fps=6
):
    """
    frames: torch.Tensor [T, C, H, W]

    Overlay:
      f: i/T
      GT: ...
      PR: ...

    GT and PR are fixed summaries for the whole clip.
    They are automatically wrapped so they fit the frame width.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    T, C, H, W = frames.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    gt_line = f"GT: {gt_summary}"
    pred_line = f"PR: {pred_summary}"

    max_text_width = W - 20
    gt_lines = wrap_text_for_width(gt_line, max_text_width, font, font_scale, thickness)
    pred_lines = wrap_text_for_width(pred_line, max_text_width, font, font_scale, thickness)

    num_header_lines = 1 + len(gt_lines) + len(pred_lines)
    line_h = 26
    top_margin = 10
    bottom_margin = 10
    header_h = top_margin + num_header_lines * line_h + bottom_margin

    out_h = H + header_h
    out_w = W

    gif_frames = []

    for local_t in range(T):
        img = tensor_frame_to_bgr(frames[local_t])

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[header_h:, :, :] = img

        y = 24
        cv2.putText(canvas, f"f: {local_t+1}/{T}", (10, y),
                    font, font_scale, (255, 255, 255), thickness)
        y += line_h

        for line in gt_lines:
            cv2.putText(canvas, line, (10, y),
                        font, font_scale, (0, 255, 0), thickness)
            y += line_h

        for line in pred_lines:
            cv2.putText(canvas, line, (10, y),
                        font, font_scale, (0, 215, 255), thickness)
            y += line_h

        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    imageio.mimsave(out_path, gif_frames, fps=fps, loop=0)


def save_qualitative_examples_from_global_predictions(
    dataset,
    scores_nms_dict,
    classes,
    out_dir,
    num_examples=10,
    only_event_clips=True,
    gif_fps=6
):
    """
    Saves GIFs for dataset clips using GLOBAL post-NMS predictions already computed.

    Logic:
      - If only_event_clips=True: save only clips with GT events.
      - Else: save clips with GT or predicted events.
    """
    os.makedirs(out_dir, exist_ok=True)

    idx_to_abbrev_pred, idx_to_abbrev_gt = build_class_abbrev(classes)

    print('START SAVING QUALITATIVE GIFS')
    saved = 0

    for i in range(len(dataset)):
        sample = dataset[i]

        frames = sample['frame']              # [T, C, H, W]
        gt_labels = sample['label']           # [T]
        video = sample['video']
        clip_start = int(sample['start'])

        if torch.is_tensor(gt_labels):
            gt_labels_np = gt_labels.cpu().numpy()
        else:
            gt_labels_np = np.asarray(gt_labels)

        gt_events = extract_gt_events_from_clip_labels(
            gt_labels_np,
            idx_to_abbrev_gt
        )

        scores_nms_video = scores_nms_dict[video]
        pred_events = extract_pred_events_from_global_scores(
            scores_nms_video=scores_nms_video,
            clip_start=clip_start,
            clip_len=frames.shape[0],
            idx_to_abbrev_pred=idx_to_abbrev_pred
        )

        has_gt = clip_has_gt(gt_events)
        has_pred = clip_has_pred(pred_events)

        if only_event_clips:
            if not has_gt:
                continue
        else:
            if not (has_gt or has_pred):
                continue

        gt_summary = summarize_gt_events(gt_events)
        pred_summary = summarize_pred_events(pred_events)

        safe_gt = sanitize_text(gt_summary[:40])
        safe_pred = sanitize_text(pred_summary[:40])

        out_path = os.path.join(
            out_dir,
            f'clip_{saved:03d}_gt_{safe_gt}_pred_{safe_pred}.gif'
        )

        render_spotting_clip_gif(
            frames=frames,
            out_path=out_path,
            gt_summary=gt_summary,
            pred_summary=pred_summary,
            fps=gif_fps
        )

        print(f'Saved {saved + 1}/{num_examples}: {out_path}')
        saved += 1

        if saved >= num_examples:
            break

    print('FINISHED SAVING QUALITATIVE GIFS')


def main(args):
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    args.num_workers = min(args.num_workers, 2)

    classes, _, _, _, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    model = get_model(args=args)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    print(f'Loading checkpoint from: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=model.device)
    model.load(checkpoint)

    print('START INFERENCE')

    # Metrics
    map_score, ap_score, _ = evaluate(
        model,
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nms_window=args.nms_window
    )

    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i] * 100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    ap10 = compute_ap10(classes, ap_score)

    avg_table = [
        ["AP10", f"{ap10 * 100:.2f}"],
        ["AP12", f"{map_score * 100:.2f}"]
    ]

    headers = ["Metric", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    # Qualitative
    if args.save_qualitative:
        _, _, _, _, scores_nms_dict = collect_predictions_and_targets(
            model=model,
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            nms_window=args.nms_window,
            nms_threshold=0.05
        )

        save_qualitative_examples_from_global_predictions(
            dataset=test_data,
            scores_nms_dict=scores_nms_dict,
            classes=classes,
            out_dir=args.qualitative_dir,
            num_examples=args.num_qualitative,
            only_event_clips=args.only_event_clips,
            gif_fps=args.gif_fps
        )

    print('CORRECTLY FINISHED INFERENCE')


if __name__ == '__main__':
    main(get_args())