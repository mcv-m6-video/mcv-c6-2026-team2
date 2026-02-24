from __future__ import annotations

from typing import Tuple, List
import cv2
import csv
import numpy as np
from src.task1.task1 import Video, load_ground_truth, gt_to_coco, preds_to_coco, evaluate_coco, get_bounding_boxes, remove_nested_boxes, merge_overlapping_boxes, remove_shadows
from itertools import product
import os
import csv

import cv2
import numpy as np
from tqdm import tqdm

from task1.task1 import (Video,
                         load_ground_truth,
                         gt_to_coco,
                         preds_to_coco,
                         evaluate_coco,
                         get_bounding_boxes,
                         remove_nested_boxes,
                         merge_overlapping_boxes,
                         remove_shadows)


def threshold_foreground(
    frame_gray: np.ndarray,
    mean_gray: np.ndarray,
    std_gray: np.ndarray,
    alpha: float,
    roi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw foreground mask using background model thresholding.

    Args:
        frame_gray: Grayscale frame (H, W).
        mean_gray: Background mean (H, W).
        std_gray: Background std (H, W).
        alpha: Threshold multiplier.
        roi: Boolean ROI mask.

    Returns:
        Tuple containing:
            - mask_bool: Boolean foreground mask.
            - mask_uint8: Binary uint8 mask (0/255).
    """
    diff = np.abs(frame_gray - mean_gray)

    mask_bool = (diff >= alpha * (std_gray + 2)) & roi
    mask_uint8 = mask_bool.astype(np.uint8) * 255

    return mask_bool, mask_uint8


def apply_morphological_filtering(
    mask: np.ndarray,
    open_size: int,
    close_size: int
) -> np.ndarray:

    kernel_open = np.ones((open_size, open_size), np.uint8)
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def postprocess_mask(
    mask_uint8: np.ndarray,
    frame_hsv: np.ndarray,
    mean_hsv: np.ndarray,
    open_size: int,
    close_size: int,
) -> np.ndarray:
    """
    Apply shadow removal and morphological filtering to a binary mask.

    Args:
        mask_uint8: Binary mask (0/255).
        frame_hsv: Current HSV frame.
        mean_hsv: Background HSV mean.
        open_size: Kernel size for opening.
        close_size: Kernel size for closing.

    Returns:
        Processed binary mask.
    """

    mask = remove_shadows(frame_hsv, mean_hsv, mask_uint8)
    mask = apply_morphological_filtering(mask, open_size, close_size)

    return mask


def detect(mask_proc: np.ndarray, min_area: int) -> List[List[float]]:
    """
    Convert a processed binary mask into bounding boxes and apply box post-processing.

    Args:
        mask_proc: Binary mask (uint8, 0/255).
        min_area: Minimum connected-component area to keep.

    Returns:
        List of boxes in [x_min, y_min, x_max, y_max, score] format.
    """
    boxes = get_bounding_boxes(mask_proc, min_area)
    boxes = remove_nested_boxes(boxes)
    boxes = merge_overlapping_boxes(boxes)
    return boxes


def segment_and_detect(
    frame_gray: np.ndarray,
    frame_hsv: np.ndarray,
    mean_gray: np.ndarray,
    std_gray: np.ndarray,
    mean_hsv: np.ndarray,
    alpha: float,
    rho: float,
    open_size: int,
    close_size: int,
    min_area: int,
    roi: np.ndarray,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper: segments foreground and detects bounding boxes,
    optionally updating the background model.

    Returns:
        boxes, mean_gray, std_gray, mean_hsv
    """
    # segment foreground
    mask_bool, mask_uint8 = threshold_foreground(
        frame_gray=frame_gray,
        mean_gray=mean_gray,
        std_gray=std_gray,
        alpha=alpha,
        roi=roi,
    )

    mask_proc = postprocess_mask(
        mask_uint8=mask_uint8,
        frame_hsv=frame_hsv,
        mean_hsv=mean_hsv,
        open_size=open_size,
        close_size=close_size,
    )

    # detect bounding boxes
    boxes = detect(mask_proc=mask_proc, min_area=min_area)

    if rho > 0:
        # Only take into account background class
        background_mask = (~mask_bool) & roi

        # Update mu (in-place for mean arrays)
        mean_gray[background_mask] = (
            rho * frame_gray[background_mask] +
            (1 - rho) * mean_gray[background_mask]
        )
        expanded_mask = np.repeat(background_mask[..., np.newaxis], 3, axis=-1)
        mean_hsv[expanded_mask] = (
            rho * frame_hsv[expanded_mask] +
            (1 - rho) * mean_hsv[expanded_mask]
        )

        # Update sigma (returns new std_gray)
        variance = std_gray**2
        variance[background_mask] = (
            rho * (frame_gray[background_mask] -
                   mean_gray[background_mask]) ** 2
            + (1 - rho) * variance[background_mask]
        )
        std_gray = np.sqrt(variance)

    return boxes, mean_gray, std_gray, mean_hsv


def run_task2(args):
    """
    Task 2: Adaptive Background Estimation using a Recursive Gaussian Model.
    The background parameters (mean and variance) are updated for pixels 
    classified as background to adapt to lighting changes.
    """
    print(
        f"Running Task 2: Adaptive Gaussian (alpha={args.alpha}, rho={args.rho}) ---")

    video = Video(args.video)
    train_size = int(video.num_frames * 0.25)
    mean_gray = None
    prev_mean_gray = None
    mean_hsv = None
    std_gray = None
    roi = cv2.imread(
        args.video.replace("vdo.avi", "roi.jpg"),
        cv2.IMREAD_GRAYSCALE
    )
    roi = roi > 0

    gt_dict = load_ground_truth(args.annotations)
    gt_json = gt_to_coco(gt_dict, train_size, video.num_frames - train_size)

    do_grid = (len(args.alpha) > 1) or (len(args.rho) > 1)
    save_results = bool(getattr(args, "config", None)) and do_grid

    best_alpha = None
    best_rho = None
    best_boxes = None
    best_ap50 = -1
    results_list = [] if save_results else None

    for alpha, rho, min_area, open_size, close_size in product(args.alpha, args.rho, args.min_area, args.open_size, args.close_size):
        print(f"Testing alpha={alpha}, rho={rho}...")
        video.reset()
        all_pred_boxes = []

        for idx, (frame_gray, frame_hsv) in tqdm(enumerate(video.get_next_frame()), total=video.num_frames):
            if idx < train_size:
                if idx == 0:
                    mean_gray = frame_gray
                    std_gray = np.zeros_like(mean_gray)
                    prev_mean_gray = mean_gray
                    mean_hsv = frame_hsv
                else:
                    mean_gray = prev_mean_gray + \
                        (frame_gray - prev_mean_gray) / idx
                    std_gray = std_gray + \
                        (frame_gray - prev_mean_gray) * (frame_gray - mean_gray)
                    prev_mean_gray = mean_gray
                    mean_hsv = mean_hsv + (frame_hsv - mean_hsv) / idx
            else:
                if idx == train_size:
                    std_gray = np.sqrt(std_gray / (train_size - 2))

                boxes, mean_gray, std_gray, mean_hsv = segment_and_detect(
                    frame_gray, frame_hsv, mean_gray, std_gray, mean_hsv, alpha, rho, open_size, close_size, min_area, roi)
                all_pred_boxes.append(boxes)

        pred_json = preds_to_coco(all_pred_boxes, train_size)

        ap50 = evaluate_coco(gt_json, pred_json)

        if save_results:
            results_list.append({'alpha': alpha, 'rho': rho, 'ap50': ap50})

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_alpha = alpha
            best_rho = rho
            best_boxes = all_pred_boxes

    if save_results:
        output_dir = f"{args.task}/results"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(
            output_dir, f"{os.path.basename(args.config).split('.')[0]}.csv")

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['alpha', 'rho', 'ap50'])
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\nGrid Search Finished. Results saved to: {csv_path}")

    print(f"\nFinal Adaptive AP50: {best_ap50:.4f}")
    print(f"\nFinal Alpha: {best_alpha:.4f}")
    print(f"\nFinal Rho: {best_rho:.4f}")

    video.close()

    return best_boxes, gt_dict, train_size, best_alpha, best_rho, best_ap50
