import cv2
import csv
import numpy as np
from task1.task1 import load_video_frames, load_ground_truth, gt_to_coco, preds_to_coco, evaluate_coco, get_bounding_boxes, remove_nested_boxes, merge_overlapping_boxes
from itertools import product
import os

def segment_and_detect(test_frames, mu, sigma, roi, alpha, rho, args):
    all_pred_boxes = []
    
    current_mu = mu.copy()
    current_sigma = sigma.copy()

    kernel_open = np.ones((args.open_size, args.open_size), np.uint8)
    kernel_close = np.ones((args.close_size, args.close_size), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    for frame in test_frames:

        diff = np.abs(frame - current_mu)
        sigma_safe = np.maximum(current_sigma, 1e-6)

        mask_bool = (diff >= alpha * (sigma_safe + 2)) & roi
        mask_uint8 = mask_bool.astype(np.uint8) * 255

        mask_proc = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
        mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_CLOSE, kernel_close)
        mask_proc = cv2.dilate(mask_proc, kernel_dilate, iterations=1)

        boxes = get_bounding_boxes(mask_proc, args.min_area)
        boxes = remove_nested_boxes(boxes)
        boxes = merge_overlapping_boxes(boxes)
        all_pred_boxes.append(boxes)

        if rho > 0:
            # Only take into account background class
            background_mask = (~mask_bool) & roi
            
            # Update mu
            current_mu[background_mask] = (rho * frame[background_mask] + (1 - rho) * current_mu[background_mask])
            
            # Update sigma
            variance = current_sigma**2
            variance[background_mask] = (rho * (frame[background_mask] - current_mu[background_mask])**2 +  (1 - rho) * variance[background_mask])
            current_sigma = np.sqrt(variance)

    return all_pred_boxes


def run_task2(args):
    """
    Task 2: Adaptive Background Estimation using a Recursive Gaussian Model.
    The background parameters (mean and variance) are updated for pixels 
    classified as background to adapt to lighting changes.
    """
    print(f"Running Task 2: Adaptive Gaussian (alpha={args.alpha}, rho={args.rho}) ---")

    frames = load_video_frames(args.video)
    N = len(frames)
    train_end = int(0.25 * N)
    train = frames[:train_end]
    test = frames[train_end:]
    mu_init = np.mean(train, axis=0)
    sigma_init = np.std(train, axis=0)
    roi = cv2.imread(
        args.video.replace("vdo.avi", "roi.jpg"),
        cv2.IMREAD_GRAYSCALE
    )
    roi = roi > 0

    gt_dict = load_ground_truth(args.annotations)
    gt_json = gt_to_coco(gt_dict, train_end, len(test))

    best_alpha = None
    best_rho = None
    best_boxes = None
    best_ap50 = -1
    results_list = []

    for alpha, rho in product(args.alpha, args.rho):
        print(f"Testing alpha={alpha}, rho={rho}...")

        all_pred_boxes = segment_and_detect(test, mu_init, sigma_init, roi, alpha, rho, args)

        pred_json = preds_to_coco(all_pred_boxes, train_end)

        ap50 = evaluate_coco(gt_json, pred_json)

        results_list.append({'alpha': alpha, 'rho': rho, 'ap50': ap50})

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_alpha = alpha
            best_rho = rho
            best_boxes = all_pred_boxes

    output_dir = f"{args.task}/results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{os.path.basename(args.config).split('.')[0]}.csv")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'rho', 'ap50'])
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\nGrid Search Finished. Results saved to: {csv_path}")

    print(f"\nFinal Adaptive AP50: {best_ap50:.4f}")
    print(f"\nFinal Alpha: {best_alpha:.4f}")
    print(f"\nFinal Rho: {best_rho:.4f}")

    return test, best_boxes, gt_dict, train_end, best_alpha, best_rho