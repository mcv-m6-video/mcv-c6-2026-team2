import cv2
import numpy as np
from task1.task1 import load_video_frames, load_ground_truth, gt_to_coco, preds_to_coco, evaluate_coco, get_bounding_boxes, remove_nested_boxes, merge_overlapping_boxes

def segment_and_detect(test_frames, mu, sigma, roi, rho, args):
    all_pred_boxes = []
    
    current_mu = mu.copy()
    current_sigma = sigma.copy()

    kernel_open = np.ones((args.open_size, args.open_size), np.uint8)
    kernel_close = np.ones((args.close_size, args.close_size), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    for frame in test_frames:

        diff = np.abs(frame - current_mu)
        sigma_safe = np.maximum(current_sigma, 1e-6)

        mask_bool = (diff >= args.alpha * (sigma_safe + 2)) & roi
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
    mu = np.mean(train, axis=0)
    sigma = np.std(train, axis=0)
    roi = cv2.imread(
        args.video.replace("vdo.avi", "roi.jpg"),
        cv2.IMREAD_GRAYSCALE
    )
    roi = roi > 0

    all_pred_boxes = segment_and_detect(test, mu, sigma, roi, args.rho, args)

    gt_dict = load_ground_truth(args.annotations)
    
    gt_json = gt_to_coco(gt_dict, train_end, len(test))
    pred_json = preds_to_coco(all_pred_boxes, train_end)

    ap50 = evaluate_coco(gt_json, pred_json)

    print(f"\nFinal Adaptive AP50: {ap50:.4f}")

    return test, all_pred_boxes, gt_dict, train_end