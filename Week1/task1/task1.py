import cv2
import numpy as np
import json
import csv
from itertools import product
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import xml.etree.ElementTree as ET
import imageio
from tqdm import tqdm
import os

# --------------------------------------------------
# Video loading
# --------------------------------------------------

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.reset()

    def get_next_frame(self):
        for _ in range(self.num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            yield np.astype(gray, np.float32), np.astype(hsv, np.float32)
        
    def close(self):
        self.cap.release()

    def reset(self):
        if self.cap:
            self.close()
        self.cap = cv2.VideoCapture(self.video_path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# --------------------------------------------------
# Ground truth
# --------------------------------------------------

def load_ground_truth(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt = {}

    for track in root.findall("track"):
        for box in track.findall("box"):

            frame = int(box.get("frame"))
            outside = int(box.get("outside"))

            if outside == 1:
                continue

            parked = False
            for attr in box.findall("attribute"):
                if attr.get("name") == "parked" and attr.text == "true":
                    parked = True

            if parked:
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            if frame not in gt:
                gt[frame] = []

            gt[frame].append([xtl, ytl, xbr, ybr])

    return gt


# --------------------------------------------------
# Post-processing
# --------------------------------------------------

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    union = areaA + areaB - interArea

    return interArea / (union + 1e-6)


def remove_nested_boxes(boxes):

    keep = []

    for i, boxA in enumerate(boxes):

        x1A, y1A, x2A, y2A = boxA[:4]
        areaA = (x2A - x1A) * (y2A - y1A)

        nested = False

        for j, boxB in enumerate(boxes):
            if i == j:
                continue

            x1B, y1B, x2B, y2B = boxB[:4]
            areaB = (x2B - x1B) * (y2B - y1B)

            xA = max(x1A, x1B)
            yA = max(y1A, y1B)
            xB = min(x2A, x2B)
            yB = min(y2A, y2B)

            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH

            if interArea >= 0.9 * areaA and areaA < areaB:
                nested = True
                break

        if not nested:
            keep.append(boxA)

    return keep


def merge_overlapping_boxes(boxes, iou_threshold=0.3):

    boxes = boxes.copy()
    merged = []

    while boxes:

        base = boxes.pop(0)
        bx1, by1, bx2, by2 = base[:4]
        score = base[4]

        to_merge = []

        for box in boxes:
            if compute_iou(base[:4], box[:4]) > iou_threshold:
                to_merge.append(box)

        for box in to_merge:
            boxes.remove(box)
            bx1 = min(bx1, box[0])
            by1 = min(by1, box[1])
            bx2 = max(bx2, box[2])
            by2 = max(by2, box[3])
            score = max(score, box[4])

        merged.append([bx1, by1, bx2, by2, score])

    return merged

def remove_shadows(frame_color, background_color, current_mask):
    # Convert both current frame and background model to HSV color space
    hsv_frame = frame_color
    hsv_bg = background_color
    
    # Hue (0), Saturation (1), Value/Brightness (2)
    h_f, s_f, v_f = hsv_frame[:,:,0], hsv_frame[:,:,1], hsv_frame[:,:,2]
    h_b, s_b, v_b = hsv_bg[:,:,0], hsv_bg[:,:,1], hsv_bg[:,:,2]
    
    # Thresholds based on Cucchiara et al. theory
    # alpha: lower bound for brightness reduction
    # beta: upper bound for brightness reduction
    alpha, beta = 0.4, 0.9
    # tau_s: maximum allowed change in Saturation
    # tau_h: maximum allowed change in Hue
    tau_s, tau_h = 50, 20   

    # Calculate the Brightness ratio (Value) between frame and background
    v_ratio = v_f / (v_b + 1e-6)
    
    # Shadow mask criteria:
    # 1. Brightness decreases within the expected range (alpha < ratio < beta)
    # 2. Saturation remains similar or slightly lower
    # 3. Hue (chromaticity) remains similar to the background
    shadow_mask = (v_ratio > alpha) & (v_ratio < beta) & \
                  (np.abs(s_f - s_b) <= tau_s) & \
                  (np.abs(h_f - h_b) <= tau_h)
    
    # Refine the original foreground mask: 
    # If a pixel was marked as foreground but meets shadow criteria, set it to background (0)
    cleaned_mask = current_mask.copy()
    cleaned_mask[shadow_mask] = 0
    
    return cleaned_mask

# --------------------------------------------------
# Bounding boxes
# --------------------------------------------------

def get_bounding_boxes(mask, min_area):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bbox_area = w * h
        fill_ratio = area / (bbox_area + 1e-6)
        aspect_ratio = h / (w + 1e-6)

        if (
            area >= min_area and
            fill_ratio > 0.4 and
            0.4 < aspect_ratio < 2.5
        ):
            score = 0.5
            boxes.append([x, y, x+w, y+h, score])

    return boxes


# --------------------------------------------------
# COCO conversion
# --------------------------------------------------

def gt_to_coco(gt, train_end, num_test_frames,
               height=1080, width=1920,
               output_json="gt.json"):

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "car"}]
    }

    ann_id = 1

    for i in range(num_test_frames):

        frame_id = train_end + i

        coco_dict["images"].append({
            "id": int(frame_id),
            "width": width,
            "height": height,
            "file_name": f"{frame_id}.jpg"
        })

        if frame_id in gt:
            for box in gt[frame_id]:

                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                coco_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": int(frame_id),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w*h),
                    "iscrowd": 0
                })
                ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_dict, f)

    return output_json


def preds_to_coco(all_boxes, train_end, output_json="preds.json"):

    preds = []

    for i, boxes in enumerate(all_boxes):

        image_id = train_end + i

        for box in boxes:

            x1, y1, x2, y2, score = box
            w = x2 - x1
            h = y2 - y1

            preds.append({
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    with open(output_json, "w") as f:
        json.dump(preds, f)

    return output_json


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_coco(gt_json, pred_json):

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[1]

# --------------------------------------------------
# Core segmentation engine
# --------------------------------------------------

def segment_and_detect(frame_gray, frame_hsv, mean_gray, std_gray, mean_hsv, alpha, open_size, close_size, min_area, roi):
    kernel_open = np.ones((open_size, open_size), np.uint8)
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    diff = np.abs(frame_gray - mean_gray)

    mask_bool = (diff >= alpha * (std_gray + 2)) & roi
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    mask_uint8 = remove_shadows(frame_hsv, mean_hsv, mask_uint8)
    mask_proc = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
    mask_proc = cv2.morphologyEx(mask_proc, cv2.MORPH_CLOSE, kernel_close)
    mask_proc = cv2.dilate(mask_proc, kernel_dilate, iterations=1)

    boxes = get_bounding_boxes(mask_proc, min_area)
    boxes = remove_nested_boxes(boxes)
    boxes = merge_overlapping_boxes(boxes)
    return boxes


# --------------------------------------------------
# Task 1 Runner (MATCHES NOTEBOOK)
# --------------------------------------------------

def run_task1(args):
    print(f"Running Task 2: Single Gaussian (alpha={args.alpha}, min_area={args.min_area}, open_size={args.open_size}, close_size={args.close_size}) ---")
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
    best_min_are = None
    best_open_size = None
    best_close_size = None
    best_boxes = None
    best_ap50 = -1
    results_list = [] if save_results else None

    for alpha, min_area, open_size, close_size in product(args.alpha, args.min_area, args.open_size, args.close_size):
        print(f"Testing alpha={alpha}, min_area={min_area}, open_size={open_size}, close_size={close_size}...")
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
                    mean_gray = prev_mean_gray + (frame_gray - prev_mean_gray) / idx
                    std_gray = std_gray + (frame_gray - prev_mean_gray) * (frame_gray - mean_gray)
                    prev_mean_gray = mean_gray
                    mean_hsv = mean_hsv + (frame_hsv - mean_hsv) / idx
            else:
                if idx == train_size:
                    std_gray = np.sqrt(std_gray / (train_size - 2))
                
                boxes = segment_and_detect(frame_gray, frame_hsv, mean_gray, std_gray, mean_hsv, alpha, open_size, close_size, min_area, roi)
                all_pred_boxes.append(boxes)

        pred_json = preds_to_coco(all_pred_boxes, train_size)
        
        ap50 = evaluate_coco(gt_json, pred_json)

        if save_results:
            results_list.append({'alpha': alpha, 'min_area': min_area, 'open_size': open_size, 'close_size': close_size, 'ap50': ap50})

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_alpha = alpha
            best_min_area = min_area
            best_open_size = open_size
            best_close_size = close_size
            best_boxes = all_pred_boxes

    
    if save_results:
        output_dir = f"{args.task}/results"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{os.path.basename(args.config).split('.')[0]}.csv")

        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['alpha', 'min_area', 'open_size', 'close_size', 'ap50'])
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\nGrid Search Finished. Results saved to: {csv_path}")

    print(f"\nFinal Adaptive AP50: {best_ap50:.4f}")
    print(f"\nFinal Alpha: {best_alpha:.4f}")
    print(f"\nFinal Min Area: {best_min_area:.4f}")
    print(f"\nFinal Open Size: {best_open_size:.4f}")
    print(f"\nFinal Close Size: {best_close_size:.4f}")

    video.close()

    return best_boxes, gt_dict, train_size, best_alpha, best_min_area, best_open_size, best_close_size, best_ap50