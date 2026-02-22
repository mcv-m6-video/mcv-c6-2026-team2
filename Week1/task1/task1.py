import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import xml.etree.ElementTree as ET


# --------------------------------------------------
# Video loading
# --------------------------------------------------

def load_video_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))

    cap.release()
    return np.array(frames)


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

def segment_and_detect(test_frames, mu, sigma, roi, args):
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

    return all_pred_boxes


# --------------------------------------------------
# Task 1 Runner (MATCHES NOTEBOOK)
# --------------------------------------------------

def run_task1(args):

    print("Running Task 1")

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

    all_boxes = segment_and_detect(test, mu, sigma, roi, args)

    gt = load_ground_truth(args.annotations)

    gt_json = gt_to_coco(gt, train_end, len(test))
    pred_json = preds_to_coco(all_boxes, train_end)

    ap50 = evaluate_coco(gt_json, pred_json)

    print(f"\nFinal AP50: {ap50:.4f}")

    return test, all_boxes, gt, train_end