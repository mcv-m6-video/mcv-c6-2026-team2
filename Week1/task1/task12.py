import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import xml.etree.ElementTree as ET
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances, Boxes
from task1.prepare_dataset import convert_dataset, load_video_frames, load_ground_truth, gt_to_coco
import torch
from pprint import pprint


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

def format_for_d2_xywh(image_id, height, width, bboxes_xywh, scores, classes):
    """
    bboxes_xywh: List or array of [x, y, width, height]
    scores: List of confidence scores
    classes: List of predicted class integers
    """
    instances = Instances((height, width))
    boxes_tensor = torch.tensor(bboxes_xywh, dtype=torch.float32).unsqueeze(0)
    
    if len(boxes_tensor) > 0:
        boxes_tensor[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2]
        boxes_tensor[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3]
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

    instances.pred_boxes = Boxes(boxes_tensor)
    instances.scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
    instances.pred_classes = torch.tensor(classes, dtype=torch.int64).unsqueeze(0)
    
    d2_input = {"image_id": image_id, "height": height, "width": width}
    d2_output = {"instances": instances}
    
    return d2_input, d2_output

def evaluate_coco(height, width, pred_json, args):
    convert_dataset(args.dataset, args.video, args.annotations, args.train_ratio)
    dataset_name = args.dataset
    register_coco_instances(dataset_name, {}, os.path.join(dataset_name, 'annotations/val.json'), os.path.join(dataset_name, 'images/val'))

    evaluator = COCOEvaluator(dataset_name, tasks=('bbox',), distributed=False, output_dir='logs/')
    evaluator.reset()

    with open(pred_json, 'r') as f:
        pred_list = json.load(f)

    for pred in pred_list:
        image_id = pred['image_id']
        bboxes = pred['bbox']
        scores = pred['score']
        classes = pred['category_id']
        d2_input, d2_output = format_for_d2_xywh(image_id, height, width, bboxes, scores, classes)
        evaluator.process([d2_input], [d2_output])
    
    results = evaluator.evaluate()
    pprint(results)

    return results['bbox']['AP50']


# --------------------------------------------------
# Task 1 Runner (MATCHES NOTEBOOK)
# --------------------------------------------------

def run_task1(args):
    print("Running Task 1")

    frames = load_video_frames(args.video)
    height, width = frames.shape[1:3]

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

    all_boxes = []

    kernel_open = np.ones((args.open_size, args.open_size), np.uint8)
    kernel_close = np.ones((args.close_size, args.close_size), np.uint8)
    kernel_dilate = np.ones((5,5), np.uint8)

    for frame in test:

        diff = np.abs(frame - mu)
        sigma_safe = np.maximum(sigma, 1e-6)

        mask = diff >= args.alpha * (sigma_safe + 2)
        mask = mask & roi
        mask = mask.astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        boxes = get_bounding_boxes(mask, args.min_area)
        boxes = remove_nested_boxes(boxes)
        boxes = merge_overlapping_boxes(boxes)

        all_boxes.append(boxes)

    gt = load_ground_truth(args.annotations)

    gt_json = gt_to_coco(gt, train_end, len(test))
    pred_json = preds_to_coco(all_boxes, train_end)

    ap50 = evaluate_coco(height, width, pred_json, args)

    print(f"\nFinal AP50: {ap50:.4f}")

    return test, all_boxes, gt, train_end