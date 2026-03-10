import numpy as np


def compute_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Bbox format: [xtl, ytl, xbr, ybr]
    """
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


def filter_duplicates(detections, threshold=0.9):
    """
    Removes redundant bounding boxes in the same frame.
    If IoU > threshold, keep the one with higher confidence. 
    """
    if not detections:
        return []

    # Sort detections by confidence (index 4) descending
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    kept_detections = []

    while len(detections) > 0:
        best_box = detections.pop(0)
        kept_detections.append(best_box)

        # Only keep boxes that don't overlap too much with the 'best_box'
        detections = [det for det in detections if compute_iou(
            best_box[:4], det[:4]) < threshold]

    return kept_detections


def filter_duplicates_numpy(detections, threshold=0.9):
    """
    Greedy duplicate filtering / NMS-like suppression.
    detections: list of detections, each with at least [x1, y1, x2, y2, conf]
    """
    if not detections:
        return []

    dets = np.asarray(detections, dtype=float)

    boxes = dets[:, :4]
    scores = dets[:, 4]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-6)

        order = rest[iou < threshold]

    return [detections[i] for i in keep]
