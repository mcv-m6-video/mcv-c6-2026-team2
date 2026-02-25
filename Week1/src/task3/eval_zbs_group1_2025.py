import cv2
import numpy as np
import os
import argparse

#from src.utils import *
from lxml import etree


def mean_avg_precision(gt, pred, iou_threshold=0.5):
    """Calculate the mean average precision for a given ground truth and prediction. From Team 5-2024 (slightly modified).

    Args:
        gt (list): List of ground truth bounding boxes
        pred (list): List of predicted bounding boxes
        iou_threshold (float): The intersection over union threshold. Defaults to 0.5.

    Returns:
        float: The mean average precision
    """
    if len(gt) == 0 or len(pred) == 0:
        return 1 if len(gt) == len(pred) == 0 else 0

    # Initialize variables
    tp = np.zeros(len(pred))
    fp = np.zeros(len(pred))
    gt_matched = [False] * len(gt)

    # Loop through each prediction
    for i, p in enumerate(pred):
        ious = [iou(p, g) for g in gt]
        if len(ious) == 0:
            fp[i] = 1
            continue

        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]

        if max_iou >= iou_threshold and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = True
        else:
            fp[i] = 1

    # Calculate precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt)  # len(gt) is equivalent to TP + FN
    precision = tp / (tp + fp)

    # Generate graph with the 11-point interpolated precision-recall curve (Team5-2024)
    recall_interp = np.linspace(0, 1, 11)
    precision_interp = np.zeros(11)
    for i, r in enumerate(recall_interp):
        array_precision = precision[recall >= r]
        if len(array_precision) == 0:
            precision_interp[i] = 0
        else:
            precision_interp[i] = max(precision[recall >= r])
    return np.mean(precision_interp)


def iou(boxA, boxB):
    """Calculate the intersection over union (IoU) of two bounding boxes. From Team 5-2024 (slightly modified)

    Format of bounding boxes is top-left and bottom-right coordinates [x1, y1, x2, y2].

    Args:
        boxA (list): List containing the coordinates of the first bounding box [x1, y1, x2, y2]
        boxB (list): List containing the coordinates of the second bounding box [x1, y1, x2, y2]

    Returns:
        float: The IoU value
    """
    # Calculate the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate the area of each bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate the union area
    unionArea = boxAArea + boxBArea - interArea

    # Calculate the IoU
    return interArea / unionArea


def read_annotations(annotations_path: str):
    """Read the annotations from the XML file. From Team 6-2024 (slightly modified).

    Args:
        annotations_path (str): Path to the XML file

    Returns:
        dict: Dictionary containing the car bounding boxes for each frame
    """
    # Read the XML file where the annotations (GT) are
    tree = etree.parse(annotations_path)
    # Take the root element of the XML file
    root = tree.getroot()
    # Where GT 2DBB are kept
    car_boxes = {}

    # Iterate over all car annotations (label = "car")
    for track in root.xpath(".//track[@label='car']"):
        # Iterate over all 2DBB annotated for the car
        for box in track.xpath(".//box"):
            # Only interested in moving cars --> If the car is not parked, store the GT 2DBB
            # annotation and the corresponding frame id
            parked_attribute = box.find(".//attribute[@name='parked']")
            if parked_attribute is not None and parked_attribute.text == 'false':
                frame = box.get("frame")

                # Check if the box is occluded
                # is_occluded = box.get("occluded")
                # if is_occluded == "1":
                #     continue

                box_attributes = {
                    "xtl": float(box.get("xtl")),
                    "ytl": float(box.get("ytl")),
                    "xbr": float(box.get("xbr")),
                    "ybr": float(box.get("ybr")),
                    "occluded": int(box.get("occluded")),
                }

                if frame in car_boxes:
                    car_boxes[frame].append(box_attributes)
                else:
                    car_boxes[frame] = [box_attributes]
    return car_boxes


def get_bounding_box(mask: np.ndarray, output_frame: np.ndarray, aspect_ratio_threshold: float = 1.2, area_threshold: int = 918) -> tuple:
    """Get the bounding box of the mask

    Args:
        mask (np.ndarray): The mask to calculate the bounding box of
        output_frame (np.ndarray): The frame to draw the bounding box on

    Returns:
        tuple: Tuple containing the top-left and bottom-right coordinates of the bounding box, and the output frame
    """
    # Get connected components
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=4)

    coords = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]

        if area < area_threshold:  # Filter out small areas
            continue

        # Filter out by aspect ratio
        aspect_ratio = h/w
        if aspect_ratio > aspect_ratio_threshold:
            continue

        top_left = (x, y)
        bottom_right = (x + w, y + h)
        coords.append((top_left, bottom_right))

    for (top_left, bottom_right) in coords:
        cv2.rectangle(output_frame, top_left, bottom_right, (0, 0, 255), 2)

    return coords, output_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mask", help="Path to the mask video file from ZBS")
    parser.add_argument("-gt", "--ground_truth",
                        help="Path to the ground truth file")
    parser.add_argument("-v", "--verbose", help="Print extra information",
                        action="store_true", default=False)
    args = parser.parse_args()

    mask_path = args.mask
    ground_truth_path = args.ground_truth
    verbose = args.verbose

    print("Evaluating ZBS algorithm over the Ground Truth")
    print("-" * 50)
    # Print the paths
    if verbose:
        print("Mask path: ", mask_path)
        print("Ground truth path: ", ground_truth_path)

    # Read the ground truth file
    gt_boxes = read_annotations(ground_truth_path)

    # Read the mask video
    mask_cap = cv2.VideoCapture(mask_path)

    # Write the output video
    output_path = os.path.join(os.path.dirname(mask_path), "output_gt.avi")
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'XVID'), int(mask_cap.get(cv2.CAP_PROP_FPS)), (int(mask_cap.get(3)), int(mask_cap.get(4))))

    num_frames = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metrics = []
    while True:
        ret, frame = mask_cap.read()
        if not ret:
            break

        # Get the frame number
        frame_number = int(mask_cap.get(cv2.CAP_PROP_POS_FRAMES))
        if verbose:
            print(f"Processing frame {frame_number}/{num_frames}")

        # Convert to grayscale and get the binarized frame
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(
            grayscale_frame, 127, 255, cv2.THRESH_BINARY)

        # Get the predicted boxes
        pred_boxes, out_frame = get_bounding_box(binary_frame, frame)

        # Convert to the same format
        gt_box = gt_boxes.get(str(frame_number + 215), [])
        gt_box = [list(map(int, [box["xtl"], box["ytl"], box["xbr"], box["ybr"]]))
                  for box in gt_box]
        pred_box = [[int(box[0][0]), int(box[0][1]), int(
            box[1][0]), int(box[1][1])] for box in pred_boxes]

        # Compare the predicted boxes with the ground truth
        avg_precision = mean_avg_precision(gt_box, pred_box)

        # Print the average precision
        if verbose:
            print("Average Precision: ", avg_precision)
        metrics.append(avg_precision)

        # Print GT boxes if they exist
        if gt_boxes:
            try:
                for box in gt_boxes[str(frame_number + 215)]:
                    xtl, ytl, xbr, ybr = int(box["xtl"]), int(
                        box["ytl"]), int(box["xbr"]), int(box["ybr"])
                    cv2.rectangle(out_frame, (xtl, ytl),
                                  (xbr, ybr), (0, 255, 0), 2)
            except KeyError:
                pass

        # Write the frame to the output video
        out_writer.write(out_frame)

        print(
            f"Frame {frame_number}/{num_frames} - Average Precision: {avg_precision}")
        print("-" * 50)

    print("Mean Average Precision: ", np.mean(metrics))

    # Print mAP of 1606 last frames
    print("Mean Average Precision of the last 1606 (75%) frames: ",
          np.mean(metrics[-1606:]))

    mask_cap.release()
    out_writer.release()
