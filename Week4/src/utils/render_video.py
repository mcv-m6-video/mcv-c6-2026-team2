from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Creates a deterministic BGR color from an id."""
    rng = np.random.default_rng(int(track_id) + 12345)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_labeled_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draws one labeled bounding box on a frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Computes IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def find_best_gt_id(
    pred_bbox: tuple[int, int, int, int],
    gt_frame: Optional[pd.DataFrame],
) -> int:
    """Finds the GT id with the highest IoU for one predicted box."""
    if gt_frame is None or gt_frame.empty:
        return -1

    best_gt_id = -1
    best_iou = 0.0
    for _, gt_row in gt_frame.iterrows():
        gt_bbox = (
            int(gt_row["X"]),
            int(gt_row["Y"]),
            int(np.ceil(gt_row["X"] + gt_row["Width"])),
            int(np.ceil(gt_row["Y"] + gt_row["Height"])),
        )
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt_id = int(gt_row["Id"])

    return best_gt_id if best_iou > 0 else -1


def render_prediction_videos(
    dataset_root: str,
    dstype: str,
    seq: str,
    pred: pd.DataFrame,
    output_dir: str,
    suffix: str = "evaluated",
    gt: Optional[pd.DataFrame] = None,
) -> None:
    """Renders annotated videos from prediction tracks."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    grouped = {
        cam_id: {
            frame_id: frame_df
            for frame_id, frame_df in cam_df.groupby("FrameId")
        }
        for cam_id, cam_df in pred.groupby("CameraId")
    }
    gt_grouped = {}
    if gt is not None:
        gt_grouped = {
            cam_id: {
                frame_id: frame_df
                for frame_id, frame_df in cam_df.groupby("FrameId")
            }
            for cam_id, cam_df in gt.groupby("CameraId")
        }

    for cam_id in sorted(grouped.keys()):
        cam_name = f"c{int(cam_id):03d}"
        video_path = Path(dataset_root) / dstype / seq / cam_name / "vdo.avi"
        if not video_path.exists():
            print(f"Warning: video not found for camera {cam_name}: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        writer = cv2.VideoWriter(
            str(output_root / f"{seq}_{cam_name}_{suffix}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_idx = 1
        camera_frames = grouped[cam_id]
        camera_gt_frames = gt_grouped.get(cam_id, {})
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_dets = camera_frames.get(frame_idx)
            if frame_dets is not None:
                gt_frame = camera_gt_frames.get(frame_idx)
                for _, row in frame_dets.iterrows():
                    x1 = int(row["X"])
                    y1 = int(row["Y"])
                    x2 = int(np.ceil(row["X"] + row["Width"]))
                    y2 = int(np.ceil(row["Y"] + row["Height"]))
                    track_id = int(row["Id"])
                    gt_id = find_best_gt_id((x1, y1, x2, y2), gt_frame)
                    color = color_for_id(track_id)
                    draw_labeled_box(
                        frame,
                        (x1, y1, x2, y2),
                        f"P{track_id} | GT{gt_id}",
                        color,
                    )

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()


def render_assignment_videos(
    output_dir: str,
    seq: str,
    video_paths: list[str],
    gt_by_camera: list[dict],
    camera_names: list[str],
    assignments: dict[tuple[int, int], int],
    suffix: str = "matched",
    allowed_track_ids: Optional[set[int]] = None,
    logger=None,
) -> None:
    """Renders final annotated videos using finished global assignments."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.info("Rendering annotated videos to %s", output_root)

    for cam_idx, (video_path, gt_frames, cam_name) in enumerate(
        zip(video_paths, gt_by_camera, camera_names)
    ):
        if logger is not None:
            logger.info("Rendering camera %s from %s", cam_name, video_path)
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        writer = cv2.VideoWriter(
            str(output_root / f"{seq}_{cam_name}_{suffix}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_idx = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = gt_frames.get(frame_idx, [])
            for det in detections:
                local_id = det["track_id"]
                if allowed_track_ids is not None and local_id not in allowed_track_ids:
                    continue
                bbox = det["bbox"].astype(int)
                global_id = assignments.get((cam_idx, local_id), -1)
                color = color_for_id(global_id if global_id >= 0 else local_id)
                draw_labeled_box(
                    frame,
                    tuple(bbox.tolist()),
                    f"P{global_id} | GT{local_id}",
                    color,
                )

            writer.write(frame)
            if logger is not None and (frame_idx == 1 or frame_idx % 500 == 0):
                logger.info(
                    "Rendered camera %s | frame %d",
                    cam_name,
                    frame_idx,
                )
            frame_idx += 1

        cap.release()
        writer.release()
        if logger is not None:
            logger.info("Finished rendering camera %s", cam_name)
