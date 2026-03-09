import os

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def create_tracking_video(
    video_path, results_path, output_video_path, start_frame=1, end_frame=500
):
    """
    Overlays tracking results (bounding boxes, IDs and track trails) onto
    the video between start_frame and end_frame.
    """
    print("Generating video...")

    def draw_label(
        frame,
        x,
        y,
        text,
        bg_color=(0, 0, 255),
        text_color=(255, 255, 255),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.75,
        thickness=2,
        pad=4,
    ):

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        H, W = frame.shape[:2]

        x = max(0, min(x, W - tw - 2 * pad))
        y = max(th + baseline + 2 * pad, min(y, H - 2))

        x1 = x
        y1 = y - th - baseline - 2 * pad
        x2 = x + tw + 2 * pad
        y2 = y

        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

        cv2.putText(
            frame,
            text,
            (x + pad, y - baseline - pad),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    results = {}
    with open(results_path, "r") as f:
        for line in f:
            p = line.strip().split(",")
            f_id = int(p[0])
            obj_id = int(p[1])
            l, t, w, h = map(float, p[2:6])
            results.setdefault(f_id, []).append([obj_id, l, t, w, h])

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    trails = defaultdict(list)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    frame_idx = start_frame

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in results:
            for obj_id, l, t, w, h in results[frame_idx]:
                cx = int(l + w / 2)
                cy = int(t + h / 2)
                trails[obj_id].append((cx, cy))

                cv2.rectangle(
                    frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 0, 255), 2
                )

                draw_label(frame, int(l), int(t) - 5, f"ID {obj_id}")

        # Trails
        for obj_id, pts in trails.items():
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_arr], False, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def generate_video(images: list | np.ndarray, output_path: str, fps: int = 24):
    print("Generating Video...")

    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)

    width = images[0].shape[1]
    height = images[0].shape[0]

    cap_write = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in tqdm(images):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cap_write.write(bgr_image)

    cap_write.release()

    print(f"Video saved in {output_path}")


def save_flow(flow: np.ndarray, output_path: str):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = 255
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, bgr)


def generate_flow_reference(size: tuple):
    # Initial variable definitions
    width = size[1]
    half_width = width / 2
    height = size[0]
    half_height = height / 2

    # Get coords and build reference flow matrix
    ys, xs = np.ogrid[0:height, 0:width]
    xs = (xs - half_width) / width
    ys = (ys - half_height) / height
    xs = xs.repeat(ys.shape[0], axis=0)
    ys = ys.repeat(xs.shape[1], axis=1)
    flow = np.stack((xs, ys), axis=2)

    # Save reference as image
    return flow
