import os

import cv2
import imageio
import numpy as np


def create_video(
    video_path, results_path, output_video_path, max_frames=500, tracking=False
):
    """
    Overlays tracking results (bounding boxes and IDs) onto the video.
    """
    # Load results into a dict: {frame_id: [[id, x, y, w, h], ...]}
    results = {}
    with open(results_path, "r") as f:
        for line in f:
            p = line.strip().split(",")
            f_id, obj_id, l, t, w, h = (
                int(p[0]),
                int(p[1]),
                float(p[2]),
                float(p[3]),
                float(p[4]),
                float(p[5]),
            )
            if f_id not in results:
                results[f_id] = []
            results[f_id].append([obj_id, l, t, w, h])

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 1
    while cap.isOpened() and frame_idx <= max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in results:
            for res in results[frame_idx]:
                obj_id, l, t, w, h = res
                # Draw box and ID
                cv2.rectangle(
                    frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2
                )
                if tracking:
                    cv2.putText(
                        frame,
                        f"ID: {obj_id}",
                        (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def create_detection_gif(
    test_frames,
    all_pred_boxes,
    gt_per_frame,
    train_end,
    output_path="detections.gif",
    max_frames=None,
):

    frames_for_gif = []

    total_frames = len(test_frames)

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print("Generating GIF with", total_frames, "frames...")

    for i in range(total_frames):
        original_frame = train_end + i

        frame = test_frames[i].astype(np.uint8)
        pred_boxes = all_pred_boxes[i]
        gt_boxes = gt_per_frame.get(original_frame, [])

        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Predictions
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ground truth
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert BGR → RGB for imageio
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)

        frames_for_gif.append(frame_rgb)

    imageio.mimsave(output_path, frames_for_gif, fps=10, loop=0)

    print("GIF saved to:", output_path)

def gif_selector(video_path, output_gif, start_second=None, start_frame=None, end_frame=None, opt=False):
    tmp_frame_path = 'tmp_frame.jpg'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_size = (width // 2, height // 2)
    recording = -1
    skip_second = 0
    gif = []

    if start_frame is None:
        start_frame = int(start_second * fps)

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            end_frame = idx
            break

        if idx < start_frame:
            continue

        frame = cv2.resize(frame, new_size)
        cv2.imwrite(tmp_frame_path, frame)

        if not end_frame:
            if recording != -1:
                if idx % 3 == 0:
                    gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if skip_second > 0:
                skip_second -= 1
                continue

            decision = input()

            if decision == '':
                continue
            elif decision == 'r':
                gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                recording = idx
            elif decision == 's':
                skip_second = 60
                print(f"Skipping to frame {idx + skip_second}")
            elif decision == 'f':
                end_frame = idx
                break
        else:
            recording = start_frame
            if idx % 3 == 0:
                gif.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if idx >= end_frame:
                break
    
    print(f"Starting frame: {recording}")
    print(f"End frame: {end_frame}")
    
    if len(gif) > 0:
        imageio.mimsave(output_gif, gif, loop=0)
    
    if os.path.exists(tmp_frame_path):
        os.remove(tmp_frame_path)

    cap.release()