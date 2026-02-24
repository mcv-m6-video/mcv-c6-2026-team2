import imageio
import os
import cv2
import numpy as np
from tqdm import tqdm

def create_detection_video(video_path,
                         all_pred_boxes,
                         gt_per_frame,
                         train_size,
                         output_path="detections.avi"):
    
    cap_read = cv2.VideoCapture(video_path)
    fps = cap_read.get(cv2.CAP_PROP_FPS)
    size = (int(cap_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    num_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap_read.get(cv2.CAP_PROP_FOURCC))

    cap_write = cv2.VideoWriter(output_path, fourcc, fps, size)
    print("Generating Annotated Video...")

    for idx in tqdm(range(num_frames)):
        ret, frame = cap_read.read()
        if not ret:
            break

        if idx < train_size:
            continue

        pred_boxes = all_pred_boxes[idx - train_size]
        gt_boxes = gt_per_frame.get(idx, [])

        # Predictions
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Ground truth
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        
        cap_write.write(frame)
    
    cap_write.release()
    cap_read.release()


    

def create_detection_gif(test_frames,
                         all_pred_boxes,
                         gt_per_frame,
                         train_end,
                         output_path="detections.gif",
                         max_frames=None):

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
            cv2.rectangle(frame_color, (x1, y1), (x2, y2), (0,255,0), 2)

        # Ground truth
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_color, (x1, y1), (x2, y2), (0,0,255), 2)

        # Convert BGR → RGB for imageio
        frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)

        frames_for_gif.append(frame_rgb)

    imageio.mimsave(output_path, frames_for_gif, fps=10, loop=0)

    print("GIF saved to:", output_path)
