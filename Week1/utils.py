import imageio
import os
import cv2
import numpy as np
from tqdm import tqdm
from pygifsicle import optimize

def create_detection_video(video_path,
                         all_pred_boxes,
                         gt_per_frame,
                         train_size,
                         output_path="detections.mp4"):
    
    cap_read = cv2.VideoCapture(video_path)
    fps = cap_read.get(cv2.CAP_PROP_FPS)
    size = (int(cap_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    num_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = int(cap_read.get(cv2.CAP_PROP_FOURCC))

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
        if opt:
            print("Optimizing gif...")
            optimize(output_gif)
    
    if os.path.exists(tmp_frame_path):
        os.remove(tmp_frame_path)

    cap.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--start_second', type=int, default=None)
    parser.add_argument('--start_frame', type=int, default=None)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--opt', action='store_true')
    args = parser.parse_args()

    assert args.start_second is not None or args.start_frame is not None, f"Must use start_second or start_frame. Now they are {args.start_second} and {args.start_frame}"
    gif_selector(args.video_path, args.output_path, start_second=args.start_second, start_frame=args.start_frame, end_frame=args.end_frame, opt=args.opt)
    