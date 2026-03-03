import xml.etree.ElementTree as ET
import cv2
import subprocess
import os
import shutil
import cv2
from collections import defaultdict, deque
import numpy as np
import imageio

def load_detections(file_path):
    """
    Loads detections from AICity_data/train/S03/c010/det/det_mask_rcnn.txt (we can change this when we finish Task 1).
    Format: <frame>, <id>, <left>, <top>, <width>, <height>, <conf>, ...
    """
    detections = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            # We ignore parts[1] because detections are unassigned (-1) 
            # Convert to [xtl, ytl, xbr, ybr] 
            left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            
            if frame not in detections:
                detections[frame] = []
            detections[frame].append([left, top, left + w, top + h, conf])
    return detections

def convert_xml_to_mot(xml_path, output_txt_path):
    """
    Converts AICity XML annotations to MOTChallenge format for TrackEval.
    Filters out those outside the field of view.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for track in root.findall('track'):
        track_id = track.get('id')
        if track.get('label') != 'car': # We are only tracking cars
            continue 
        
        for box in track.findall('box'):
            # Remove if it's outside the frame 
            if box.get('outside') == '1': 
                continue
            
            # Note: We keep parked and occluded as they are valid tracking targets
            frame = int(box.get('frame')) + 1
            xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
            xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
            w, h = xbr - xtl, ybr - ytl
            
            # Format: frame, id, left, top, width, height, conf, -1, -1, -1
            lines.append(f"{frame},{track_id},{xtl:.2f},{ytl:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    
    lines.sort(key=lambda x: int(x.split(',')[0]))
    with open(output_txt_path, 'w') as f:
        f.writelines(lines)

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
        detections = [det for det in detections if compute_iou(best_box[:4], det[:4]) < threshold]

    return kept_detections

# def create_tracking_video(video_path, results_path, output_video_path, max_frames=500):
#     """
#     Overlays tracking results (bounding boxes and IDs) onto the video.
#     """
#     # Load results into a dict: {frame_id: [[id, x, y, w, h], ...]}
#     results = {}
#     with open(results_path, 'r') as f:
#         for line in f:
#             p = line.strip().split(',')
#             f_id, obj_id, l, t, w, h = int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
#             if f_id not in results: results[f_id] = []
#             results[f_id].append([obj_id, l, t, w, h])

#     cap = cv2.VideoCapture(video_path)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     frame_idx = 1
#     while cap.isOpened() and frame_idx <= max_frames:
#         ret, frame = cap.read()
#         if not ret: break
        
#         if frame_idx in results:
#             for res in results[frame_idx]:
#                 obj_id, l, t, w, h = res
#                 # Draw box and ID
#                 cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0, 255, 0), 2)
#                 cv2.putText(frame, f"ID: {obj_id}", (int(l), int(t)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         out.write(frame)
#         frame_idx += 1

#     cap.release()
#     out.release()

def create_tracking_video(video_path, results_path, output_video_path,
                          start_frame=1, end_frame=500):
    """
    Overlays tracking results (bounding boxes, IDs and track trails) onto
    the video between start_frame and end_frame.
    """
    print("Generating video...")

    results = {}
    with open(results_path, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            f_id = int(p[0])
            obj_id = int(p[1])
            l, t, w, h = map(float, p[2:6])
            results.setdefault(f_id, []).append([obj_id, l, t, w, h])

    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    trails = defaultdict(list)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    frame_idx = start_frame

    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in results:
            for obj_id, l, t, w, h in results[frame_idx]:
                cx = int(l + w/2)
                cy = int(t + h/2)
                trails[obj_id].append((cx, cy))

                cv2.rectangle(frame, (int(l), int(t)),
                              (int(l+w), int(t+h)),
                              (0, 0, 255), 2)

                cv2.putText(frame, f"ID: {obj_id}",
                            (int(l), int(t)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        # Trails
        for obj_id, pts in trails.items():
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_arr], False,
                              (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def video_to_gif(video_path, output_gif):
    print("Converting to gif...")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (700, 350))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    imageio.mimsave(output_gif, frames, loop=0)


def prepare_trackeval_folders(base_trackeval_path, results_path, gt_path, tracker_name="overlap"):
    """
    Creates the exact MOTChallenge directory structure expected by TrackEval.
    """
    # Define the benchmark name and sequence name consistently
    benchmark = "AICity"
    split = "train"
    sequence = "s03" # This is the folder name TrackEval will look for
    
    # Define internal TrackEval paths
    # TrackEval/data/gt/mot_challenge/AICity-train/s03/gt/gt.txt
    # TrackEval/data/trackers/mot_challenge/AICity-train/overlap/data/s03.txt
    data_path = os.path.join(base_trackeval_path, "data")
    gt_base_path = os.path.join(data_path, "gt", "mot_challenge", f"{benchmark}-{split}")
    seqmap_path = os.path.join(data_path, "gt", "mot_challenge", "seqmaps")
    tracker_data_path = os.path.join(data_path, "trackers", "mot_challenge", f"{benchmark}-{split}", tracker_name, "data")

    # Create the directory tree
    os.makedirs(os.path.join(gt_base_path, sequence, "gt"), exist_ok=True)
    os.makedirs(seqmap_path, exist_ok=True)
    os.makedirs(tracker_data_path, exist_ok=True)

    # Create seqinfo.ini (Crucial for TrackEval to know the sequence length)
    ini_content = f"[Sequence]\nname={sequence}\nseqLength=2141\n"
    with open(os.path.join(gt_base_path, sequence, "seqinfo.ini"), "w") as f:
        f.write(ini_content)

    # Create the seqmap file (Must be named: [BENCHMARK]-[SPLIT].txt)
    with open(os.path.join(seqmap_path, f"{benchmark}-{split}.txt"), "w") as f:
        f.write(f"name\n{sequence}")

    # Copy Ground Truth and Predictions
    # Prediction file MUST be named after the sequence (s03.txt)
    shutil.copy(results_path, os.path.join(tracker_data_path, f"{sequence}.txt"))
    # GT file MUST be named gt.txt
    shutil.copy(gt_path, os.path.join(gt_base_path, sequence, "gt", "gt.txt"))

    print(f"TrackEval structure ready for benchmark '{benchmark}' and tracker '{tracker_name}'")

def run_trackeval_script(trackeval_path, tracker_name="overlap", save_path=None):
    """
    Runs the TrackEval script using the standardized folder structure created above.
    """
    command = [
        "python", os.path.join(trackeval_path, "scripts", "run_mot_challenge.py"),
        "--BENCHMARK", "AICity",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS", "HOTA", "Identity", "CLEAR",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(result.stdout)
        print(f"Metrics saved to: {save_path}")