import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import json

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
        frames.append(gray)

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

def convert_dataset(new_root: str, video: str, xml_path: str, train_ratio: int = 0.25):
    if os.path.exists(new_root):
        return
    
    os.makedirs(new_root)

    frames = load_video_frames(video)
    gt = load_ground_truth(xml_path)

    train_num = int(train_ratio * len(frames))

    train_set = frames[:train_num]
    val_set = frames[train_num:]

    os.makedirs(os.path.join(new_root, 'images', 'train'))
    for idx, img in enumerate(train_set):
        cv2.imwrite(os.path.join(new_root, 'images', 'train', f'{idx}.jpg'), img)

    os.makedirs(os.path.join(new_root, 'images', 'val'))
    for idx, img in enumerate(val_set):
        cv2.imwrite(os.path.join(new_root, 'images', 'val', f'{idx + train_num}.jpg'), img)

    os.makedirs(os.path.join(new_root, 'annotations'))
    train_json = gt_to_coco(gt, 0, len(train_set), output_json=os.path.join(new_root, 'annotations', 'train.json'))
    val_json = gt_to_coco(gt, train_num, len(val_set), output_json=os.path.join(new_root, 'annotations', 'val.json'))

if __name__ == '__main__':
    convert_dataset('dataset', 'Data/AICity_data/train/S03/c010/vdo.avi', 'Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml', train_ratio=0.25)