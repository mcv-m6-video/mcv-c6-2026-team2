import glob
import os

import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from src.models.tracker import OverlapTracker
from src.utils.tracker import filter_duplicates


def initialize_model(model_path: str, device: torch.device):
    model = fasterrcnn_resnet50_fpn_v2(weights=None)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def detect_and_filter(
    obj_detector,
    image,
    device,
    conf_threshold=0.5,
    dup_iou_threshold=0.9,
):
    """
    image: np.ndarray in BGR
    returns: [[x1, y1, x2, y2, conf], ...]
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        det = obj_detector([img_tensor])[0]

    dets_frame = []
    for box_i, box in enumerate(det["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        conf = det["scores"][box_i].item()

        if conf < conf_threshold:
            continue

        dets_frame.append([x1, y1, x2, y2, conf])

    dets_frame = filter_duplicates(dets_frame, threshold=dup_iou_threshold)
    return dets_frame


def process_camera(
    video_path,
    output_txt_path,
    obj_detector,
    device,
    iou_threshold=0.4,
    dup_iou_threshold=0.9,
    max_age=5,
    conf_threshold=0.5,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    tracker = OverlapTracker(
        iou_threshold=iou_threshold,
        max_age=max_age,
        conf_threshold=conf_threshold,
    )

    all_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_and_filter(
            obj_detector=obj_detector,
            image=frame,
            device=device,
            conf_threshold=conf_threshold,
            dup_iou_threshold=dup_iou_threshold,
        )

        active_tracks = tracker.update(detections)

        for tr in active_tracks:
            if tr.misses == 0:
                x1, y1, x2, y2 = tr.last_bbox()
                w = x2 - x1
                h = y2 - y1

                all_results.append(
                    f"{frame_idx + 1},{tr.id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )

        frame_idx += 1

    cap.release()

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.writelines(all_results)

    print(f"Saved: {output_txt_path}", flush=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Tracking started", flush=True)
    print(f"dataset_root={args.dataset_root}", flush=True)
    print(f"det_checkpoint={args.det_checkpoint}", flush=True)
    print(f"output_path={args.output_path}", flush=True)

    dataset_path = args.dataset_root
    output_path = args.output_path
    obj_detector_path = args.det_checkpoint

    iou_threshold = args.iou_threshold
    dup_iou_threshold = args.dup_iou_threshold
    max_age = args.max_age
    conf_threshold = args.conf_threshold

    obj_detector = initialize_model(obj_detector_path, device)

    train_root = os.path.join(dataset_path, "train")
    seq_dirs = sorted(glob.glob(os.path.join(train_root, "S*")))

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        cam_dirs = sorted(glob.glob(os.path.join(seq_dir, "c*")))

        for cam_dir in cam_dirs:
            cam_name = os.path.basename(cam_dir)
            video_path = os.path.join(cam_dir, "vdo.avi")

            if not os.path.exists(video_path):
                print(f"[WARNING] Video does not exist: {video_path}")
                continue

            output_txt_path = os.path.join(output_path, seq_name, cam_name, "pred.txt")

            print(f"Processing {seq_name}/{cam_name}", flush=True)

            process_camera(
                video_path=video_path,
                output_txt_path=output_txt_path,
                obj_detector=obj_detector,
                device=device,
                iou_threshold=iou_threshold,
                dup_iou_threshold=dup_iou_threshold,
                max_age=max_age,
                conf_threshold=conf_threshold,
            )