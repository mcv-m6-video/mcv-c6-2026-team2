import os

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from src.models.tracker_model import OFTracker
from src.models import (
    BaseModel,
    FarnebackModel,
    MEMFOFModel,
    PerceiverModel,
    PyflowModel,
)
from src.utils.week2 import (
    convert_xml_to_mot,
    run_trackeval_script,
    create_tracking_video,
    prepare_trackeval_folders,
    video_to_gif,
)


def online_tracking(
    tracker: OFTracker,
    image0: np.ndarray,
    image1: np.ndarray,
):
    """
    Track objects across two frames using the provided tracker model.

    Args:
        tracker: an instance of OFTracker for tracking objects;
        image0: the previous frame as a numpy array;
        image1: the current frame as a numpy array;

    Returns:
        tracks: list of Track objects representing the tracked objects in image1.
    """
    tracks = tracker.update(image0, image1)

    return tracks


def offline_tracking(
    tracker: OFTracker,
    video_path: str
):
    video = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    video.release()

    if len(frames) == 0:
        raise RuntimeError("Could not read any frame from video.")

    print(f"[Offline] Loaded {len(frames)} frames", flush=True)

    all_dets = tracker.detect_all_frames(frames, batch_size=8)
    all_flows = tracker.compute_all_flows(frames)

    tracker.initialize_tracks_from_dets(all_dets[0])

    tracks = []

    for frame_id in range(1, len(frames)):
        print(f"[Offline] Tracking frame {frame_id}", flush=True)

        tracks_ = tracker.update_from_precomputed(
            of_output=all_flows[frame_id - 1],
            dets1=all_dets[frame_id],
        )

        for track in tracks_:
            tracks.append(track.tolist() + [frame_id])

    return tracks


def initialize_model(model_path: str, device: str):
    """
    Load a fine-tuned Faster R-CNN model from a checkpoint.

    Args:
        model_path (str): Path to the saved checkpoint (.pth)
        device (str): Device to load the model on ("cpu" or "cuda")

    Returns:
        torch.nn.Module: Loaded model
    """

    # Initialize model architecture (no pretrained weights)
    model = fasterrcnn_resnet50_fpn_v2(weights=None)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Some training scripts save {"model_state_dict": ...}
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # important for inference

    return model


def save_tracking_results(
    all_results: list,
    output_txt_path: str
):
    """Save tracking results to a text file in MOT format."""
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.writelines(all_results)

    print(f"Tracking results saved to: {output_txt_path}")


def evaluate_tracking_results(
    output_txt_path: str,
    xml_gt_path: str = None,
    trackeval_path: str = None
):
    """Evaluate tracking results using TrackEval."""
    # Evaluation (TrackEval)
    if xml_gt_path and trackeval_path:
        gt_txt_path = output_txt_path.replace(".txt", "_gt.txt")
        metrics_path = output_txt_path.replace(".txt", "_metrics.txt")

        convert_xml_to_mot(xml_gt_path, gt_txt_path)

        prepare_trackeval_folders(
            trackeval_path,
            output_txt_path,
            gt_txt_path,
            tracker_name="optical_flow",
        )

        run_trackeval_script(
            trackeval_path,
            tracker_name="optical_flow",
            save_path=metrics_path,
        )


def make_tracking_video(
    video_path: str,
    make_video: bool,
    output_txt_path: str,
    video_out: str,
    start_frame: int = None,
    end_frame: int = None
):
    """Create a video visualization of the tracking results."""
    # Video viz
    if video_path and make_video:
        video_out = output_txt_path.replace(".txt", "_viz.mp4")
        gif_out = output_txt_path.replace(".txt", ".gif")

        # If a range is not passed, keep entire video
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = -1

        create_tracking_video(
            video_path,
            output_txt_path,
            video_out,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        video_to_gif(video_path=video_out, output_gif=gif_out)


def main(args):
    print(f"Running Task 1.2 with args: {args}")

    # Define optical flow methods
    methods: dict[str, BaseModel] = {
        "pyflow": PyflowModel,
        "farneback": FarnebackModel,
        "perceiverio": PerceiverModel,
        "memfof": MEMFOFModel
    }

    of = methods[args.of_method](args)

    print(f"Initialized OF method: {args.of_method}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_detector = initialize_model(
        model_path=args.obj_detector_path, device=device)

    print(f"Initialized object detector model", flush=True)

    # initialize tracker model
    tracker = OFTracker(
        of=of,
        obj_detector=obj_detector,
        iou_threshold=args.iou_threshold,
        dup_iou_threshold=args.dup_iou_threshold,
        max_age=args.max_age,
        min_hits=args.min_hits,
        conf_threshold=args.conf_threshold,
        device=device,
    )

    print(f"Initialized tracker", flush=True)

    if args.mode == "online":

        video_path = args.video_path
        video = cv2.VideoCapture(video_path)
        ret, frame_prev = video.read()
        if not ret:
            raise RuntimeError("Could not read first frame")

        tracker.initialize_tracks(frame_prev)

        all_results: list[str] = []

        # Write MOT lines
        for tr in tracker.tracks:
            x, y, xbr, ybr = tr.bboxes[-1]
            tid = tr.id
            w = xbr - x
            h = ybr - y

            all_results.append(
                f"{1},{int(tid)},"
                f"{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},"
                f"1,-1,-1,-1\n"
            )
        print(all_results)

        frame_id = 0

        while True:
            if frame_id % 10 == 0:
                print(f"Processing frame {frame_id}...", flush=True)
            ret, frame = video.read()

            if not ret:
                break  # end of video

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_id += 1

            tracks = online_tracking(tracker, frame_prev, frame)

            # Write MOT lines
            for tr in tracks:
                x, y, xbr, ybr, tid = tr
                w = xbr - x
                h = ybr - y

                all_results.append(
                    f"{frame_id+1},{int(tid)},"
                    f"{x:.2f},{y:.2f},"
                    f"{w:.2f},{h:.2f},"
                    f"1,-1,-1,-1\n"
                )

            frame_prev = frame

    elif args.mode == "offline":
        tracks = offline_tracking(tracker, args.video_path)

        # Write MOT lines
        for tr in tracks:
            x, y, xbr, ybr, tid, frame_id = tr
            w = xbr - x
            h = ybr - y

            all_results.append(
                f"{frame_id+1},{int(tid)},"
                f"{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},"
                f"1,-1,-1,-1\n"
            )

    print("Saving tracking results...", flush=True)
    save_tracking_results(all_results, args.output_txt_path)

    print("Evaluating tracking results...", flush=True)
    evaluate_tracking_results(
        output_txt_path=args.output_txt_path,
        xml_gt_path=args.xml_gt_path,
        trackeval_path=args.trackeval_path,
    )

    print("Creating tracking visualization video...", flush=True)
    make_tracking_video(
        video_path=args.video_path,
        output_txt_path=args.output_txt_path,
        video_out=args.video_out,
        make_video=args.make_video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    return args.output_txt_path
