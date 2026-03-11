import os

import torch
from src.models import MEMFOFModel
from src.models.tracker_model import OFTracker
from src.utils.city_dataset import AICityDataset
from src.utils.track_eval import prepare_trackeval_folders, run_trackeval_script
from src.utils.visualizations import create_tracking_video
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from tqdm import tqdm


def evaluate_tracking_results(
    pred_path: str,
    gt_path: str,
    output_metrics_path: str,
    trackeval_path: str,
):
    """Evaluate tracking results using TrackEval."""
    # Evaluation (TrackEval)

    prepare_trackeval_folders(
        trackeval_path,
        pred_path,
        gt_path,
        tracker_name="optical_flow",
    )

    run_trackeval_script(
        trackeval_path,
        tracker_name="optical_flow",
        save_path=output_metrics_path,
    )


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process arguments
    dataset_path = args.dataset_path
    output_path = args.output_path
    # tracker_name = args.tracker_name
    make_video = args.make_video

    iou_threshold = args.iou_threshold
    dup_iou_threshold = args.dup_iou_threshold
    max_age = args.max_age
    min_hits = args.min_hits
    conf_threshold = args.conf_threshold

    obj_detector_path = args.obj_detector_path

    trackeval_path = args.trackeval_path

    # Read dataset
    dataset = AICityDataset(dataset_path)

    # Load tracker model
    opt_flow = MEMFOFModel(args)
    obj_detector = initialize_model(obj_detector_path, device=device)
    tracker = OFTracker(
        of=opt_flow,
        obj_detector=obj_detector,
        iou_threshold=iou_threshold,
        dup_iou_threshold=dup_iou_threshold,
        max_age=max_age,
        min_hits=min_hits,
        conf_threshold=conf_threshold,
        device=device,
    )

    # Process and evaluate dataset
    for track in range(len(dataset)):
        # Set active track
        dataset.change_active_track(track)

        # Process dataset
        all_results = []
        frame_prev = None
        for idx, frame in tqdm(
            enumerate(dataset.get_video_stream()), desc=f"Track {track}"
        ):
            if frame_prev is None:
                tracker.initialize_tracks(frame)
                frame_prev = frame
                continue

            tracks = tracker.update(frame_prev, frame)

            for tr in tracks:
                x, y, xbr, ybr, tid = tr
                w = xbr - x
                h = ybr - y

                all_results.append(
                    f"{idx + 1},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )

            frame_prev = frame

        # Save prediction
        track_path = os.path.split(dataset.get_active_video_path())[0]
        track_name = os.path.basename(track_path)
        pred_path = os.path.join(output_path, track_name, "pred.txt")

        with open(pred_path, "w") as f:
            f.writelines(tracker.tracks())

        # Evaluate dataset with TrackEval
        gt_path = dataset.get_gt()
        output_metrics_path = os.path.join(output_path, track_name, "results.txt")

        print("Evaluating tracking results...")
        evaluate_tracking_results(
            pred_path, gt_path, trackeval_path, output_metrics_path
        )

        if make_video:
            output_video_path = os.path.join(output_path, track_name, "track.mp4")
            print("Creating tracking video...")
            create_tracking_video(
                dataset.get_active_video_path(),
                pred_path,
                output_video_path,
                end_frame=100000000000,
            )
