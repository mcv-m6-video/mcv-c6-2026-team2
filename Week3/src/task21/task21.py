import os

from src.models.tracker_model import Track
from src.utils.city_dataset import AICityDataset
from src.utils.track_eval import prepare_trackeval_folders, run_trackeval_script
from src.utils.visualizations import create_tracking_video


def main(args):
    # Process arguments
    dataset_path = args.dataset_path
    output_path = args.output_path
    tracker_name = args.tracker_name
    make_video = args.make_video

    base_trackeval_path = args.base_trackeval_path

    # Read dataset
    dataset = AICityDataset(dataset_path)

    # Load tracker model
    # (Change it for the correct one, this is a placeholder)
    tracker = Track()
    track_input_size = tracker.get_input_size()

    # Process and evaluate dataset
    for track in range(len(dataset)):
        # Set active track
        dataset.change_active_track(track)

        # Process dataset
        frames = [None for i in range(track_input_size)]
        for frame in dataset.get_video_stream():
            frames = frames[1:] + [frames[0]]
            frames[-1] = frame

            if None in frames:
                continue

            tracker(frames)

        # Save prediction
        track_path = os.path.split(dataset.get_active_video_path())[0]
        track_name = os.path.basename(track_path)
        pred_path = os.path.join(output_path, track_name, "pred.txt")

        with open(pred_path, "w") as f:
            f.writelines(tracker.tracks())

        # Evaluate dataset with TrackEval
        gt_path = dataset.get_gt()

        prepare_trackeval_folders(base_trackeval_path, pred_path, gt_path, tracker_name)

        output_metrics_path = os.path.join(output_path, track_name, "results.txt")
        run_trackeval_script(base_trackeval_path, tracker_name, output_metrics_path)

        if make_video:
            output_video_path = os.path.join(output_path, track_name, "track.mp4")
            create_tracking_video(
                dataset.get_active_video_path(),
                pred_path,
                output_video_path,
                end_frame=100000000000,
            )
