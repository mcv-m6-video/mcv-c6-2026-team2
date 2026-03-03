import os
from src.task2.overlap_tracker import OverlapTracker
from src.task2.utils import load_detections, filter_duplicates, convert_xml_to_mot, run_trackeval_script, create_tracking_video, prepare_trackeval_folders, video_to_gif

def run_task21(det_path,
               output_txt_path,
               video_path=None,
               xml_gt_path=None,
               trackeval_path=None,
               make_video=False,
               iou_threshold=0.4,
               max_age=5,
               conf_threshold=0.5,
               filter_threshold=0.9):
    """
    Task 2.1: Tracking by Maximum Overlap (Greedy + max_age + conf_threshold + duplicate threshold)
    """

    all_detections = load_detections(det_path)

    tracker = OverlapTracker(
        iou_threshold=iou_threshold,
        max_age=max_age,
        conf_threshold=conf_threshold
    )

    all_results = []

    for frame_id in sorted(all_detections.keys()):
        raw_dets = all_detections[frame_id]
        clean_dets = filter_duplicates(raw_dets, threshold=filter_threshold)

        active_tracks = tracker.update(clean_dets)

        frame_out = frame_id + 1

        for track in active_tracks:
            if track.misses == 0:
                bbox = track.last_bbox()
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]

                all_results.append(
                    f"{frame_out},{track.id},"
                    f"{bbox[0]:.2f},{bbox[1]:.2f},"
                    f"{w:.2f},{h:.2f},"
                    f"1,-1,-1,-1\n"
                )

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, 'w') as f:
        f.writelines(all_results)

    print(f"Tracking results saved to: {output_txt_path}")

    # Evaluation
    if xml_gt_path and trackeval_path:
        gt_txt_path = output_txt_path.replace(".txt", "_gt.txt")
        metrics_path = output_txt_path.replace(".txt", "_metrics.txt")

        convert_xml_to_mot(xml_gt_path, gt_txt_path)

        prepare_trackeval_folders(
            trackeval_path,
            output_txt_path,
            gt_txt_path,
            tracker_name="overlap"
        )

        run_trackeval_script(
            trackeval_path,
            tracker_name="overlap",
            save_path=metrics_path
        )

    # Video
    if video_path and make_video:
        video_out = output_txt_path.replace(".txt", "_viz.mp4")
        gif_out = output_txt_path.replace(".txt", ".gif")
        create_tracking_video(video_path, output_txt_path, video_out, start_frame=900, end_frame=970)
        video_to_gif(video_path=video_out, output_gif=gif_out)

    return output_txt_path