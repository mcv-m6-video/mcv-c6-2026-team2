# task22_kalman_sort.py

import os
import numpy as np

from src.task2.sort import Sort
from src.task2.utils import (
    load_detections,
    filter_duplicates,
    convert_xml_to_mot,
    run_trackeval_script,
    create_tracking_video,
    prepare_trackeval_folders,
    video_to_gif,
)


def main(
    det_path: str,
    output_txt_path: str,
    video_path: str | None = None,
    xml_gt_path: str | None = None,
    trackeval_path: str | None = None,
    make_video: bool = False,
    preprocess: bool = False,
    filter_threshold: float = 0.9,
    conf_threshold: float = 0.5,
    # SORT params
    iou_threshold: float = 0.3,
    max_age: int = 5,
    min_hits: int = 3,
    # video viz params (optional)
    start_frame: int | None = 900,
    end_frame: int | None = 970,
):
    """
    Task 2.2: Tracking with Kalman Filter (SORT)

    Input detections format expected from load_detections():
      per frame: list of detections
      each detection: [xtl, ytl, xbr, ybr, score]

    Output MOT format:
      frame, id, x, y, w, h, conf, -1, -1, -1
    """

    all_detections = load_detections(det_path)

    tracker = Sort(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
    )

    all_results: list[str] = []

    for frame_id in sorted(all_detections.keys()):
        # list[[xtl,ytl,xbr,ybr,score], ...]
        raw_dets = all_detections[frame_id]

        # 1) optional duplicate filtering preprocess
        if preprocess:
            dets = filter_duplicates(raw_dets, threshold=filter_threshold)
        else:
            dets = raw_dets

        # 2) confidence thresholding
        dets = [d for d in dets if d[4] >= conf_threshold]

        # 3) SORT expects Nx4 in xyxy
        if len(dets) == 0:
            dets = np.empty((0, 4), dtype=np.float32)
        else:
            dets = np.asarray(dets, dtype=np.float32)[:, :4]  # drop score

        # 4) Update tracker; returns Nx5: [x,y,x,y,track_id]
        tracks = tracker.update(dets)

        frame_out = frame_id + 1  # MOT is 1-indexed

        # 5) Write MOT lines
        for tr in tracks:
            x, y, xbr, ybr, tid = tr
            w = xbr - x
            h = ybr - y

            all_results.append(
                f"{frame_out},{int(tid)},"
                f"{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},"
                f"1,-1,-1,-1\n"
            )

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w") as f:
        f.writelines(all_results)

    print(f"Tracking results saved to: {output_txt_path}")

    # Evaluation (TrackEval)
    if xml_gt_path and trackeval_path:
        gt_txt_path = output_txt_path.replace(".txt", "_gt.txt")
        metrics_path = output_txt_path.replace(".txt", "_metrics.txt")

        convert_xml_to_mot(xml_gt_path, gt_txt_path)

        prepare_trackeval_folders(
            trackeval_path,
            output_txt_path,
            gt_txt_path,
            tracker_name="kalman_sort",
        )

        run_trackeval_script(
            trackeval_path,
            tracker_name="kalman_sort",
            save_path=metrics_path,
        )

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

    return output_txt_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--det_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--xml_gt_path", type=str, default=None)
    parser.add_argument("--trackeval_path", type=str, default=None)
    parser.add_argument("--make_video", action="store_true")

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--filter_threshold", type=float, default=0.9)
    parser.add_argument("--conf_threshold", type=float, default=0.5)

    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=5)
    parser.add_argument("--min_hits", type=int, default=3)

    parser.add_argument("--start_frame", type=int, default=900)
    parser.add_argument("--end_frame", type=int, default=970)

    args = parser.parse_args()

    main(
        det_path=args.det_path,
        output_txt_path=args.out,
        video_path=args.video_path,
        xml_gt_path=args.xml_gt_path,
        trackeval_path=args.trackeval_path,
        make_video=args.make_video,
        preprocess=args.preprocess,
        filter_threshold=args.filter_threshold,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        min_hits=args.min_hits,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
