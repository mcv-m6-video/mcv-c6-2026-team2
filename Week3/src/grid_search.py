import argparse
import itertools
import os
import csv
import yaml

from . import t12


def parse_metrics(metrics_path: str):
    """
    Parse HOTA and IDF1 from TrackEval metrics file.
    """
    hota = None
    idf1 = None
    section = None

    if not os.path.exists(metrics_path):
        print(f"[WARNING] Metrics file not found: {metrics_path}")
        return None, None

    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("HOTA:"):
                section = "HOTA"
                continue
            if line.startswith("Identity:"):
                section = "Identity"
                continue
            if line.startswith("CLEAR:"):
                section = "CLEAR"
                continue

            if line.startswith("COMBINED"):
                values = line.split()
                if section == "HOTA" and hota is None:
                    hota = float(values[1])
                elif section == "Identity" and idf1 is None:
                    idf1 = float(values[1])

            if hota is not None and idf1 is not None:
                break

    return hota, idf1


def run_grid_search(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # General paths
    dataset_path = config.get("dataset_path", "datasets/KITTI")
    output_dir = config.get("output_path", "results/task12_grid_search")
    os.makedirs(output_dir, exist_ok=True)

    # Fixed arguments for task12
    fixed = config.get("fixed_args", {})
    of_method = fixed.get("of_method", "farneback")
    obj_detector_path = fixed.get(
        "obj_detector_path",
        "models/fasterrcnn_faster-rcnn_best.pth"
    )
    video_path = fixed.get(
        "video_path",
        "datasets/AICity_data/train/S03/c010/vdo.avi"
    )
    mode = fixed.get("mode", "online")
    predominant_of_method = fixed.get("predominant_of_method", "median")
    xml_gt_path = fixed.get(
        "xml_gt_path",
        "datasets/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml"
    )
    trackeval_path = fixed.get("trackeval_path", "src/utils/TrackEval")
    make_video = fixed.get("make_video", False)
    start_frame = fixed.get("start_frame", 200)
    end_frame = fixed.get("end_frame", 270)
    video_out_default = fixed.get("video_out", None)

    # Optical flow method-specific params
    pf_alpha = fixed.get("pf_alpha", 0.012)
    pf_ratio = fixed.get("pf_ratio", 0.75)
    pf_minWidth = fixed.get("pf_minWidth", 20)
    pf_nOuterFPIters = fixed.get("pf_nOuterFPIters", 7)
    pf_nInnerFPIters = fixed.get("pf_nInnerFPIters", 1)
    pf_nSORIters = fixed.get("pf_nSORIters", 30)
    pf_colType = fixed.get("pf_colType", 0)

    fb_pyrScale = fixed.get("fb_pyrScale", 0.5)
    fb_levels = fixed.get("fb_levels", 3)
    fb_winSize = fixed.get("fb_winSize", 15)
    fb_iters = fixed.get("fb_iters", 3)
    fb_polyN = fixed.get("fb_polyN", 5)
    fb_polySigma = fixed.get("fb_polySigma", 1.2)

    perc_path = fixed.get("perc_path", "deepmind/optical-flow-perceiver")
    mf_path = fixed.get("mf_path", "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")

    # Grid parameters
    grid = config["grid_search"]
    iou_list = grid["iou_threshold"]
    dup_iou_list = grid["dup_iou_threshold"]
    max_age_list = grid["max_age"]
    min_hits_list = grid["min_hits"]
    conf_list = grid["conf_threshold"]

    summary_path = os.path.join(output_dir, "summary.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iou_threshold",
            "dup_iou_threshold",
            "max_age",
            "min_hits",
            "conf_threshold",
            "HOTA",
            "IDF1",
            "output_txt_path"
        ])

    combinations = itertools.product(
        iou_list,
        dup_iou_list,
        max_age_list,
        min_hits_list,
        conf_list
    )

    for iou_thr, dup_iou_thr, max_age, min_hits, conf_thr in combinations:
        print(
            f"Running: "
            f"IoU={iou_thr}, "
            f"dup_IoU={dup_iou_thr}, "
            f"max_age={max_age}, "
            f"min_hits={min_hits}, "
            f"conf={conf_thr}"
        )

        run_name = (
            f"iou{iou_thr}_dup{dup_iou_thr}_age{max_age}"
            f"_minhits{min_hits}_conf{conf_thr}"
        )

        output_txt_path = os.path.join(output_dir, f"{run_name}.txt")
        video_out = (
            os.path.join(output_dir, f"{run_name}.mp4")
            if make_video
            else (video_out_default or os.path.join(output_dir, f"{run_name}.mp4"))
        )

        args = argparse.Namespace(
            # general
            config=None,
            dataset_path=dataset_path,
            output_path=output_dir,

            # task12 args
            of_method=of_method,
            obj_detector_path=obj_detector_path,
            video_path=video_path,
            mode=mode,
            iou_threshold=float(iou_thr),
            dup_iou_threshold=float(dup_iou_thr),
            max_age=int(max_age),
            min_hits=int(min_hits),
            conf_threshold=float(conf_thr),
            predominant_of_method=predominant_of_method,
            output_txt_path=output_txt_path,
            video_out=video_out,
            xml_gt_path=xml_gt_path,
            trackeval_path=trackeval_path,
            make_video=make_video,
            start_frame=int(start_frame),
            end_frame=int(end_frame),

            # pyflow params
            pf_alpha=float(pf_alpha),
            pf_ratio=float(pf_ratio),
            pf_minWidth=int(pf_minWidth),
            pf_nOuterFPIters=int(pf_nOuterFPIters),
            pf_nInnerFPIters=int(pf_nInnerFPIters),
            pf_nSORIters=int(pf_nSORIters),
            pf_colType=int(pf_colType),

            # farneback params
            fb_pyrScale=float(fb_pyrScale),
            fb_levels=int(fb_levels),
            fb_winSize=int(fb_winSize),
            fb_iters=int(fb_iters),
            fb_polyN=int(fb_polyN),
            fb_polySigma=float(fb_polySigma),

            # perceiver / MEMFOF
            perc_path=perc_path,
            mf_path=mf_path,
        )

        try:
            t12(args)

            metrics_path = output_txt_path.replace(".txt", "_metrics.txt")
            hota, idf1 = parse_metrics(metrics_path)

            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    iou_thr,
                    dup_iou_thr,
                    max_age,
                    min_hits,
                    conf_thr,
                    hota,
                    idf1,
                    output_txt_path
                ])

            print(f" -> HOTA={hota}, IDF1={idf1}")

        except Exception as e:
            print(f"[ERROR] Failed run {run_name}: {e}")

            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    iou_thr,
                    dup_iou_thr,
                    max_age,
                    min_hits,
                    conf_thr,
                    None,
                    None,
                    output_txt_path
                ])

    print("Grid search finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print("Starting optical flow tracker grid search...")
    run_grid_search(args.config)


# usage example:
# python -m src.grid_search --config configs/config_task12_grid_search.yaml