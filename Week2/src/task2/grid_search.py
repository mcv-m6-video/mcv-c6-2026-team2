import yaml
import itertools
import os
from src.task2.task21 import run_task21
from src.task2.kalman_filter import main as run_task22


def parse_metrics(metrics_path):
    hota = None
    idf1 = None

    section = None  # "HOTA" | "Identity" | other

    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detect which block we are in
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
                    hota = float(values[1])     # HOTA column
                elif section == "Identity" and idf1 is None:
                    idf1 = float(values[1])     # IDF1 column

            if hota is not None and idf1 is not None:
                break

    return hota, idf1


def run_grid_search(config_path, task):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    det_path = config["data"]["det_path"]
    video_path = config["data"]["video_path"]
    xml_gt_path = config["data"]["gt_xml_path"]
    trackeval_path = config["data"]["trackeval_path"]

    iou_list = config["tracking"]["iou_threshold"]
    max_age_list = config["tracking"]["max_age"]
    conf_list = config["tracking"]["conf_threshold"]
    filter_list = config["tracking"]["duplicate_iou_threshold"]
    min_hits_list = config["tracking"]["min_hits"]

    output_dir = config["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.csv")

    with open(summary_path, "w") as f:
        f.write("iou,max_age,conf_threshold,filter_thr,min_hits,HOTA,IDF1\n")

    for iou_thr, max_age, conf_thr, filter_thr, min_hits in itertools.product(iou_list, max_age_list, conf_list, filter_list, min_hits_list):

        print(
            f"Running: IoU={iou_thr}, max_age={max_age}, conf={conf_thr}, filter={filter_thr}, min_hits={min_hits}")

        output_txt = os.path.join(
            output_dir,
            f"iou{iou_thr}_age{max_age}_conf{conf_thr}_filter{filter_thr}_minhits{min_hits}.txt"
        )

        if task == '2.1':
            run_task21(
                det_path=det_path,
                output_txt_path=output_txt,
                video_path=None,
                xml_gt_path=xml_gt_path,
                trackeval_path=trackeval_path,
                make_video=False,
                iou_threshold=iou_thr,
                max_age=max_age,
                conf_threshold=conf_thr,
                filter_threshold=filter_thr
            )
        elif task == "2.2":
            run_task22(
                det_path=det_path,
                output_txt_path=output_txt,
                video_path=None,
                xml_gt_path=xml_gt_path,
                trackeval_path=trackeval_path,
                make_video=False,
                iou_threshold=iou_thr,
                max_age=max_age,
                conf_threshold=conf_thr,
                filter_threshold=filter_thr,
                preprocess=True,
                min_hits=min_hits
            )

        metrics_path = output_txt.replace(".txt", "_metrics.txt")
        hota, idf1 = parse_metrics(metrics_path)

        with open(summary_path, "a") as f:
            f.write(
                f"{iou_thr},{max_age},{conf_thr},{filter_thr},{min_hits},{hota},{idf1}\n")

        print(f" → HOTA={hota}, IDF1={idf1}")

    print("Grid search finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--task', type=str, default='2.1')
    args = parser.parse_args()

    print("Starting grid search...")
    run_grid_search(args.config, args.task)


# Example usage:
# python -m src.task2.grid_search --config config/config_task22_grid_search2.yaml --task 2.2
