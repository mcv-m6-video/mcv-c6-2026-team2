from itertools import product
import os
import csv

import cv2
from tqdm import tqdm
import numpy as np

from task1.task1 import (Video,
                         load_ground_truth,
                         gt_to_coco,
                         preds_to_coco,
                         evaluate_coco,
                         get_bounding_boxes,
                         remove_nested_boxes,
                         merge_overlapping_boxes)
from task2.task2 import (apply_morphological_filtering,
                         detect)
from utils import create_detection_gif, create_detection_video
# from bgsCNN.bgsCNN_v5 import bgsCNN_v5


fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2()
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg_lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()
fgbg_gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg_knn = cv2.createBackgroundSubtractorKNN()
fgbg_cnt = cv2.bgsegm.createBackgroundSubtractorCNT()
# bgsCNN = bgsCNN_v5()


models = [
    # (fgbg_mog, "MOG"),
    (fgbg_mog2, "MOG2"),
    # (fgbg_gmg, "GMG"),
    # (fgbg_lsbp, "LSBP"),
    # (fgbg_gsoc, "GSOC"),
    # (fgbg_knn, "KNN"),
    # (bgsCNN, "bgsCNN")
]


def run_task3(args):
    print(f"Running Task 2: State-of-The-Art model comparison")

    video = Video(args.video)
    train_size = int(video.num_frames * 0.25)
    roi = cv2.imread(
        args.video.replace("vdo.avi", "roi.jpg"),
        cv2.IMREAD_GRAYSCALE
    )
    roi = roi > 0

    gt_dict = load_ground_truth(args.annotations)
    gt_json = gt_to_coco(gt_dict, train_size, video.num_frames - train_size)

    do_grid = len(args.post_process) > 1
    save_results = bool(getattr(args, "config", None)) and do_grid

    results_list = [] if save_results else None

    for model, model_name in models:
        print(f"\nTesting model: {model_name}")
        best_boxes = None
        best_post_process = None
        best_ap50 = -1

        for min_area, open_size, close_size, post_process in product(
            args.min_area, args.open_size, args.close_size,
                args.post_process):
            print(f"Testing post_process={post_process}...")
            all_pred_boxes = []
            video.reset()

            for idx, frame in tqdm(enumerate(video.get_next_raw_frame()),
                                   total=video.num_frames):
                if idx < train_size:
                    _ = model.apply(frame)
                else:
                    mask = model.apply(frame)
                    mask = mask * roi
                    # we only want to keep values exactly 255, some models
                    # predict shadows as 126 -> we want to remove those
                    mask = (mask == 255).astype(np.uint8) * 255

                    if int(post_process) == 1:
                        mask = apply_morphological_filtering(
                            mask, open_size, close_size)
                    elif int(post_process) == 2:
                        # GMG specific post-processing
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (3, 3))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # detect bounding boxes
                    boxes = detect(mask_proc=mask, min_area=min_area)
                    all_pred_boxes.append(boxes)

            pred_json = preds_to_coco(all_pred_boxes, train_size)

            ap50 = evaluate_coco(gt_json, pred_json)

            if save_results:
                results_list.append({
                    'model': model_name,
                    'post_process': post_process,
                    'ap50': ap50
                })

            if ap50 > best_ap50:
                best_ap50 = ap50
                best_post_process = post_process
                best_boxes = all_pred_boxes

        if save_results:
            output_dir = f"{args.task}/results"
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(
                output_dir, f"{os.path.basename(args.config).split('.')[0]}.csv")

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['model', 'post_process', 'ap50'])
                writer.writeheader()
                writer.writerows(results_list)

            print(f"\nGrid Search Finished. Results saved to: {csv_path}")

        print(f"\nModel: {model_name}")
        print(f"\nFinal Adaptive AP50: {best_ap50:.4f}")
        print(f"\nFinal post-process: {best_post_process}")

        if args.gif:
            prefix = f"detections_{args.task}_"
            prefix = prefix + model_name
            file_name = prefix + f"_ap50_{best_ap50:.04f}.mp4"
            output_folder = os.path.join(args.task, 'results', 'videos')
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(
                args.task, 'results', 'videos', file_name)

            video.close()

            create_detection_video(
                args.video,
                best_boxes,
                gt_dict,
                train_size,
                output_path=output_path
            )

    video.close()
