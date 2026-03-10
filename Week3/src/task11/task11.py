import os

import cv2
import numpy as np
import pandas as pd
import pyflow
from PIL import Image
from src.models import (
    BaseModel,
    FarnebackModel,
    MEMFOFModel,
    PerceiverModel,
    PyflowModel,
)
from src.utils.kitti_dataset import kitti_of_gt_processing
from src.utils.metrics import evaluate
from src.utils.visualizations import generate_flow_reference, save_flow


def main(args):
    # Process arguments
    dataset_path = args.dataset_path
    output_path = args.output_path
    image_id = args.image_id
    num_iters = args.num_iters

    subfolder = "image_0"

    # Load pair of images
    image10_path = os.path.join(
        dataset_path, "training", subfolder, f"{image_id:06d}_10.png"
    )
    image11_path = os.path.join(
        dataset_path, "training", subfolder, f"{image_id:06d}_11.png"
    )

    image10 = np.array(Image.open(image10_path))
    image11 = np.array(Image.open(image11_path))

    inputs = [image10, image11]

    # Load and save (visualization) non-occluded groundtruth
    gt_nocc_path = os.path.join(
        dataset_path, "training", "flow_noc", f"{image_id:06d}_10.png"
    )
    gt_nocc = cv2.imread(gt_nocc_path, cv2.IMREAD_UNCHANGED)
    gt_nocc, valid_mask = kitti_of_gt_processing(gt_nocc)

    # Save non-occluded groundtruth
    gt_path = os.path.join(output_path, "gt.png")
    save_flow(gt_nocc, gt_path)
    print(f"Save gt image in {gt_path}")

    # Generate and save reference flow (color guide)
    ref_flow = generate_flow_reference((512, 512))
    ref_path = os.path.join(output_path, "ref.png")
    save_flow(ref_flow, ref_path)
    print(f"Saved reference image in {ref_path}")

    # Define methods to try
    methods: list[tuple[any, str, BaseModel]] = [
        (
            PyflowModel(args),
            "pyflow",
        ),
        (
            FarnebackModel(args),
            "farneback",
        ),
        (
            PerceiverModel(args),
            "perceiverio",
        ),
        (
            MEMFOFModel(args),
            "memfof"
        ),
    ]

    # Define results dict
    results = {
        "method": [],
        "msen": [],
        "pepn": [],
        "mean_runtime": [],
        "std_runtime": [],
        "efficiency": [],
        "num_iters": [],
    }

    # Compute optical flow with pyflow
    for method, name in methods:
        output, results = evaluate(
            method,
            inputs,
            gt_nocc,
            name,
            results,
            mask=valid_mask,
            num_iters=num_iters,
        )

        # Save predicted flows
        pred_path = os.path.join(output_path, f"pred_{name}.png")
        save_flow(output, pred_path)
        print(f"Saved predicted image in {pred_path}")

    # Save results
    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(output_path, "results.csv"))
    print(f"Saved results CSV {os.path.join(output_path)}")
