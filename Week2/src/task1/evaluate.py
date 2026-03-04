from pprint import pprint

import albumentations as A
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from src.utils.dataset import CustomDataset
from src.utils.postprocess import filter_predictions, save_prediction_txt
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from tqdm import tqdm


def main(args):
    # Get relevant args
    model_name = args.model_name
    data_path = args.data_path
    annotations_path = args.annotations_path
    batch_size = args.batch_size
    log_wandb = args.log_wandb
    threshold = args.threshold
    checkpoint = args.checkpoint
    output_file = args.output_file
    split = args.split

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transforms = A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    # Load dataset
    eval_dataset = CustomDataset(
        data_path, annotations_path, split=split, transforms=transforms, log_level=1
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Load model
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.to(device=device)
    model.eval()
    """
    Output format:
    list of dicts:
    [
        {
            'boxes': (N, 4) tensor in device (unknown format)
            'labels': (N) tensor in device (int referring to each detection)
            'scores': (N) tensor in device (float 0 to 1 scoring confidence)
        }
    ]
    """
    id2label = {3: "car"}

    # Define metrics
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", backend="pycocotools"
    )
    metric.reset()

    # Start wandb project
    if log_wandb:
        wandb.init(
            project="C6-Week2",
            entity="c6-team2",
            name=f"Eval-{model_name}",
            config=args,
        )

    for idx, (images, targets) in enumerate(
        tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader))
    ):
        images = [img.to(device=device) for img in images]
        images_id = [t["image_id"].item() for t in targets]

        with torch.no_grad():
            prediction = model(images)
        prediction = filter_predictions(
            prediction, target_class=3, id2label=id2label, threshold=threshold
        )

        for pred, id in zip(prediction, images_id):
            save_prediction_txt(pred, output_file, id, threshold=threshold)

        metric.update(prediction, targets)

    results = metric.compute()
    pprint(results)

    if log_wandb:
        metrics_to_log = {
            "mAP/main": results["map"],          # AP @.50:.95
            "mAP/50": results["map_50"],         # AP @.50
            "mAP/75": results["map_75"],         # AP @.75
            "mAP/small": results["map_small"],
            "mAP/medium": results["map_medium"],
            "mAP/large": results["map_large"],
            "mAR/Det1": results["mar_1"],        # AR
            "mAR/Det10": results["mar_10"],
            "mAR/Det100": results["mar_100"],
            "mAR/small": results["mar_small"],
            "mAR/medium": results["mar_medium"],
            "mAR/large": results["mar_large"],
        }
        wandb.log(metrics_to_log)


if __name__ == "__main__":
    main()
