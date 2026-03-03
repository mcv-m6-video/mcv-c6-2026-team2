from src.utils.dataset import CustomDataset
from src.utils.postprocess import filter_predictions, save_prediction_txt
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
import wandb
from tqdm import tqdm
from pprint import pprint


def main(args):
    # Get relevant args
    model_name = args.model_name
    data_path = args.data_path
    annotations_path = args.annotations_path
    batch_size = args.batch_size
    log_wandb = args.log_wandb
    threshold = args.threshold
    output_file = args.output_file

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transforms = None

    # Load dataset
    eval_dataset = CustomDataset(
        data_path, annotations_path, transforms=transforms, log_level=1
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
    id2label = {
        3: "car"
    }

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
        prediction = filter_predictions(prediction, target_class=3, id2label=id2label, threshold=threshold)

        for pred, id in zip(prediction, images_id):
            save_prediction_txt(pred, output_file, id, threshold=threshold)

        metric.update(prediction, targets)

    results = metric.compute()
    pprint(results)


if __name__ == "__main__":
    main()
