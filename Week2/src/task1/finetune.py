import os
from pprint import pprint

import torch
import wandb
from src.utils.dataset import CustomDataset
from src.utils.postprocess import filter_predictions, save_prediction_txt
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import Compose
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def main(args):
    # Get relevant args
    model_name = args.model_name
    data_path = args.data_path
    annotations_path = args.annotations_path
    batch_size = args.batch_size
    log_wandb = args.log_wandb
    threshold = args.threshold
    epochs = args.epochs
    lr = args.lr
    eval_steps = args.eval_steps
    unfreeze_depth = args.unfreeze_depth
    patience = args.patience

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    train_transforms = A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    eval_transforms = A.Compose(
        [ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    # Load datasetç
    train_dataset = CustomDataset(
        data_path,
        annotations_path,
        split="train",
        transforms=train_transforms,
        log_level=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    eval_dataset = CustomDataset(
        data_path,
        annotations_path,
        split="eval",
        transforms=eval_transforms,
        log_level=1,
    )
    eval_loader = DataLoader(
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
    # Freeze everything first
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Progressive unfreezing
    if unfreeze_depth >= 1:
        for p in model.backbone.body.layer4.parameters():
            p.requires_grad = True

    if unfreeze_depth >= 2:
        for p in model.backbone.body.layer3.parameters():
            p.requires_grad = True

    if unfreeze_depth >= 3:
        for p in model.backbone.body.layer2.parameters():
            p.requires_grad = True

    if unfreeze_depth >= 4:
        for p in model.backbone.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
            name=f"Finetune-{model_name}",
            config=args,
        )

    best_map = 0
    epochs_no_improve = 0

    for e in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {e} [Train]")

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        lr_scheduler.step()

        print(f"\n{'=' * 30}")
        print(f" Epoch {e} Summary ")
        print(f"{'=' * 30}")
        print(f" Train Loss: {epoch_train_loss / len(train_loader):.4f}")

        if log_wandb:
            metrics_to_log = {
                "train/loss": epoch_train_loss / len(train_loader),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }

        if e % eval_steps == 0:
            model.eval()
            metric.reset()
            for idx, (images, targets) in enumerate(
                tqdm(eval_loader, desc="Evaluating", total=len(eval_loader))
            ):
                images = [img.to(device=device) for img in images]

                with torch.no_grad():
                    prediction = model(images)
                prediction = filter_predictions(
                    prediction, target_class=3, id2label=id2label, threshold=threshold
                )

                metric.update(prediction, targets)

            results = metric.compute()
            pprint(results)

            if log_wandb:
                metrics_to_log.update(
                    {
                        "mAP/main": results["map"],
                        "mAP/50": results["map_50"],
                        "mAP/75": results["map_75"],
                        "mAP/class_Car": results["map_per_class"][0],
                        "mAP/class_Pedestrian": results["map_per_class"][1],
                        "mAP/small": results["map_small"],
                        "mAP/medium": results["map_medium"],
                        "mAP/large": results["map_large"],
                        "mAR/Det1": results["mar_1"],
                        "mAR/Det10": results["mar_10"],
                        "mAR/Det100": results["mar_100"],
                        "mAR/small": results["mar_small"],
                        "mAR/medium": results["mar_medium"],
                        "mAR/large": results["mar_large"],
                        "mAR/Det100_class_Car": results["mar_100_per_class"][0],
                        "mAR/Det100_class_Pedestrian": results["mar_100_per_class"][1],
                    }
                )
                wandb.log(metrics_to_log)

            current_map = results["map"].item()
            if current_map > best_map:
                best_map = current_map
                epochs_no_improve = 0

                os.makedirs(f"checkpoints/{unfreeze_depth}/{lr}", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"checkpoints/{unfreeze_depth}/{args.lr}/fasterrcnn_{model_name}_best.pth",
                )
                print(f"New Best mAP: {best_map:.4f}. Checkpoint saved.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
