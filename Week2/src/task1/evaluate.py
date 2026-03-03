from src.utils.dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import wandb
from tqdm import tqdm
from pprint import pprint

def main(args):
    # Get relevant args
    model_name = args.model_name
    data_path = args.data_path
    annotations_path = args.annotation_path
    batch_size = args.batch_size
    log_wandb = args.log_wandb
    threshold = args.threshold

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transforms = None

    # Load dataset
    eval_dataset = CustomDataset(
        data_path, annotations_path, transforms=transforms, log_level=1
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x))
    )

    # Load model
    model = fasterrcnn_resnet50_fpn_v2(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.to(device=device)

    id2label = {model.config.label2id["car"]: "car"}

    # Define metrics
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")

    # Start wandb project
    if log_wandb:
        wandb.init(
            project="C6-Week2",
            entity="c6-team2",
            name=f"Eval-{model_name}",
            config=args
        )
    
    for idx, (images, targets) in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        prediction = model(images)
        pprint(prediction)
        return


if __name__ == "__main__":
    main()
