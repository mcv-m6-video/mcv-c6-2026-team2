import argparse
import logging
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml


DEFAULT_CONFIG = {
    "data_root": None,
    "output_dir": "./outputs/reid_cityflow",
    "backbone": "densenet121",
    "embedding_dim": 512,
    "img_size": 256,
    "train_sequences": "S01,S04",
    "val_sequences": "",
    "test_sequences": "S03",
    "val_ratio": 0.2,
    "epochs": 60,
    "batch_size": 32,
    "num_instances": 4,
    "lr": 3e-4,
    "weight_decay": 5e-4,
    "margin": 0.3,
    "ce_weight": 1.0,
    "tri_weight": 1.0,
    "num_workers": 4,
    "seed": 42,
    "log_interval": 50,
}


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: Optional[str]) -> Dict:
    """Loads training arguments from a YAML config file.

    Args:
        config_path: Path to a YAML file or None.

    Returns:
        A dictionary with config values.

    Raises:
        ValueError: If the YAML content is not a mapping or contains unknown keys.
    """
    if not config_path:
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must define a mapping of arguments: {config_path}")

    unknown_keys = sorted(set(config.keys()) - set(DEFAULT_CONFIG.keys()))
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}"
        )

    return config


def setup_logging() -> logging.Logger:
    """Configures and returns the module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("train_matcher")


def log_dataset_summary(logger: logging.Logger, name: str, dataset) -> None:
    """Logs a compact summary of a dataset or subset."""
    sequence_counts = defaultdict(int)
    camera_counts = defaultdict(int)
    for sample in dataset.samples:
        sequence_counts[sample.sequence] += 1
        camera_counts[sample.camid] += 1

    logger.info(
        "%s summary | samples=%d | ids=%d | sequences=%s | cameras=%d",
        name,
        len(dataset),
        len(dataset.pids),
        dict(sorted(sequence_counts.items())),
        len(camera_counts),
    )


def build_parser(defaults: Optional[Dict] = None) -> argparse.ArgumentParser:
    """Builds the CLI parser for training configuration."""
    parser = argparse.ArgumentParser(description="Train a vehicle ReID model on CityFlow-ReID")
    if defaults:
        parser.set_defaults(**defaults)

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument(
        "--data_root",
        type=str,
        required=defaults is None or defaults.get("data_root") is None,
        help="Path to the AI City dataset root or its train/ directory.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_CONFIG["backbone"],
        choices=["densenet121", "resnet50"],
    )
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_CONFIG["embedding_dim"])
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--train_sequences", type=str, default=DEFAULT_CONFIG["train_sequences"])
    parser.add_argument(
        "--val_sequences",
        type=str,
        default=DEFAULT_CONFIG["val_sequences"],
        help="Optional comma-separated validation sequences. If empty, validation is split from train sequences.",
    )
    parser.add_argument("--test_sequences", type=str, default=DEFAULT_CONFIG["test_sequences"])
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_CONFIG["val_ratio"],
        help="Fraction of train-sequence identities reserved for validation.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_instances", type=int, default=DEFAULT_CONFIG["num_instances"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--margin", type=float, default=DEFAULT_CONFIG["margin"])
    parser.add_argument("--ce_weight", type=float, default=DEFAULT_CONFIG["ce_weight"])
    parser.add_argument("--tri_weight", type=float, default=DEFAULT_CONFIG["tri_weight"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument(
        "--log_interval",
        type=int,
        default=DEFAULT_CONFIG["log_interval"],
        help="How many training batches between progress logs.",
    )
    return parser


def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes pairwise cosine distances between two embedding sets.

    Args:
        a: Tensor of shape [N, D].
        b: Tensor of shape [M, D].

    Returns:
        A tensor of shape [N, M] with cosine distances.
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = a @ b.t()
    return 1.0 - sim


class ReIDNet(nn.Module):
    """Backbone plus embedding and classification heads for vehicle ReID."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "densenet121",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.densenet121(weights=weights)
            feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.embedding = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """Computes normalized embeddings and classification logits."""
        feats = self.backbone(x)
        emb = self.embedding(feats)
        logits = self.classifier(emb)
        emb = F.normalize(emb, dim=1)
        return emb, logits


class BatchHardTripletLoss(nn.Module):
    """Implements batch-hard triplet loss for metric learning."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes triplet loss from embeddings and identity labels."""
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.unsqueeze(1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos

        diag = torch.eye(mask_pos.size(0), dtype=torch.bool, device=mask_pos.device)
        mask_pos = mask_pos & ~diag

        hardest_pos = []
        hardest_neg = []

        for i in range(dist.size(0)):
            pos_dists = dist[i][mask_pos[i]]
            neg_dists = dist[i][mask_neg[i]]

            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue

            hardest_pos.append(pos_dists.max())
            hardest_neg.append(neg_dists.min())

        if len(hardest_pos) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        hardest_pos = torch.stack(hardest_pos)
        hardest_neg = torch.stack(hardest_neg)

        return F.relu(hardest_pos - hardest_neg + self.margin).mean()


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    """Extracts embeddings and metadata for all samples in a loader."""
    model.eval()

    all_feats = []
    all_pids = []
    all_camids = []
    all_paths = []

    for batch in loader:
        imgs = batch["image"].to(device)
        feats, _ = model(imgs)

        all_feats.append(feats.cpu())
        all_pids.extend(batch["pid"].tolist())
        all_camids.extend(batch["camid"].tolist())
        all_paths.extend(batch["path"])

    return (
        torch.cat(all_feats, dim=0),
        np.array(all_pids),
        np.array(all_camids),
        all_paths,
    )


def compute_cmc_map(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 10,
) -> Tuple[np.ndarray, float]:
    """Computes CMC and mAP metrics for a ReID ranking task."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []

    for q_idx in range(num_q):
        order = indices[q_idx]
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precisions = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precisions * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    if len(all_cmc) == 0:
        raise RuntimeError("No valid query identities found during evaluation.")

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = float(np.mean(all_ap))
    return cmc, mAP


@torch.no_grad()
def evaluate(model, query_loader, gallery_loader, device):
    """Evaluates a model on query/gallery loaders and returns ranking metrics."""
    q_feats, q_pids, q_camids, _ = extract_features(model, query_loader, device)
    g_feats, g_pids, g_camids, _ = extract_features(model, gallery_loader, device)

    distmat = cosine_distance_matrix(q_feats, g_feats).cpu().numpy()
    cmc, mAP = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10)

    return {
        "mAP": mAP,
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]) if len(cmc) > 4 else float(cmc[-1]),
        "rank10": float(cmc[9]) if len(cmc) > 9 else float(cmc[-1]),
    }


def build_transforms(img_size: int = 256):
    """Builds training and evaluation image transforms."""
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
    )

    test_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return train_tfms, test_tfms


def train_one_epoch(
    model,
    loader,
    optimizer,
    ce_loss_fn,
    tri_loss_fn,
    device,
    ce_weight: float,
    tri_weight: float,
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    log_interval: int,
):
    """Runs one training epoch and returns averaged loss metrics."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_tri = 0.0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)

        emb, logits = model(imgs)

        ce_loss = ce_loss_fn(logits, labels)
        tri_loss = tri_loss_fn(emb, labels)
        loss = ce_weight * ce_loss + tri_weight * tri_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_tri += tri_loss.item()

        if batch_idx == 1 or batch_idx % max(log_interval, 1) == 0 or batch_idx == num_batches:
            logger.info(
                "Epoch %03d/%03d | batch %04d/%04d | loss=%.4f | ce=%.4f | tri=%.4f",
                epoch,
                total_epochs,
                batch_idx,
                num_batches,
                loss.item(),
                ce_loss.item(),
                tri_loss.item(),
            )

    n = len(loader)
    return {
        "loss": total_loss / max(n, 1),
        "ce_loss": total_ce / max(n, 1),
        "tri_loss": total_tri / max(n, 1),
    }
