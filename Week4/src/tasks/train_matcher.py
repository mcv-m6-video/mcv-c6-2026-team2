import argparse
import logging
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models

import wandb


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_config(config_path: Optional[str]) -> Dict:
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("train_matcher")


def log_dataset_summary(logger: logging.Logger, name: str, dataset) -> None:
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
    parser = argparse.ArgumentParser(description="Train a vehicle ReID model on CityFlow-ReID")
    if defaults:
        parser.set_defaults(**defaults)

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--data_root", type=str, required=defaults is None or defaults.get("data_root") is None, help="Path to the AI City dataset root or its train/ directory.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--backbone", type=str, default=DEFAULT_CONFIG["backbone"], choices=["densenet121", "resnet50"])
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_CONFIG["embedding_dim"])
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--train_sequences", type=str, default=DEFAULT_CONFIG["train_sequences"])
    parser.add_argument("--val_sequences", type=str, default=DEFAULT_CONFIG["val_sequences"], help="Optional comma-separated validation sequences. If empty, validation is split from train sequences.")
    parser.add_argument("--test_sequences", type=str, default=DEFAULT_CONFIG["test_sequences"])
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_CONFIG["val_ratio"], help="Fraction of train-sequence identities reserved for validation.")
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
    parser.add_argument("--log_interval", type=int, default=DEFAULT_CONFIG["log_interval"], help="How many training batches between progress logs.")
    return parser


@dataclass
class Sample:
    path: str
    pid: int
    camid: int
    frame: int
    bbox: Tuple[float, float, float, float]
    sequence: str


def parse_camid_from_name(path: str) -> int:
    """
    Attempts to parse camera id from filename patterns such as:
    c001_xxx.jpg, cam12_xxx.jpg, camera_3_xxx.jpg
    Returns -1 if not found.
    """
    name = Path(path).name.lower()

    patterns = [
        r'c(\d+)',
        r'cam(?:era)?[_-]?(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Returns cosine distance = 1 - cosine similarity.
    a: [N, D]
    b: [M, D]
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    sim = a @ b.t()
    return 1.0 - sim


# ============================================================
# Dataset
# ============================================================

class ReIDFolderDataset(Dataset):
    """
    Loads vehicle crops from AI City Challenge videos and ground truth.
    Expects:
      root/
        S01/
          c001/
            vdo.avi
            gt/gt.txt
          ...
        S04/
          ...
    """
    def __init__(self, root: str, sequences: List[str], transform=None):
        self.root = self._resolve_root(root)
        self.sequences = sequences
        self.transform = transform

        self.samples: List[Sample] = []
        self.pid_set = set()
        self._video_caps: Dict[str, cv2.VideoCapture] = {}

        for seq in self.sequences:
            seq_path = self.root / seq
            if not seq_path.exists():
                continue
            for cam_dir in seq_path.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name
                camid = parse_camid_from_name(cam_name)
                vdo_path = cam_dir / "vdo.avi"
                gt_path = cam_dir / "gt" / "gt.txt"
                if not vdo_path.exists() or not gt_path.exists():
                    continue

                # Load ground truth
                gt_data = defaultdict(list)
                with open(gt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 6:
                            continue
                        frame = int(parts[0])
                        track_id = int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        gt_data[track_id].append((frame, x, y, w, h))

                # Create samples for each detection
                for track_id, detections in gt_data.items():
                    # Track IDs are sequence-local; namespace them so different
                    # sequences cannot accidentally share the same identity label.
                    pid = self._build_pid(seq, track_id)
                    for frame, x, y, w, h in detections:
                        self.samples.append(Sample(
                            path=str(vdo_path),
                            pid=pid,
                            camid=camid,
                            frame=frame,
                            bbox=(x, y, w, h),
                            sequence=seq,
                        ))
                    self.pid_set.add(pid)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in sequences {self.sequences}")

        self.pids = sorted(self.pid_set)
        self.pid2label = {pid: idx for idx, pid in enumerate(self.pids)}

    @staticmethod
    def _resolve_root(root: str) -> Path:
        root_path = Path(root)
        if (root_path / "train").exists():
            return root_path / "train"
        return root_path

    @staticmethod
    def _build_pid(sequence: str, track_id: int) -> int:
        seq_num = int(re.sub(r"\D", "", sequence))
        return seq_num * 1_000_000 + track_id

    def _read_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        cap = self._video_caps.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            self._video_caps[video_path] = cap

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(
        self,
        sample: Sample,
        transform=None,
        pid2label: Optional[Dict[int, int]] = None,
    ):
        frame = self._read_frame(sample.path, sample.frame - 1)  # GT frames are 1-based
        if frame is None:
            # Return a blank image or handle error
            frame = np.zeros((256, 256, 3), dtype=np.uint8)

        x, y, w, h = sample.bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Ensure bbox is within frame
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        crop = frame[y:y+h, x:x+w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop)
        if transform is None:
            transform = self.transform
        if transform is not None:
            img = transform(img)

        # Label is remapped only for training classification
        if pid2label is None:
            pid2label = self.pid2label
        label = pid2label.get(sample.pid, sample.pid)
        return {
            "image": img,
            "pid": sample.pid,
            "label": label,
            "camid": sample.camid,
            "path": sample.path,
            "sequence": sample.sequence,
        }

    def __getitem__(self, index: int):
        return self._load_sample(self.samples[index])


class ReIDSubset(Dataset):
    def __init__(
        self,
        dataset: ReIDFolderDataset,
        indices: List[int],
        transform=None,
        pid2label: Optional[Dict[int, int]] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.samples = [dataset.samples[idx] for idx in indices]
        self.pids = sorted({sample.pid for sample in self.samples})
        self.pid2label = pid2label if pid2label is not None else {
            pid: idx for idx, pid in enumerate(self.pids)
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset._load_sample(
            self.samples[index],
            transform=self.transform,
            pid2label=self.pid2label,
        )


def split_indices_by_pid(
    samples: List[Sample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        pid_to_indices[sample.pid].append(index)

    pids = list(pid_to_indices.keys())
    random.Random(seed).shuffle(pids)

    num_val_pids = max(1, int(len(pids) * val_ratio)) if val_ratio > 0 else 0
    val_pid_set = set(pids[:num_val_pids])

    train_indices = []
    val_indices = []
    for pid, indices in pid_to_indices.items():
        if pid in val_pid_set:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return sorted(train_indices), sorted(val_indices)


def build_query_gallery_indices(samples: List[Sample]) -> Tuple[List[int], List[int]]:
    pid_cam_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        pid_cam_to_indices[(sample.pid, sample.camid)].append(index)

    query_indices = []
    gallery_indices = []
    for indices in pid_cam_to_indices.values():
        indices = sorted(indices, key=lambda idx: samples[idx].frame)
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:] if len(indices) > 1 else indices)

    if not query_indices or not gallery_indices:
        raise RuntimeError("Could not build query/gallery split from the selected samples.")

    return sorted(set(query_indices)), sorted(gallery_indices)


# ============================================================
# PK Sampler
# ============================================================

class RandomIdentitySampler(Sampler):
    """
    Samples P identities and K images per identity.
    Batch size = P * K
    """
    def __init__(self, dataset: ReIDFolderDataset, batch_size: int, num_instances: int):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.index_dic: Dict[int, List[int]] = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            self.index_dic[sample.pid].append(index)

        self.pids = list(self.index_dic.keys())

        # Approximate epoch length
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            n = len(idxs)
            if n < self.num_instances:
                n = self.num_instances
            self.length += n - n % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)

            batch = []
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.num_instances:
                    batch_idxs_dict[pid].append(batch)
                    batch = []

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self) -> int:
        return self.length


# ============================================================
# Model
# ============================================================

class ReIDNet(nn.Module):
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
        feats = self.backbone(x)
        emb = self.embedding(feats)
        logits = self.classifier(emb)
        emb = F.normalize(emb, dim=1)
        return emb, logits


# ============================================================
# Losses
# ============================================================

class BatchHardTripletLoss(nn.Module):
    """
    Standard batch-hard triplet loss.
    Assumes batches are formed with multiple instances per identity.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        labels = labels.unsqueeze(1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos

        # remove self-comparisons from positives
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

        loss = F.relu(hardest_pos - hardest_neg + self.margin).mean()
        return loss


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
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
    """
    Standard ReID evaluation.
    Excludes gallery images with same pid and same camid as query.
    """
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

        # AP
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


# ============================================================
# Training
# ============================================================

def build_transforms(img_size: int = 256):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
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


def main():
    logger = setup_logging()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    config_values = load_config(config_args.config)
    parser_defaults = DEFAULT_CONFIG.copy()
    parser_defaults.update(config_values)

    parser = build_parser(parser_defaults)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config

    if args.config:
        logger.info("Loaded config file: %s", args.config)
    else:
        logger.info("Running without config file; using defaults and CLI arguments.")
    logger.info("Resolved arguments: %s", vars(args))

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Seed set to %d", args.seed)
    logger.info("Output directory: %s", args.output_dir)

    # Initialize wandb
    wandb.init(
        project="vehicle-reid",
        name=f"{args.backbone}_emb{args.embedding_dim}_lr{args.lr}",
        config=vars(args),
    )
    logger.info("Weights & Biases run initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_tfms, test_tfms = build_transforms(args.img_size)
    train_sequences = [seq.strip() for seq in args.train_sequences.split(",") if seq.strip()]
    val_sequences = [seq.strip() for seq in args.val_sequences.split(",") if seq.strip()]
    test_sequences = [seq.strip() for seq in args.test_sequences.split(",") if seq.strip()]
    logger.info(
        "Sequence selection | train=%s | val=%s | test=%s",
        train_sequences,
        val_sequences if val_sequences else "<split from train>",
        test_sequences,
    )

    base_train_set = ReIDFolderDataset(args.data_root, sequences=train_sequences, transform=None)
    log_dataset_summary(logger, "Base train pool", base_train_set)
    val_query_set = None
    val_gallery_set = None

    if val_sequences:
        logger.info("Using dedicated validation sequences.")
        train_set = ReIDSubset(
            dataset=base_train_set,
            indices=list(range(len(base_train_set.samples))),
            transform=train_tfms,
        )

        base_val_set = ReIDFolderDataset(args.data_root, sequences=val_sequences, transform=None)
        log_dataset_summary(logger, "Validation pool", base_val_set)
        val_query_indices, val_gallery_indices = build_query_gallery_indices(base_val_set.samples)
        val_query_set = ReIDSubset(
            dataset=base_val_set,
            indices=val_query_indices,
            transform=test_tfms,
        )
        val_gallery_set = ReIDSubset(
            dataset=base_val_set,
            indices=val_gallery_indices,
            transform=test_tfms,
        )
    else:
        logger.info("Splitting validation identities from training pool with val_ratio=%.3f", args.val_ratio)
        train_indices, val_indices = split_indices_by_pid(
            base_train_set.samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        logger.info(
            "Identity split complete | train_samples=%d | val_samples=%d",
            len(train_indices),
            len(val_indices),
        )

        train_samples = [base_train_set.samples[idx] for idx in train_indices]
        train_pid2label = {pid: idx for idx, pid in enumerate(sorted({sample.pid for sample in train_samples}))}
        train_set = ReIDSubset(
            dataset=base_train_set,
            indices=train_indices,
            transform=train_tfms,
            pid2label=train_pid2label,
        )

        if val_indices:
            val_query_indices, val_gallery_indices = build_query_gallery_indices(
                [base_train_set.samples[idx] for idx in val_indices]
            )
            val_query_set = ReIDSubset(
                dataset=base_train_set,
                indices=[val_indices[idx] for idx in val_query_indices],
                transform=test_tfms,
            )
            val_gallery_set = ReIDSubset(
                dataset=base_train_set,
                indices=[val_indices[idx] for idx in val_gallery_indices],
                transform=test_tfms,
            )

    base_test_set = ReIDFolderDataset(args.data_root, sequences=test_sequences, transform=None)
    log_dataset_summary(logger, "Test pool", base_test_set)
    test_query_indices, test_gallery_indices = build_query_gallery_indices(base_test_set.samples)
    test_query_set = ReIDSubset(
        dataset=base_test_set,
        indices=test_query_indices,
        transform=test_tfms,
    )
    test_gallery_set = ReIDSubset(
        dataset=base_test_set,
        indices=test_gallery_indices,
        transform=test_tfms,
    )
    log_dataset_summary(logger, "Train subset", train_set)
    if val_query_set is not None and val_gallery_set is not None:
        log_dataset_summary(logger, "Validation query", val_query_set)
        log_dataset_summary(logger, "Validation gallery", val_gallery_set)
    log_dataset_summary(logger, "Test query", test_query_set)
    log_dataset_summary(logger, "Test gallery", test_gallery_set)

    train_sampler = RandomIdentitySampler(
        dataset=train_set,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_query_loader = None
    val_gallery_loader = None
    if val_query_set is not None and val_gallery_set is not None:
        val_query_loader = DataLoader(
            val_query_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_gallery_loader = DataLoader(
            val_gallery_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    test_query_loader = DataLoader(
        test_query_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_gallery_loader = DataLoader(
        test_gallery_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(
        "Data loaders ready | train_batches=%d | val_query_batches=%s | val_gallery_batches=%s | test_query_batches=%d | test_gallery_batches=%d",
        len(train_loader),
        len(val_query_loader) if val_query_loader is not None else "n/a",
        len(val_gallery_loader) if val_gallery_loader is not None else "n/a",
        len(test_query_loader),
        len(test_gallery_loader),
    )

    num_classes = len(train_set.pids)
    logger.info("Training classes: %d", num_classes)

    model = ReIDNet(
        num_classes=num_classes,
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=True,
    ).to(device)
    logger.info(
        "Model initialized | backbone=%s | embedding_dim=%d",
        args.backbone,
        args.embedding_dim,
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    tri_loss_fn = BatchHardTripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    logger.info(
        "Optimizer and scheduler ready | lr=%.6f | weight_decay=%.6f | margin=%.3f",
        args.lr,
        args.weight_decay,
        args.margin,
    )

    best_rank1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %03d/%03d", epoch, args.epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            tri_loss_fn=tri_loss_fn,
            device=device,
            ce_weight=args.ce_weight,
            tri_weight=args.tri_weight,
            logger=logger,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )

        scheduler.step()
        logger.info(
            "Finished epoch %03d/%03d | avg_loss=%.4f | avg_ce=%.4f | avg_tri=%.4f | lr=%.6f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["ce_loss"],
            train_metrics["tri_loss"],
            scheduler.get_last_lr()[0],
        )

        eval_metrics = None
        if val_query_loader is not None and val_gallery_loader is not None:
            logger.info("Running validation for epoch %03d", epoch)
            eval_metrics = evaluate(model, val_query_loader, val_gallery_loader, device)

        if eval_metrics is not None:
            logger.info(
                "Validation results | epoch=%03d | mAP=%.4f | R1=%.4f | R5=%.4f | R10=%.4f",
                epoch,
                eval_metrics["mAP"],
                eval_metrics["rank1"],
                eval_metrics["rank5"],
                eval_metrics["rank10"],
            )

        # Log to wandb
        wandb_payload = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/ce_loss": train_metrics["ce_loss"],
            "train/tri_loss": train_metrics["tri_loss"],
            "lr": scheduler.get_last_lr()[0],
        }
        if eval_metrics is not None:
            wandb_payload.update({
                "val/mAP": eval_metrics["mAP"],
                "val/rank1": eval_metrics["rank1"],
                "val/rank5": eval_metrics["rank5"],
                "val/rank10": eval_metrics["rank10"],
            })
        wandb.log(wandb_payload)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "eval_metrics": eval_metrics,
            "train_pid2label": train_set.pid2label,
        }

        torch.save(ckpt, os.path.join(args.output_dir, "last_model.pth"))

        current_rank1 = eval_metrics["rank1"] if eval_metrics is not None else -train_metrics["loss"]
        if current_rank1 > best_rank1:
            best_rank1 = current_rank1
            torch.save(ckpt, best_path)
            logger.info("Saved new best checkpoint to %s", best_path)

    logger.info("Loading best checkpoint from %s", best_path)
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Running final test evaluation on sequences: %s", test_sequences)
    test_metrics = evaluate(model, test_query_loader, test_gallery_loader, device)

    logger.info("Training finished.")
    if val_query_loader is not None and val_gallery_loader is not None:
        logger.info("Best validation Rank-1: %.4f", best_rank1)
    logger.info(
        "Test results | sequences=%s | mAP=%.4f | R1=%.4f | R5=%.4f | R10=%.4f",
        ",".join(test_sequences),
        test_metrics["mAP"],
        test_metrics["rank1"],
        test_metrics["rank5"],
        test_metrics["rank10"],
    )

    wandb.log({
        "test/mAP": test_metrics["mAP"],
        "test/rank1": test_metrics["rank1"],
        "test/rank5": test_metrics["rank5"],
        "test/rank10": test_metrics["rank10"],
    })

    wandb.finish()


if __name__ == "__main__":
    main()
