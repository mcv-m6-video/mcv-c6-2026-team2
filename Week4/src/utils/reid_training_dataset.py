import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler


@dataclass
class Sample:
    """Stores the metadata required to recover one vehicle crop."""

    path: str
    pid: int
    camid: int
    frame: int
    bbox: Tuple[float, float, float, float]
    sequence: str


def parse_camid_from_name(path: str) -> int:
    """Parses a camera id from a file or folder name.

    Args:
        path: File or directory name containing a camera token.

    Returns:
        The parsed camera id, or -1 if no pattern matches.
    """
    name = Path(path).name.lower()

    patterns = [
        r"c(\d+)",
        r"cam(?:era)?[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


class ReIDFolderDataset(Dataset):
    """Loads ReID samples from AI City videos and ground-truth boxes.

    The dataset reads `vdo.avi` and `gt/gt.txt` for each camera and builds
    crop metadata lazily, so frames are extracted only when requested.
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

                gt_data = defaultdict(list)
                with open(gt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 6:
                            continue
                        frame = int(parts[0])
                        track_id = int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        gt_data[track_id].append((frame, x, y, w, h))

                for track_id, detections in gt_data.items():
                    pid = self._build_pid(seq, track_id)
                    for frame, x, y, w, h in detections:
                        self.samples.append(
                            Sample(
                                path=str(vdo_path),
                                pid=pid,
                                camid=camid,
                                frame=frame,
                                bbox=(x, y, w, h),
                                sequence=seq,
                            )
                        )
                    self.pid_set.add(pid)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in sequences {self.sequences}")

        self.pids = sorted(self.pid_set)
        self.pid2label = {pid: idx for idx, pid in enumerate(self.pids)}

    @staticmethod
    def _resolve_root(root: str) -> Path:
        """Resolves either a dataset root or a direct `train/` path."""
        root_path = Path(root)
        if (root_path / "train").exists():
            return root_path / "train"
        return root_path

    @staticmethod
    def _build_pid(sequence: str, track_id: int) -> int:
        """Creates a sequence-aware identity id from a local track id."""
        seq_num = int(re.sub(r"\D", "", sequence))
        return seq_num * 1_000_000 + track_id

    def _read_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Reads one frame from a video, reusing open captures when possible."""
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
        """Loads, crops, and transforms a single dataset sample."""
        frame = self._read_frame(sample.path, sample.frame - 1)
        if frame is None:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)

        x, y, w, h = sample.bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        crop = frame[y : y + h, x : x + w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop)
        if transform is None:
            transform = self.transform
        if transform is not None:
            img = transform(img)

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
    """Wraps a base dataset with a subset of indices and optional remapping."""

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
        self.pid2label = (
            pid2label
            if pid2label is not None
            else {pid: idx for idx, pid in enumerate(self.pids)}
        )

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
    """Splits samples into train and validation sets by identity."""
    pid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        pid_to_indices[sample.pid].append(index)

    pids = list(pid_to_indices.keys())
    random_state = np.random.RandomState(seed)
    random_state.shuffle(pids)

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
    """Builds simple query/gallery splits from grouped pid-camera samples."""
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


class RandomIdentitySampler(Sampler):
    """Samples batches with multiple instances per identity.

    This is the standard P x K sampling pattern used by triplet-loss training.
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
            np.random.shuffle(idxs)

            batch = []
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.num_instances:
                    batch_idxs_dict[pid].append(batch)
                    batch = []

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_indices = np.random.choice(
                len(avai_pids), size=self.num_pids_per_batch, replace=False
            )
            selected_pids = [avai_pids[idx] for idx in selected_indices]
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self) -> int:
        return self.length
