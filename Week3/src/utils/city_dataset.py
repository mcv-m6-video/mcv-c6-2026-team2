import glob
import os

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from tqdm import tqdm


class AICityDataset:
    def __init__(
        self, root: str, video_idx: int = 0, transforms: Compose | None = None
    ):
        self.active_video = None
        self.active_gt = None
        self.roid = None
        self.active_idx = -1
        self.transforms = transforms

        self.data = self._get_data(root)
        self.change_active_track(video_idx)

    def _get_data(self, root: str):
        print("Fetching videos and groundtruth...")
        data = {"video": [], "gt": [], "roi": []}

        for subfolder in tqdm(sorted(glob.glob(os.path.join(root, "*")))):
            data["video"].append(os.path.join(subfolder, "vdo.avi"))
            data["roi"].append(os.path.join(subfolder, "roi.jpg"))
            data["gt"].append(os.path.join(subfolder, "gt", "gt.txt"))

        return data

    def change_active_track(self, idx: int):
        self.close()

        self.active_idx = idx

        self.active_video = cv2.VideoCapture(self.data["video"][idx])

        roi = np.array(Image.open(self.data["roi"][idx]))
        self.active_roi = (roi > 0)[..., np.newaxis]

        self.active_gt = self.data["gt"][idx]

    def get_video_stream(self):
        if self.active_video is None:
            return

        num_frames = int(self.active_video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(num_frames):
            ret, frame = self.active_video.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            valid_frame = frame * self.active_roi

            yield valid_frame

    def get_active_video_path(self):
        return self.data["video"][self.active_idx]
    
    def get_gt(self):
        return self.active_gt

    def close(self):
        if self.active_video is not None:
            self.active_video.release()
            self.active_video = None

        if self.active_gt is not None:
            self.active_gt = None

        if self.active_roi is not None:
            self.active_roi = None

    def __len__(self):
        return len(self.data["video"])


if __name__ == "__main__":
    dataset = AICityDataset("datasets/AI_CITY_CHALLENGE_2022_TRAIN/train/S01")

    for idx, (frame, gt) in enumerate(dataset.get_video_stream()):
        print(frame.shape)
        print(frame.dtype)
        print(frame.max())
        print(frame.min())
        print(gt)

        if idx == 1:
            break
