import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset as HF_Dataset


class CustomDataset(torch.utils.data.Dataset):
    PATH_KEY = "img_path"
    ANNO_KEY = "annotations"
    CATEGORIES = {0: "LOG", 1: "WARNING"}

    def __init__(
        self,
        data_path: str,
        annotations_path: str,
        images_folder: str = None,
        transforms: any = None,
        hf: bool = False,
        log_level: int = 1,
    ):
        self.log_level = log_level
        self.transforms = transforms
        self.hf = hf
        self.data = self._load_data(
            data_path, annotations_path, images_folder=images_folder
        )

        if self.hf:
            self.data = self._convert_to_hf_format(self.data)

    def log(self, msg: str, level: int):
        if level <= self.log_level:
            print(f"{self.CATEGORIES[level]}: {msg}")

    def _process_video(self, video_path: str, images_folder: str):
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in tqdm(range(num_frames), total=num_frames):
            ret, image = cap.read()
            if not ret:
                break

            name = f"{idx:04d}.jpg"
            image_path = os.path.join(images_folder, name)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image.save(image_path)

    def _load_data(
        self, data_path: str, annotations_path: str, images_folder: str = None
    ):
        self.log("Loading data...", 0)
        is_dir = os.path.isdir(data_path)
        if not is_dir:
            ext = os.path.splitext(data_path)[1]
            if ext != ".avi":
                raise ValueError(
                    f"data_path must be either a folder with '.jpg' images or a '.avi' video. Instead it is {data_path}"
                )

            if images_folder:
                folder = images_folder
            else:
                dir = os.path.dirname(data_path)
                folder = os.path.join(dir, "images")

            os.makedirs(folder, exist_ok=True)
            if len(os.listdir(folder)) != 0:
                self.log(
                    f"Directory {folder} is not empty. This may cause some problems with the data.",
                    1,
                )

            self.log(
                f"Detected video. Starting to parse video. Contents will go into {folder}",
                0,
            )
            self._process_video(data_path, folder)
            self.log(f"Finished saving images into {folder}", 0)
            data_path = folder

        dataset_dict = defaultdict(lambda: {self.PATH_KEY: "", self.ANNO_KEY: []})

        tree = ET.parse(annotations_path)
        root = tree.getroot()

        self.log(f"Parsing annotations file {annotations_path}", 0)
        for track in tqdm(root.findall("track")):
            track_id = int(track.get("id"))
            label = track.get("label")

            if label != "car":
                continue

            for box in track.findall("box"):
                frame = int(box.get("frame"))
                outside = int(box.get("outside"))

                if outside == 1:
                    continue

                parked = False
                for attr in box.findall("attribute"):
                    if attr.get("name") == "parked" and attr.text == "true":
                        parked = True
                        break

                image_path = os.path.join(data_path, f"{frame:04d}.png")

                x = float(box.get("xtl"))
                y = float(box.get("ytl"))
                w = float(box.get("xbr")) - x
                h = float(box.get("ybr")) - y

                annotation = {
                    "track_id": track_id,
                    "frame": frame,
                    "label": label,
                    "parked": parked,
                    "bbox": np.array([x, y, w, h]),
                }

                dataset_dict[frame][self.PATH_KEY] = image_path
                dataset_dict[frame][self.ANNO_KEY].append(annotation)
        self.log("Finished loading dataset!", 0)
        return dict(dataset_dict)

    def _convert_to_hf_format(self, data_dict: dict):
        self.log("Formatting dataset for HF compatibility.", 0)
        hf_data = []

        for frame_idx, info in tqdm(data_dict.items(), total=len(data_dict)):
            bboxes = [anno["bbox"] for anno in info[self.ANNO_KEY]]
            labels = [anno["bbox"] for anno in info[self.ANNO_KEY]]
            areas = [b[2] * b[3] for b in bboxes]

            hf_data.append(
                {
                    "image_id": int(frame_idx),
                    "image": info[self.PATH_KEY],
                    "objects": {
                        "id": [int(anno["frame"]) for anno in info[self.ANNO_KEY]],
                        "area": areas,
                        "bbox": bboxes,
                        "category": labels,
                    },
                }
            )

        return hf_data

    def get_hf_dataset(self):
        if not self.hf:
            self.log("Dataset is not parsed for HF. Format may be incorrect.", 1)

        return HF_Dataset.from_list(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.hf and idx == 0:
            self.log("Dataset is parsed for HF. Format may be incorrect.", 1)
        image_path = self.data[idx][self.PATH_KEY]
        image = Image.open(image_path)

        annotation = self.data[idx][self.ANNO_KEY]

        return image, annotation


if __name__ == "__main__":
    d = CustomDataset(
        "Data/AICity_data/train/S03/c010/vdo.avi",
        "Data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
    )
