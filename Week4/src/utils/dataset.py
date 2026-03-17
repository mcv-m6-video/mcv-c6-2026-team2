import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


class MOMCDataset:
    def __init__(self, root: str, seq: str, tracking_file: str):
        # Declare attributes (and Initialize if possible)
        self.data = self.__get_videos(root, seq, tracking_file)
        self.start_videos()

    def __getitem__(self, index: tuple[int, int]):
        camera_idx, frame_idx = index

        cap: cv2.VideoCapture = self.data["video_captures"][camera_idx]

        # Indexing wrt time instant 0. Check offset
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_offset = float(self.data["offsets"][camera_idx])
        frame_offset = time_offset * fps
        frame_idx = int(frame_idx - frame_offset)
        if frame_idx < 0:
            return None, None

        ret, frame = cap.read()
        if not ret:
            return None, None

        dets = self.data["dets"][camera_idx].get(frame_idx, [])

        return frame, dets

    def __get_videos(self, root: str, seq: str, tracking_file: str):
        print("Fetching videos and groundtruth...")
        data = {
            "videos": [],
            "video_captures": [],
            "rois": [],
            "dets": [],
            "homographies": [],
            "num_frames": [],
            "offsets": [],
            "cam_name": []
        }
        sequence_folder = os.path.join(root, "train", seq)
        for subfolder in tqdm(sorted(glob.glob(os.path.join(sequence_folder, "*")))):
            data["videos"].append(os.path.join(subfolder, "vdo.avi"))
            data["rois"].append(os.path.join(subfolder, "roi.jpg"))
            data["cam_name"].append(os.path.basename(subfolder))

            detections = self.__load_detections(
                os.path.join(subfolder, tracking_file)
            )
            data["dets"].append(detections)

            with open(os.path.join(subfolder, "calibration.txt"), "r") as f:
                homography_str = f.readline().strip().split(" ")[2:]
                homography_matrix = [[]]
                for element in homography_str:
                    subelements = element.split(";")
                    homography_matrix[-1].append(subelements[0])
                    if len(subelements) > 1:
                        homography_matrix.append([subelements[1]])
                homography = np.array(homography_matrix, dtype=np.float32)
                data["homographies"].append(homography)

        cam_framenum_file = os.path.join(root, "cam_framenum", seq + ".txt")
        with open(cam_framenum_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                num_frames = int(line.split(" ")[1])
                data["num_frames"].append(num_frames)

        cam_offset_file = os.path.join(root, "cam_timestamp", seq + ".txt")
        with open(cam_offset_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                offset = line.split(" ")[1]
                data["offsets"].append(offset)

        return data

    def __load_detections(self, file: str):
        detections: dict[int, list[str]] = {}
        with open(file, "r") as f:
            for det in f.readlines():
                det = det.strip()
                frame = int(det.split(",")[0])
                if frame not in detections:
                    detections[frame] = []
                detections[frame].append(det)

        return detections

    def start_videos(self):
        for cap in self.data["video_captures"]:
            cap.release()

        self.data["video_captures"] = []

        for video in self.data["videos"]:
            cap = cv2.VideoCapture(video)
            self.data["video_captures"].append(cap)

    def get_all_cameras(self):
        for idx in range(len(self.data["videos"])):
            yield self.get_cam_info(idx)

    def get_cam_info(self, idx: int):
        cap: cv2.VideoCapture = self.data["video_captures"][idx]
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        homography = self.data["homographies"][idx]
        num_frames = self.data["num_frames"][idx]
        offsets = self.data["offsets"][idx]

        roi_path = self.data["rois"][idx]
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        return homography, num_frames, offsets, (width, height), roi_mask

    def get_max_frame(self):
        frame_idx = np.argmax(self.data["num_frames"])
        max_frames = self.data["num_frames"][frame_idx]
        return max_frames
    
    def get_cam_names(self):
        return self.data["cam_name"]
