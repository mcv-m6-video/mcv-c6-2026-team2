import cv2
import numpy as np


class Car:
    def __init__(self, car_id: int):
        # Declare attributes (and Initialize if possible)
        self.car_id = car_id  # ID of the detected car
        self.pixel_bbox = []  # Bboxes in frames coordinates
        self.gps_bbox = []  # Bboxes in GPS coordinates
        self.frame_idx = []  # Indexes (independent from camera)
        self.cam_idx = []  # Camera indexes
        self.image = None  # Last detected instance

        # Momentum like direction, to get possible next camera to check.
        # Maybe I can compute this on demand?
        self.gps_direction = None

    def __eq__(self, other):
        # This checks if two cars are the same (uses the siamese network, OHBOI I'M COOKING)
        if isinstance(other, Car):
            # TODO: Use model for matching (I'm a genious)
            pass
        return NotImplemented

    def add_detection(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        homography: np.ndarray,
        frame_idx: int,
        camera_idx: int,
    ):
        self.image = image
        self.pixel_bbox.append(bbox)

        if bbox.ndim == 2:
            bbox = bbox[np.newaxis, ...]

        gps_coords = cv2.perspectiveTransform(bbox, homography)
        self.gps_bbox.append(gps_coords)

        self.frame_idx.append(frame_idx)
        self.cam_idx.append(camera_idx)

    def merge_cars(self, other):
        if isinstance(other, Car):
            self.pixel_bbox.extend(other.pixel_bbox)
            self.gps_bbox.extend(other.gps_bbox)
            self.frame_idx.extend(other.frame_idx)
            self.cam_idx.extend(other.cam_idx)
