import cv2
import numpy as np

from shapely.geometry import Polygon


class Car:
    def __init__(self, car_id: int):
        # Declare attributes (and Initialize if possible)
        self.car_id = car_id  # ID of the detected car
        self.pixel_bbox = []  # Bboxes in frames coordinates
        self.gps_bbox = []  # Bboxes in GPS coordinates
        self.frame_idx = []  # Indexes (independent from camera)
        self.cam_idx = None  # Camera index
        self.image = None  # Last detected instance

        # Momentum like direction, to get possible next camera to check.
        # Maybe I can compute this on demand?
        self.gps_direction = None

    def __eq__(self, other):
        # This checks if two cars are the same (uses the siamese network)
        if isinstance(other, Car):
            # TODO: Use model for matching
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

        pts = np.array(
            [[
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ]],
            dtype=np.float32,
        )

        gps_coords = cv2.perspectiveTransform(pts, homography).squeeze()
        self.gps_bbox.append(Polygon(gps_coords))

        self.frame_idx.append(frame_idx)
        self.cam_idx = camera_idx

