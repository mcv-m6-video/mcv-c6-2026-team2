import cv2
import numpy as np
from shapely.geometry import Polygon, Point


class Car:
    def __init__(self, car_id: int):
        # Declare attributes (and Initialize if possible)
        self.car_id: int = car_id  # ID of the detected car
        self.pixel_bbox: list[np.ndarray] = []  # Bboxes in frames coordinates
        self.gps_bbox: list[Polygon] = []  # Bboxes in GPS coordinates
        self.frame_idx: list[int] = []  # Indexes (independent from camera)
        self.cam_idx: int = None  # Camera index
        self.image: np.ndarray = None  # Last detected instance
        self.confidence: list[float] = []  # Confidences of the bbox in each frame

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
        confidence: int,
    ):
        self.image = image
        self.pixel_bbox.append(bbox)

        xleft, ytop, xright, ybottom = bbox
        xcenter = (xleft - xright) / 2.0

        pts = np.array([[[xcenter, ybottom]]], dtype=np.float32)

        gps_point = cv2.perspectiveTransform(pts, homography).squeeze()

        radius_meters = 2.0

        car_footprint_polygon = Point(gps_point).buffer(radius_meters)

        self.gps_bbox.append(car_footprint_polygon)
        self.frame_idx.append(frame_idx)
        self.confidence.append(confidence)
        self.cam_idx = camera_idx

    def get_history(self):
        detections: list[list] = []
        for i in range(len(self.pixel_bbox)):
            frame_idx = self.frame_idx[i]
            xleft, ytop, xright, ybottom = self.pixel_bbox[i]
            conf = self.confidence[i]

            d = [frame_idx, xleft, ytop, xright, ybottom, conf]
            detections.append(d)

        return detections
