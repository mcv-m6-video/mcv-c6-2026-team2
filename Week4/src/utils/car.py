import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from src.models.matcher import compare_car_embeddings, get_matcher


class Car:
    def __init__(self, car_id: int):
        # Declare attributes (and Initialize if possible)
        self.car_id: int = car_id  # ID of the detected car
        self.pixel_bbox: list[np.ndarray] = []  # Bboxes in frames coordinates
        self.gps_bbox: list[Polygon] = []  # Bboxes in GPS coordinates
        self.frame_idx: list[int] = []  # Indexes (independent from camera)
        self.cam_idx: int = None  # Camera index
        self.image: np.ndarray = None  # Last detected instance
        # Confidences of the bbox in each frame
        self.confidence: list[float] = []
        # Embeddings for all detected instances
        self.embeddings: list[np.ndarray] = []

        # Momentum like direction, to get possible next camera to check.
        # Maybe I can compute this on demand?
        self.gps_direction = None

    def __eq__(self, other):
        # This checks if two cars are the same (uses the siamese network)
        if isinstance(other, Car):
            if len(self.embeddings) == 0 or len(other.embeddings) == 0:
                return False
            return compare_car_embeddings(self.embeddings, other.embeddings)
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
        matcher = get_matcher()
        if matcher is not None and image is not None and image.size > 0:
            embedding = matcher.embed_image(image).detach().cpu().numpy()
            self.embeddings.append(embedding)
        self.pixel_bbox.append(bbox)

        xleft, ytop, xright, ybottom = bbox
        xcenter = (xleft + xright) / 2.0

        pts = np.array([[[xcenter, ybottom]]], dtype=np.float32)

        gps_point = cv2.perspectiveTransform(pts, homography).squeeze()

        radius_meters = 2.0
        radius_degrees = radius_meters / 111139.0

        car_footprint_polygon = Point(gps_point).buffer(radius_degrees)

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
