import numpy as np

from .of_model import BaseModel


class Sort:
    def __init__(self):
        pass

class Track:
    def __init__(self, bbox, id: int):
        self.id = id
        self.bboxes = [bbox]
        self.misses = 0

    def last_bbox(self):
        return self.bboxes[-1]

    def update(self, bbox):
        self.bboxes.append(bbox)
        self.misses = 0


class OFTracker:
    def __init__(
        self,
        obj_detector,
        of: BaseModel,
        iou_threshold: float = 0.4,
        max_age: int = 5,
        conf_threshold: float = 0.5,
    ):
        self.obj_detector = obj_detector
        self.of = of

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.conf_threshold = conf_threshold

    def detect_and_filter(self, image: np.ndarray):
        dets = self.obj_detector(image)
        dets = [d for d in dets if d[4] >= self.conf_threshold]
        return dets

    def online_tracking(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        dets0: list = None,
        labels1: list[int] = None,
    ):

        pass

    def offline_tracking(self):
        pass
