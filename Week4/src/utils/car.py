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
            return True
            # ---------------------------------------------------------
            # DUMMY BASELINE: HSV Color Histogram Comparison
            # ---------------------------------------------------------
            if not isinstance(other, Car):
                return NotImplemented

            # 1. Safety check: Ensure both cars actually have image crops
            if self.image is None or other.image is None:
                return False

            # 2. Safety check: Ensure crops aren't empty (0x0 pixels from edge clipping)
            if self.image.size == 0 or other.image.size == 0:
                return False

            # 3. Convert both images from BGR to HSV
            hsv1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(other.image, cv2.COLOR_BGR2HSV)

            # 4. Calculate 2D histograms for Hue (Color) and Saturation (Intensity)
            # We use 50 bins for Hue and 60 bins for Saturation
            hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

            # 5. Normalize! (Crucial so a giant truck and a tiny car far away can be compared fairly)
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # 6. Compare using Correlation (Returns 1.0 for a perfect match, down to -1.0)
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # 7. Set your threshold! (0.85 to 0.90 is usually a good starting point)
            MATCH_THRESHOLD = 0.85
            
            return similarity >= MATCH_THRESHOLD

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
