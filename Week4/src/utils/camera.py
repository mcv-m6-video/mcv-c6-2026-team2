import cv2
import numpy as np

from .car import Car


class Camera:
    def __init__(
        self,
        camera_idx: int,
        resolution: tuple[int, int],
        homography: np.ndarray,
        offset: float,
        num_frames: int,
    ):
        # Declare attributes (and Initialize if possible)
        self.camera_idx = camera_idx  # Camera index (ID)
        self.resolution = resolution  # Video resolution (W, H)
        self.homography = homography  # Homography that maps to GPS coordinates
        self.offset = (
            offset  # Temporal offset from respect to an initial time (seconds)
        )
        self.num_frames = num_frames

        self.gps_bbox = self.compute_gps_bbox(
            resolution, homography
        )  # Bbox in GPS coordinates
        self.overlapping_cameras = []  # List of overlapping cameras
        self.adjacent_cameras = []  # List of adjacent cameras

    def __contains__(self, other):
        # Vaya locura estoy COOKING
        # This checks whether two cameras overlap or a detected car appears in this camera
        if isinstance(other, Car) or isinstance(other, Camera):
            self_xleft, self_ytop, self_xright, self_ybottom = self.gps_bbox

            other_bbox = (
                other.gps_bbox[-1] if isinstance(other, Car) else other.gps_bbox
            )
            other_xleft, other_ytop, other_xright, other_ybottom = other_bbox

            xleft = max(other_xleft, self_xleft)
            ytop = max(other_ytop, self_ytop)
            xright = max(other_xright, self_xright)
            ybottom = max(other_ybottom, self_ybottom)

            return xright >= xleft and ybottom >= ytop

        return NotImplemented

    def compute_gps_bbox(self, resolution: tuple[int, int], homography: np.ndarray):
        width, height = resolution
        bbox = np.array(
            [[[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]],
            dtype=np.float32,
        )
        gps_bbox = cv2.perspectiveTransform(bbox, homography)
        return gps_bbox

    def add_overlapping_camera(self, cam):
        self.overlapping_cameras.append(cam)

    def add_adjacent_camera(self, cam):
        self.adjacent_cameras.append(cam)
