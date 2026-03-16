import cv2
import numpy as np
from shapely import intersects
from shapely.geometry import LineString, Polygon

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

        self.gps_polygon, self.centroid = self.compute_gps_bbox(
            resolution, homography
        )  # Bbox in GPS coordinates
        self.overlapping_cameras = []  # List of overlapping cameras
        self.adjacent_cameras = []  # List of adjacent cameras

    def __contains__(self, item):
        if isinstance(item, Car):
            return self.gps_polygon.intersects(item.gps_bbox[-1])
        return NotImplemented

    def compute_gps_bbox(self, resolution: tuple[int, int], homography: np.ndarray):
        width, height = resolution
        bbox = np.array(
            [[[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]],
            dtype=np.float32,
        )
        gps_polygon = cv2.perspectiveTransform(bbox, homography).squeeze()
        gps_polygon = Polygon(gps_polygon)
        return gps_polygon, gps_polygon.centroid

    def add_overlapping_camera(self, cam):
        self.overlapping_cameras.append(cam)

    def add_adjacent_camera(self, cam):
        self.adjacent_cameras.append(cam)


def compute_relationships(camera_list: list[Camera]):
    num_cams = len(camera_list)

    for i in range(num_cams):
        cam_i = camera_list[i]

        for j in range(i + 1, num_cams):
            cam_j = camera_list[j]

            if cam_i.gps_polygon.intersects(cam_j.gps_polygon):
                camera_list[i].add_overlapping_camera(camera_list[j])
                camera_list[j].add_overlapping_camera(camera_list[i])
                continue

            los_line = LineString([cam_i.centroid, cam_j.centroid])
            is_blocked = False

            for k in range(num_cams):
                if k == i or k == j:
                    continue

                cam_k = camera_list[k]

                if los_line.intersects(cam_k.gps_polygon):
                    is_blocked = True
                    break

            if not is_blocked:
                camera_list[i].add_adjacent_camera(camera_list[j])
                camera_list[j].add_adjacent_camera(camera_list[i])

    return camera_list
