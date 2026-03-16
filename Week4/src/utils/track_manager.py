from .car import Car


class TrackManager:
    def __init__(self):
        self.local_to_global: dict[tuple[int, int], int] = {}
        self.global_tracks: dict[int, dict[int, Car]] = {}
        self.next_global_id = 0

    def register_car(self, camera_idx: int, local_car_id: int, car_instance: Car):
        if (camera_idx, local_car_id) not in self.local_to_global:
            global_id = self.next_global_id
            self.local_to_global[(camera_idx, local_car_id)] = global_id
            self.global_tracks[global_id] = {camera_idx: car_instance}
            self.next_global_id += 1

    def link_cars(
        self, cam_idx1: int, cam_idx2: int, local_car_id1: int, local_car_id2: int
    ):
        global_id1 = self.local_to_global[(cam_idx1, local_car_id1)]
        global_id2 = self.local_to_global[(cam_idx2, local_car_id2)]

        if global_id1 == global_id2:
            return

        self.__merge_cars(global_id1, global_id2)

    def __merge_cars(self, keep_id: int, merge_id: int):
        for cam_idx, car_instance in self.global_tracks[merge_id].items():
            self.global_tracks[keep_id][cam_idx] = car_instance
            self.local_to_global[(cam_idx, car_instance.car_id)] = keep_id

        del self.global_tracks[merge_id]
