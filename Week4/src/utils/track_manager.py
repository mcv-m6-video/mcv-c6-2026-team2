import os

from .car import Car


class TrackManager:
    def __init__(self):
        self.local_to_global: dict[tuple[int, int], int] = {}
        # A global track may contain several local tracks from the same camera
        # after incorrect merges, so we keep them nested by camera and local id.
        self.global_tracks: dict[int, dict[int, dict[int, Car]]] = {}
        self.next_global_id = 0

    def register_car(self, camera_idx: int, local_car_id: int, car_instance: Car):
        if (camera_idx, local_car_id) not in self.local_to_global:
            global_id = self.next_global_id
            self.local_to_global[(camera_idx, local_car_id)] = global_id
            self.global_tracks[global_id] = {camera_idx: {local_car_id: car_instance}}
            self.next_global_id += 1

    def link_cars(
        self, cam_idx1: int, local_car_id1: int, cam_idx2: int, local_car_id2: int
    ):
        global_id1 = self.local_to_global[(cam_idx1, local_car_id1)]
        global_id2 = self.local_to_global[(cam_idx2, local_car_id2)]

        if global_id1 == global_id2:
            return

        self.__merge_cars(global_id1, global_id2)

    def __merge_cars(self, keep_id: int, merge_id: int):
        if keep_id not in self.global_tracks or merge_id not in self.global_tracks:
            raise KeyError(
                f"Cannot merge tracks keep_id={keep_id}, merge_id={merge_id}; "
                f"available global ids: {sorted(self.global_tracks.keys())}"
            )

        for cam_idx, local_tracks in self.global_tracks[merge_id].items():
            if cam_idx not in self.global_tracks[keep_id]:
                self.global_tracks[keep_id][cam_idx] = {}

            for local_car_id, car_instance in local_tracks.items():
                self.global_tracks[keep_id][cam_idx][local_car_id] = car_instance
                self.local_to_global[(cam_idx, local_car_id)] = keep_id

        del self.global_tracks[merge_id]

    def save(self, output_folder: str, cam_names: list[str] = None):
        os.makedirs(output_folder, exist_ok=True)
        file = os.path.join(output_folder, "pred.txt")

        detections: list[str] = []
        for dict_cam_idx, cam_name in enumerate(cam_names):
            cam_idx = int(cam_name[1:]) - 1

            cam_dets: list[str] = []
            for global_id, car_registry in self.global_tracks.items():
                if len(car_registry) < 2 or dict_cam_idx not in car_registry:
                    continue

                for car_instance in car_registry[dict_cam_idx].values():
                    dets = car_instance.get_history()
                    for d in dets:
                        frame_idx, xleft, ytop, xright, ybottom, conf = d
                        formated_det = f"{cam_idx},{global_id},{frame_idx},{xleft},{ytop},{xright - xleft},{ybottom - ytop},-1,-1\n"
                        cam_dets.append(formated_det)

            cam_dets.sort(key=lambda x: int(x.split(",")[1]))
            detections.extend(cam_dets)

        with open(file, "w") as f:
            f.writelines(detections)
