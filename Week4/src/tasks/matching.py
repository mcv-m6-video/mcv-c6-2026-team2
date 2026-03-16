import numpy as np
from src.utils.camera import Camera, compute_relationships
from src.utils.car import Car
from src.utils.dataset import MOMCDataset
from src.utils.track_manager import TrackManager


def evaluate_car_state(
    car: Car,
    curr_cam: Camera,
    t_manager: TrackManager,
    border_threshold_pixels: int = 50,
    overlap_threshold: float = 0.8,
):
    # Testing overlapping cameras
    valid_overlap_cams: list[Camera] = []
    for overlap_cam in curr_cam.overlapping_cameras:
        if car in overlap_cam:
            intersection_area = (
                car.gps_bbox[-1].intersection(overlap_cam.gps_polygon).area
            )
            car_area = car.gps_bbox[-1].area

            if car_area > 0:
                overlap_ratio = intersection_area / car_area

                if overlap_ratio >= overlap_threshold:
                    global_id = t_manager.local_to_global[
                        (curr_cam.camera_idx, car.car_id)
                    ]
                    already_linked = False

                    if overlap_cam.camera_idx in t_manager.global_tracks[global_id]:
                        already_linked = True

                    if not already_linked:
                        valid_overlap_cams.append(overlap_cam)

    if len(valid_overlap_cams) > 0:
        return {
            "action": "OVERLAP",
            "target_cameras": valid_overlap_cams,
            "life_frames": 3,
        }

    # Testing if going to adjacent cameras
    width, height = curr_cam.resolution
    xleft, ytop, xright, ybottom = car.pixel_bbox[-1]

    dist_left = xleft
    dist_right = width - xright
    dist_top = ytop
    dist_bottom = height - ybottom

    min_dist_to_border = min(dist_left, dist_right, dist_top, dist_bottom)

    if min_dist_to_border < border_threshold_pixels:
        return {
            "action": "ADJACENCY",
            "target_cameras": curr_cam.adjacent_cameras,
            "life_frames": 30,
        }

    # Car won't appear in other cameras
    return {
        "action": "IGNORE",
        "target_cameras": [],
        "life_frames": 0,
    }


def main(args):
    # Extract arguments
    dataset_root = args.dataset_root
    seq = args.seq

    match_checkpoint = args.match_checkpoint
    output_folder = args.output_folder

    # Create dataset
    dataset = MOMCDataset(dataset_root, seq)

    # Initialize TrackingManager
    t_manager = TrackManager()

    # Initialize CameraManager
    cam_list: list[Camera] = [
        Camera(idx, resolution, homography, offset, num_frames)
        for idx, (homography, offset, num_frames, resolution) in enumerate(
            dataset.get_all_cameras()
        )
    ]
    cam_list = compute_relationships(cam_list)

    frame_idx = 0

    local_cars_registry: dict[int, dict[int, Car]] = {
        i: {} for i in range(len(cam_list))
    }
    overlap_queue = {}
    adjacency_queue = {}

    while True:
        active_cars_current_frame: dict[int, list[Car]] = {
            i: [] for i in range(len(cam_list))
        }
        valid_frame_read = False

        # Register cars
        for cam_idx in range(len(cam_list)):
            frame, dets = dataset[cam_idx, frame_idx]

            if frame is None or dets is None:
                continue

            valid_frame_read = True

            for d in dets:
                f_idx, car_id, xleft, ytop, xright, ybottom, confidence, _, _, _ = (
                    d.split()
                )

                car_id = int(car_id)
                bbox = np.array(
                    [float(xleft), float(ytop), float(xright), float(ybottom)],
                    dtype=np.float32,
                )

                car_image = frame[int(ytop) : int(ybottom), int(xleft) : int(xright)]

                if car_id not in local_cars_registry[cam_idx]:
                    local_cars_registry[cam_idx][car_id] = Car(car_id)

                car = local_cars_registry[cam_idx][car_id]

                car.add_detection(
                    car_image,
                    bbox,
                    cam_list[cam_idx].homography,
                    int(f_idx),
                    cam_idx,
                    confidence,
                )

                t_manager.register_car(cam_idx, car_id, car)
                active_cars_current_frame[cam_idx].append(car)

        # Check if car overlaps / is going to another camera
        for cam_idx, active_cars in active_cars_current_frame.items():
            current_cam = cam_list[cam_idx]

            for car in active_cars:
                state = evaluate_car_state(car, current_cam, t_manager)

                if state["action"] == "OVERLAP":
                    overlap_queue[(cam_idx, car.car_id)] = {
                        "car": car,
                        "targets": state["target_cameras"],
                        "expires_at": frame_idx + state["life_frames"],
                    }
                elif state["action"] == "ADJACENCY":
                    adjacency_queue[(cam_idx, car.car_id)] = {
                        "car": car,
                        "targets": state["target_cameras"],
                        "expires_at": frame_idx + state["life_frames"],
                    }

        # Look for matching cars
        for queue in [overlap_queue, adjacency_queue]:
            keys_to_remove: list[tuple[int, int]] = []

            for key, queue_info in queue.items():
                source_cam_idx, source_car_id = key
                source_car: Car = queue_info["car"]

                cams_to_remove_from_targets: list[Camera] = []

                for target_cam in queue_info["targets"]:
                    target_active_cars: list[Car] = active_cars_current_frame[
                        target_cam.camera_idx
                    ]
                    matched = False

                    for target_car in target_active_cars:
                        global_id_source: int = t_manager.local_to_global[
                            (source_cam_idx, source_car_id)
                        ]
                        global_id_target: int = t_manager.local_to_global[
                            (target_cam.camera_idx, target_car.car_id)
                        ]

                        if global_id_source == global_id_target:
                            # Already matched
                            matched = True
                            break
                        if source_car == target_car:
                            t_manager.link_cars(
                                source_cam_idx,
                                source_car_id,
                                target_cam.camera_idx,
                                target_car.car_id,
                            )
                            matched = True
                            break

                    if matched:
                        cams_to_remove_from_targets.append(target_cam)

                for resolved_camera in cams_to_remove_from_targets:
                    if resolved_camera in queue_info["targets"]:
                        queue_info["targets"].remove(resolved_camera)

                if (
                    frame_idx >= queue_info["expires_at"]
                    or len(queue_info["targets"]) == 0
                ):
                    keys_to_remove.append(key)

            for k in keys_to_remove:
                del queue[k]

        # Check if the loop needs to end
        if not valid_frame_read:
            break

        # Move to the next frame
        frame_idx += 1

    # Save detections per camera
    t_manager.save(output_folder, len(cam_list), cam_names=dataset.get_cam_names())
