import json
import math
import os

import numpy as np
from src.models.matcher import initialize_matcher
from src.utils.camera import Camera, compute_relationships
from src.utils.car import Car
from src.utils.dataset import MOMCDataset
from src.utils.track_manager import TrackManager
from src.utils.visualization import (
    export_camera_graph,
    visualize_camera_graph,
    visualize_geospatial_graph,
    visualize_spatial_filter
)
from tqdm import tqdm


def evaluate_car_state(
    car: Car,
    curr_cam: Camera,
    t_manager: TrackManager,
    border_threshold_pixels: int = 50,
    overlap_threshold: float = 0.5,
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
            "life_frames": 10,
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
            "target_cameras": list(curr_cam.adjacent_cameras),
            "life_frames": 100,
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
    detections_root = args.detections_root
    seq = args.seq

    match_checkpoint = args.match_checkpoint
    match_threshold = args.match_threshold

    tracking_file = args.tracking_file
    output_folder = args.output_folder

    initialize_matcher(match_checkpoint, similarity_threshold=match_threshold)

    # Create dataset
    dataset = MOMCDataset(dataset_root, detections_root, seq, tracking_file)

    # Initialize TrackingManager
    t_manager = TrackManager()

    # Initialize CameraManager
    cam_list: list[Camera] = [
        Camera(idx, resolution, homography, offset, num_frames, roi_mask)
        for idx, (homography, num_frames, offset, resolution, roi_mask) in enumerate(
            dataset.get_all_cameras()
        )
    ]
    cam_list = compute_relationships(cam_list)

    graph_json_file = os.path.join(output_folder, "visuals", "camera_graph.json")
    graph_image_file = os.path.join(output_folder, "visuals", "camera_graph.png")
    geograph_image_file = os.path.join(output_folder, "visuals", "geo_camera_graph.png")
    
    export_camera_graph(
        cam_list,
        output_file=graph_json_file,
    )

    visualize_camera_graph(graph_json_file, output_file=graph_image_file)
    visualize_geospatial_graph(graph_json_file, output_file=geograph_image_file)

    frame_idx = 1

    local_cars_registry: dict[int, dict[int, Car]] = {
        i: {} for i in range(len(cam_list))
    }
    overlap_queue = {}
    adjacency_queue = {}

    pbar = tqdm(total=dataset.get_max_frame(), desc="Tracking vehicles", unit="frame")
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
                f_idx, car_id, xleft, ytop, width, height, confidence, _, _, _ = (
                    d.split(",")
                )
                xleft = float(xleft)
                ytop = float(ytop)
                xright = xleft + float(width)
                ybottom = ytop + float(height)

                car_id = int(car_id)
                bbox = np.array(
                    [xleft, ytop, xright, ybottom],
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
        for qtype, queue in [
            ("OVERLAP", overlap_queue),
            ("ADJACENCY", adjacency_queue),
        ]:
            keys_to_remove: list[tuple[int, int]] = []

            for key, queue_info in queue.items():
                source_cam_idx, source_car_id = key
                source_car: Car = queue_info["car"]
                source_centroid = source_car.gps_bbox[-1].centroid

                cams_to_remove_from_targets: list[Camera] = []

                for target_cam in queue_info["targets"]:
                    global_id_source: int = t_manager.local_to_global[
                        (source_cam_idx, source_car_id)
                    ]

                    if (
                        target_cam.camera_idx
                        in t_manager.global_tracks[global_id_source]
                    ):
                        cams_to_remove_from_targets.append(target_cam)
                        continue

                    target_active_cars: list[Car] = active_cars_current_frame[
                        target_cam.camera_idx
                    ]
                    candidate_cars: list[tuple[float, Car]] = []

                    if qtype == "OVERLAP":
                        max_search_radius_meters = 15.0
                    else:
                        max_search_radius_meters = float("inf")

                    lon1, lat1 = source_centroid.x, source_centroid.y
                    export_qualitative = (qtype == "OVERLAP")
                    qual_data = {}
                    if export_qualitative:
                        qual_data[(source_cam_idx, source_car_id)] = {
                            "frame": frame_idx,
                            "source_car": {
                                "id": source_car_id,
                                "cam": source_cam_idx,
                                "lon": lon1, "lat": lat1,
                                "bbox": source_car.pixel_bbox[-1].tolist()
                            },
                            "before_filter_targets": [],
                            "after_filter_targets": []
                        }

                    for target_car in target_active_cars:
                        target_centroid = target_car.gps_bbox[-1].centroid
                        lon2, lat2 = target_centroid.x, target_centroid.y

                        avg_lat_rad = math.radians((lat1 + lat2) / 2.0)
                        dx = (lon2 - lon1) * math.cos(avg_lat_rad)
                        dy = lat2 - lat1

                        dist_meters = 111319.0 * math.sqrt(dx**2 + dy**2)

                        if export_qualitative:
                            qual_data[(source_cam_idx, source_car_id)]["before_filter_targets"].append({
                                "id": target_car.car_id,
                                "dist_meters": dist_meters,
                                "lon": lon2, "lat": lat2,
                                "bbox": target_car.pixel_bbox[-1].tolist()
                            })

                        if dist_meters <= max_search_radius_meters:
                            candidate_cars.append((dist_meters, target_car))

                            if export_qualitative:
                                qual_data[(source_cam_idx, source_car_id)]["after_filter_targets"].append({
                                    "id": target_car.car_id,
                                    "dist_meters": dist_meters,
                                    "lon": lon2, "lat": lat2,
                                    "bbox": target_car.pixel_bbox[-1].tolist()
                                })

                    candidate_cars.sort(key=lambda x: x[0])

                    # if len(candidate_cars) > 0:
                    #     tqdm.write(
                    #         f"[Frame {frame_idx}] Cam {source_cam_idx} Car {source_car_id} found {len(candidate_cars)} close candidates in Cam {target_cam.camera_idx}:"
                    #     )
                    #     for d, t_car in candidate_cars:
                    #         tqdm.write(
                    #             f"   -> Target Car {t_car.car_id} at distance {d:.2f} units"
                    #         )

                    matched = False

                    for dist, target_car in candidate_cars:
                        global_id_target: int = t_manager.local_to_global[
                            (target_cam.camera_idx, target_car.car_id)
                        ]

                        if global_id_source == global_id_target:
                            # Already matched
                            matched = True
                            break

                        if source_cam_idx in t_manager.global_tracks[global_id_target]:
                            # Target already matched
                            continue

                        if source_car == target_car:
                            tqdm.write(
                                f"[Frame {frame_idx}] NEW MATCH: Cam {source_cam_idx} Car {source_car_id} "
                                f"merged with Cam {target_cam.camera_idx} Car {target_car.car_id} "
                                f"(Dist: {dist:.1f}m)"
                            )
                            t_manager.link_cars(
                                source_cam_idx,
                                source_car_id,
                                target_cam.camera_idx,
                                target_car.car_id,
                            )

                            if export_qualitative:
                                output_qual_json = os.path.join(output_folder, "visuals", f"qualitative_ovelap_f{frame_idx}_c{source_car_id}.json")
                                output_qual_png = os.path.join(output_folder, "visuals", f"qualitative_ovelap_f{frame_idx}_c{source_car_id}.png")
                                with open(output_qual_json, "w") as f:
                                    json.dump(qual_data[(source_cam_idx, source_car_id)], f, indent=4)

                                visualize_spatial_filter(output_qual_json, dataset.data["videos"][target_cam.camera_idx], output_qual_png)

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
        pbar.update(1)

    # Save detections per camera
    t_manager.save(os.path.join(output_folder, "preds"), cam_names=dataset.get_cam_names())
