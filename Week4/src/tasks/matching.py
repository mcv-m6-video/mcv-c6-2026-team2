import numpy as np
from src.utils.camera import Camera
from src.utils.car import Car
from src.utils.dataset import MOMCDataset


def main(args):
    # Extract arguments
    dataset_root = args.dataset_root
    seq = args.seq

    match_checkpoint = args.match_checkpoint

    # Create dataset
    dataset = MOMCDataset(dataset_root, seq)

    cars_dict: dict[int, Car] = {}
    cam_list = [
        Camera(idx, resolution, homography, offset, num_frames)
        for idx, (homography, offset, num_frames, resolution) in enumerate(
            dataset.get_all_cameras()
        )
    ]

    frame_idx = 0
    loop = True
    while loop:
        # Letsgooooo, bueno, por hoy ta bien
        for cam_idx in range(len(cam_list)):
            frame, dets = dataset[cam_idx, frame_idx]

            if frame is None or dets is None:
                loop = False
                continue

            loop = True

            for d in dets:
                f_idx, car_id, xleft, ytop, xright, ybottom, _, _, _, _ = d.split()

                car_image = frame[ytop:ybottom, xleft:xright]
                bbox = np.array([xleft, ytop, xright, ybottom], dtype=np.float32)

                car = Car(car_id)

                car.add_detection(
                    car_image, bbox, cam_list[cam_idx].homography, f_idx, cam_idx
                )

                if car_id in cars_dict:
                    cars_dict[car_id].merge_cars(car)
