import os

import cv2
import numpy as np
from tqdm import tqdm


def generate_video(images: list | np.ndarray, output_path: str, fps: int = 24):
    print("Generating Video...")

    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)

    width = images[0].shape[1]
    height = images[0].shape[0]

    cap_write = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in tqdm(images):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cap_write.write(bgr_image)

    cap_write.release()

    print(f"Video saved in {output_path}")


def save_flow(flow: np.ndarray, output_path: str):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = 255
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, bgr)


def generate_flow_reference(size: tuple):
    # Initial variable definitions
    width = size[1]
    half_width = width / 2
    height = size[0]
    half_height = height / 2

    # Get coords and build reference flow matrix
    ys, xs = np.ogrid[0:height, 0:width]
    xs = (xs - half_width) / width
    ys = (ys - half_height) / height
    xs = xs.repeat(ys.shape[0], axis=0)
    ys = ys.repeat(xs.shape[1], axis=1)
    flow = np.stack((xs, ys), axis=2)

    # Save reference as image
    return flow
