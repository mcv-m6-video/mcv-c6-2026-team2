import numpy as np


def kitti_of_gt_processing(gt: np.ndarray):
    """
    Processing of groundtruth according to the instructions
    located in the readme.txt inside the devkit.
    """
    if gt.dtype == np.uint16:
        gt = gt.astype(float)

    # flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0; flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0; valid(u,v) = (bool)I(u,v,3);
    u = (gt[..., 2] - 2**15) / 64.0
    v = (gt[..., 1] - 2**15) / 64.0

    valid_mask = gt[..., 0] > 0

    u[~valid_mask] = 0
    v[~valid_mask] = 0

    gt_flow = np.stack((u, v), axis=2)

    return gt_flow, valid_mask
