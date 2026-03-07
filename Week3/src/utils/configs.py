import itertools
import math
from collections import OrderedDict
from pprint import pprint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


class BaseConfig:
    def __init__(self):
        self.params = OrderedDict()

    def get_params_dict(self):
        return self.params

    def get_params_list(self):
        return list(self.params.values())

    def preprocess(self):
        pass

    def postprocess(self):
        pass


class PyflowConfig(BaseConfig):
    def __init__(self, args):
        super().__init__()
        self.params["pf_alpha"] = args.pf_alpha
        self.params["pf_ratio"] = args.pf_ratio
        self.params["pf_minWidth"] = args.pf_minWidth
        self.params["pf_nOuterFPIters"] = args.pf_nOuterFPIters
        self.params["pf_nInnerFPIters"] = args.pf_nInnerFPIters
        self.params["pf_nSORIters"] = args.pf_nSORIters
        self.params["pf_colType"] = args.pf_colType

    def preprocess(self, image: list[np.ndarray]):
        output = []
        for im in image:
            im = np.astype(im, float) / 255.0
            if im.ndim == 2:
                im = im[..., np.newaxis]
            output.append(im)
        return output

    def postprocess(self, output: tuple[np.ndarray]):
        u = output[0]
        v = output[1]
        flow = np.concatenate((u[..., np.newaxis], v[..., np.newaxis]), axis=2)
        return flow


class FarnebackConfig(BaseConfig):
    def __init__(self, args):
        super().__init__()
        self.params["fb_flow"] = None  # Don't worry, ignore this
        self.params["fb_pyrScale"] = args.fb_pyrScale
        self.params["fb_levels"] = args.fb_levels
        self.params["fb_winSize"] = args.fb_winSize
        self.params["fb_iters"] = args.fb_iters
        self.params["fb_polyN"] = args.fb_polyN
        self.params["fb_polySigma"] = args.fb_polySigma
        self.params["fb_flags"] = 0

    def preprocess(self, image: list[np.ndarray]):
        outputs = []
        for im in image:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            outputs.append(im)
        return outputs

    def postprocess(self, output: np.ndarray):
        return output


class PerceiverConfig(BaseConfig):
    def __init__(self, model, args):
        super().__init__()
        self.params["any"] = None
        self.model = model
        self.device = model.device
        self.train_size = model.config.train_size

    def _normalize(self, image: np.ndarray):
        return image.astype(np.float32) / 255.0 * 2 - 1

    def _extract_image_patches(
        self, x: torch.Tensor, kernel: int, stride=1, dilation=1
    ):
        # Do TF 'SAME' Padding
        b, c, h, w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(
            x,
            (
                pad_row // 2,
                pad_row - pad_row // 2,
                pad_col // 2,
                pad_col - pad_col // 2,
            ),
        )

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

        return patches.view(b, -1, patches.shape[-2], patches.shape[-1])

    def _compute_grid_indices(self, image_shape, min_overlap=20):
        if min_overlap >= self.train_size[0] or min_overlap >= self.train_size[1]:
            raise ValueError(
                f"Overlap should be less than size of patch (got {min_overlap}"
                f"for patch size {self.train_size})."
            )
        ys = list(range(0, image_shape[0], self.train_size[0] - min_overlap))
        xs = list(range(0, image_shape[1], self.train_size[1] - min_overlap))
        # Make sure the final patch is flush with the image boundary
        ys[-1] = image_shape[0] - self.train_size[0]
        xs[-1] = image_shape[1] - self.train_size[1]
        return itertools.product(ys, xs)

    def _compute_optical_flow(
        self, model, img1, img2, grid_indices, FLOW_SCALE_FACTOR=20
    ):
        """Function to compute optical flow between two images.

        To compute the flow between images of arbitrary sizes, we divide the image
        into patches, compute the flow for each patch, and stitch the flows together.

        Args:
            model: PyTorch Perceiver model
            img1: first image
            img2: second image
            grid_indices: indices of the upper left corner for each patch.
        """
        img1 = torch.tensor(np.moveaxis(img1, -1, 0))
        img2 = torch.tensor(np.moveaxis(img2, -1, 0))
        imgs = torch.stack([img1, img2], dim=0)[None]
        height = imgs.shape[-2]
        width = imgs.shape[-1]

        # print("Shape of imgs after stacking:", imgs.shape)

        patch_size = model.config.train_size

        if height < patch_size[0]:
            raise ValueError(
                f"Height of image (shape: {imgs.shape}) must be at least {patch_size[0]}."
                "Please pad or resize your image to the minimum dimension."
            )
        if width < patch_size[1]:
            raise ValueError(
                f"Width of image (shape: {imgs.shape}) must be at least {patch_size[1]}."
                "Please pad or resize your image to the minimum dimension."
            )

        flows = 0
        flow_count = 0

        for y, x in grid_indices:
            imgs = torch.stack([img1, img2], dim=0)[None]
            inp_piece = imgs[..., y : y + patch_size[0], x : x + patch_size[1]]

            print("Shape of inp_piece:", inp_piece.shape)

            batch_size, _, C, H, W = inp_piece.shape
            patches = self._extract_image_patches(
                inp_piece.view(batch_size * 2, C, H, W), kernel=3
            )
            _, C, H, W = patches.shape
            patches = patches.view(batch_size, -1, C, H, W).float().to(model.device)

            # actual forward pass
            with torch.no_grad():
                output = model(inputs=patches).logits * FLOW_SCALE_FACTOR

            # the code below could also be implemented in PyTorch
            flow_piece = output.cpu().detach().numpy()

            weights_x, weights_y = np.meshgrid(
                torch.arange(patch_size[1]), torch.arange(patch_size[0])
            )

            weights_x = np.minimum(weights_x + 1, patch_size[1] - weights_x)
            weights_y = np.minimum(weights_y + 1, patch_size[0] - weights_y)
            weights = np.minimum(weights_x, weights_y)[np.newaxis, :, :, np.newaxis]
            padding = [
                (0, 0),
                (y, height - y - patch_size[0]),
                (x, width - x - patch_size[1]),
                (0, 0),
            ]
            flows += np.pad(flow_piece * weights, padding)
            flow_count += np.pad(weights, padding)

            # delete activations to avoid OOM
            del output

        flows /= flow_count
        return flows

    def preprocess(self, images: list[np.ndarray]):
        outputs = []

        # Stack images
        for im in images:
            if im.ndim == 2:
                im = im[..., np.newaxis]
            im = im.repeat(3, axis=-1)
            outputs.append(im)

        return outputs

    def postprocess(self, output: np.ndarray):
        output = np.array(output).squeeze()
        return output

    def __call__(self, image1: np.ndarray, image2: np.ndarray, nothing):
        grid_indices = self._compute_grid_indices(image1.shape)
        flow = self._compute_optical_flow(
            self.model,
            self._normalize(image1),
            self._normalize(image2),
            grid_indices,
        )
        return flow
