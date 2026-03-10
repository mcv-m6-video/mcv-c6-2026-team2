import itertools
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_F
from transformers import PerceiverForOpticalFlow

from .of_model import BaseModel


# class PerceiverModel(BaseModel):
#     def __init__(self, args):
#         super().__init__()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = PerceiverForOpticalFlow.from_pretrained(args.perc_path).eval().to(
#             device=device
#         )
#         self.train_size = tuple(self.model.config.train_size)  # (H, W)
#         self.min_overlap = 20

#         self.params["any"] = None
class PerceiverModel(BaseModel):
    def __init__(self, args):
        super().__init__()

        print("[Perceiver] deciding device", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Perceiver] device = {device}", flush=True)

        print(f"[Perceiver] loading model from: {args.perc_path}", flush=True)
        model = PerceiverForOpticalFlow.from_pretrained(args.perc_path)
        print("[Perceiver] model loaded from pretrained", flush=True)

        print("[Perceiver] setting eval", flush=True)
        model = model.eval()
        print("[Perceiver] eval set", flush=True)

        print("[Perceiver] moving model to device", flush=True)
        model = model.to(device=device)
        print("[Perceiver] model moved to device", flush=True)

        self.model = model
        self.train_size = self.model.config.train_size
        self.min_overlap = 20
        self.params["any"] = None

        print("[Perceiver] finished initialization", flush=True)

    def _normalize(self, image: np.ndarray):
        return image.astype(np.float32) / 255.0 * 2 - 1

    def _extract_image_patches(
        self, x: torch.Tensor, kernel: int, stride: int = 1, dilation: int = 1
    ):
        """
        Extract local patches with TF-like SAME padding.
        Input shape: [B, C, H, W]
        Output shape: [B, C * kernel * kernel, H, W] for stride=1
        """
        b, c, h, w = x.shape

        out_h = math.ceil(h / stride)
        out_w = math.ceil(w / stride)

        pad_h = max((out_h - 1) * stride + (kernel - 1) * dilation + 1 - h, 0)
        pad_w = max((out_w - 1) * stride + (kernel - 1) * dilation + 1 - w, 0)

        # torch.nn.functional.pad uses:
        # (pad_left, pad_right, pad_top, pad_bottom)
        x = torch_F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        # [B, C, out_h, out_w, k, k] -> [B, C, k, k, out_h, out_w]
        patches = patches.permute(0, 1, 4, 5, 2, 3).contiguous()

        return patches.view(b, -1, patches.shape[-2], patches.shape[-1])

    def _compute_padded_size(self, h: int, w: int):
        """
        Pad to a size compatible with the sliding window grid.
        """
        patch_h, patch_w = self.train_size
        stride_h = patch_h - self.min_overlap
        stride_w = patch_w - self.min_overlap

        padded_h = max(h, patch_h)
        padded_w = max(w, patch_w)

        if padded_h > patch_h:
            padded_h = patch_h + \
                math.ceil((padded_h - patch_h) / stride_h) * stride_h
        if padded_w > patch_w:
            padded_w = patch_w + \
                math.ceil((padded_w - patch_w) / stride_w) * stride_w

        return padded_h, padded_w

    def _pad_image(self, image: np.ndarray):
        """
        Pad image on bottom/right so patch extraction is always valid.
        """
        h, w = image.shape[:2]
        padded_h, padded_w = self._compute_padded_size(h, w)

        pad_bottom = padded_h - h
        pad_right = padded_w - w

        if pad_bottom == 0 and pad_right == 0:
            return image, (h, w)

        image = cv2.copyMakeBorder(
            image,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

        return image, (h, w)

    def _compute_grid_indices(self, image_shape):
        """
        Generate valid top-left coordinates for patches.
        """
        patch_h, patch_w = self.train_size
        h, w = image_shape[:2]

        if h < patch_h or w < patch_w:
            raise ValueError(
                f"Image shape {(h, w)} is smaller than patch size {self.train_size}."
            )

        stride_h = patch_h - self.min_overlap
        stride_w = patch_w - self.min_overlap

        ys = list(range(0, h - patch_h + 1, stride_h))
        xs = list(range(0, w - patch_w + 1, stride_w))

        return itertools.product(ys, xs)

    def _compute_optical_flow(
        self, model, img1, img2, grid_indices, FLOW_SCALE_FACTOR=20
    ):
        """
        Compute optical flow by processing overlapping patches and stitching them.
        """
        img1 = torch.tensor(np.moveaxis(img1, -1, 0))
        img2 = torch.tensor(np.moveaxis(img2, -1, 0))
        imgs = torch.stack([img1, img2], dim=0)[None]

        height = imgs.shape[-2]
        width = imgs.shape[-1]
        patch_h, patch_w = model.config.train_size

        if height < patch_h:
            raise ValueError(
                f"Height of image (shape: {imgs.shape}) must be at least {patch_h}."
            )
        if width < patch_w:
            raise ValueError(
                f"Width of image (shape: {imgs.shape}) must be at least {patch_w}."
            )

        flows = np.zeros((1, height, width, 2), dtype=np.float32)
        flow_count = np.zeros((1, height, width, 1), dtype=np.float32)

        for y, x in grid_indices:
            inp_piece = imgs[..., y: y + patch_h, x: x + patch_w]

            # Safety check
            assert inp_piece.shape[-2:] == (patch_h, patch_w), inp_piece.shape

            batch_size, _, c, h, w = inp_piece.shape

            patches = self._extract_image_patches(
                inp_piece.view(batch_size * 2, c, h, w),
                kernel=3,
            )
            _, c, h, w = patches.shape
            patches = patches.view(batch_size, -1, c, h,
                                   w).float().to(model.device)

            with torch.no_grad():
                output = model(inputs=patches).logits * FLOW_SCALE_FACTOR

            flow_piece = output.cpu().numpy()

            weights_x, weights_y = np.meshgrid(
                np.arange(patch_w), np.arange(patch_h)
            )
            weights_x = np.minimum(weights_x + 1, patch_w - weights_x)
            weights_y = np.minimum(weights_y + 1, patch_h - weights_y)
            weights = np.minimum(weights_x, weights_y)[
                np.newaxis, :, :, np.newaxis]

            padding = [
                (0, 0),
                (y, height - y - patch_h),
                (x, width - x - patch_w),
                (0, 0),
            ]

            flows += np.pad(flow_piece * weights, padding)
            flow_count += np.pad(weights, padding)

            del output

        flows /= np.maximum(flow_count, 1e-6)
        return flows

    def _preprocess(self, images: list[np.ndarray]):
        outputs = []

        for im in images:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            elif im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            outputs.append(im)

        # Pad both images to the same padded size
        h0, w0 = outputs[0].shape[:2]
        h1, w1 = outputs[1].shape[:2]
        if (h0, w0) != (h1, w1):
            raise ValueError(
                f"Input images must have same shape, got {(h0, w0)} and {(h1, w1)}"
            )

        padded_h, padded_w = self._compute_padded_size(h0, w0)

        padded_outputs = []
        for im in outputs:
            pad_bottom = padded_h - im.shape[0]
            pad_right = padded_w - im.shape[1]
            im = cv2.copyMakeBorder(
                im,
                top=0,
                bottom=pad_bottom,
                left=0,
                right=pad_right,
                borderType=cv2.BORDER_REFLECT_101,
            )
            padded_outputs.append(im)

        return padded_outputs, (h0, w0)

    def _postprocess(self, output: np.ndarray, original_shape: tuple[int, int]):
        output = np.array(output).squeeze()
        h, w = original_shape
        return output[:h, :w]

    def __call__(self, images: list[np.ndarray]):
        inputs, original_shape = self._preprocess(images)

        grid_indices = self._compute_grid_indices(inputs[0].shape)
        output = self._compute_optical_flow(
            self.model,
            self._normalize(inputs[0]),
            self._normalize(inputs[1]),
            grid_indices,
        )

        flow = self._postprocess(output, original_shape)
        return flow
