import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from memfof import MEMFOF

from .of_model import BaseModel


class MEMFOFModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MEMFOF.from_pretrained(args.mf_path).eval().to(device=device)
        self.device = device

        self.params["any"] = None

    def _preprocess(self, images: list[np.ndarray]):
        outputs = []

        for im in images:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            im = F.to_tensor(im)
            outputs.append(im)

        outputs = (
            torch.stack((torch.zeros_like(outputs[0]), *outputs), dim=0)
            .unsqueeze(0)
            .to(device=self.device)
        )
        return outputs

    def _postprocess(self, outputs: dict):
        forward_flow: torch.Tensor = outputs["flow"][-1].unbind(dim=1)[1].squeeze()
        forward_flow = forward_flow.permute(1, 2, 0).detach().cpu().numpy()
        return forward_flow

    def __call__(self, images: list[np.ndarray]):
        inputs = self._preprocess(images)

        output = self.model(inputs)

        flow = self._postprocess(output)
        return flow
