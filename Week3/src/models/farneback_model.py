import cv2
import numpy as np

from .of_model import BaseModel


class FarnebackModel(BaseModel):
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

    def _preprocess(self, image: list[np.ndarray]):
        outputs = []
        for im in image:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            outputs.append(im)
        return outputs

    def _postprocess(self, output: np.ndarray):
        return output
    
    def __call__(self, images: list[np.ndarray]):
        inputs = self._preprocess(images)

        outputs = cv2.calcOpticalFlowFarneback(inputs[0], inputs[1], *self.get_params_list())
        
        flow = self._postprocess(outputs)
        return flow
