from collections import OrderedDict
from pprint import pprint

import cv2
import numpy as np


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
        output = []
        for im in image:
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            output.append(im)
        return output

    def postprocess(self, output: np.ndarray):
        return output

