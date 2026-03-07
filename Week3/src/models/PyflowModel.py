import numpy as np
import pyflow

from .BaseModel import BaseModel


class PyflowModel(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.params["pf_alpha"] = args.pf_alpha
        self.params["pf_ratio"] = args.pf_ratio
        self.params["pf_minWidth"] = args.pf_minWidth
        self.params["pf_nOuterFPIters"] = args.pf_nOuterFPIters
        self.params["pf_nInnerFPIters"] = args.pf_nInnerFPIters
        self.params["pf_nSORIters"] = args.pf_nSORIters
        self.params["pf_colType"] = args.pf_colType

    def _preprocess(self, image: list[np.ndarray]):
        output = []
        for im in image:
            im = np.astype(im, float) / 255.0
            if im.ndim == 2:
                im = im[..., np.newaxis]
            output.append(im)
        return output

    def _postprocess(self, output: tuple[np.ndarray]):
        u = output[0]
        v = output[1]
        flow = np.concatenate((u[..., np.newaxis], v[..., np.newaxis]), axis=2)
        return flow

    def __call__(self, images: list[np.ndarray]):
        inputs = self._preprocess(images)

        output = pyflow.coarse2fine_flow(inputs[0], inputs[1], *self.get_params_list())
        
        flow = self._postprocess(output)
        return flow
