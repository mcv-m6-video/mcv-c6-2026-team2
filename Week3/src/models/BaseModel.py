from collections import OrderedDict

import numpy as np


class BaseModel:
    def __init__(self):
        self.params = OrderedDict()

    def get_params_dict(self):
        return self.params

    def get_params_list(self):
        return list(self.params.values())

    def _preprocess(self, images: list[np.ndarray]):
        pass

    def _postprocess(self, images: list[np.ndarray]):
        pass
