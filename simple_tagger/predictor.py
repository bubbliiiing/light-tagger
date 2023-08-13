from typing import Any

import numpy as np
from PIL import Image

from .models.wd14_predictor import WD14_Predictor, wd_model_name_2_url


class TaggerPredictor(object):
    def __init__(self, model_name:str, cache_dir:str=None) -> None:
        if model_name in wd_model_name_2_url.keys():
            self.predictor = WD14_Predictor(model_name, cache_dir)
        else:
            raise ValueError("The model_name must be in " + str(wd_model_name_2_url.keys()) + ".")
    
    def __call__(
            self, 
            image,
            **kwargs,
        ) -> str:
        image_pil   = Image.fromarray(np.array(image, np.uint8)).convert("RGB")
        output      = self.predictor(image_pil)
        return output