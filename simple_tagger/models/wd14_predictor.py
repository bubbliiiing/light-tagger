import onnxruntime as rt
import os
import pandas as pd
import cv2
from .utils import *

wd_model_name_2_url = {
    "SmilingWolf/wd-v1-4-convnext-tagger-v2"        : "https://huggingface.co/bubbliiiing/simple-tagger/resolve/main/SmilingWolf-wd-v1-4-convnext-tagger-v2.onnx",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2.onnx" : "https://huggingface.co/bubbliiiing/simple-tagger/resolve/main/SmilingWolf-wd-v1-4-convnextv2-tagger-v2.onnx",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2"          : "https://huggingface.co/bubbliiiing/simple-tagger/resolve/main/SmilingWolf-wd-v1-4-swinv2-tagger-v2.onnx",
    "SmilingWolf/wd-v1-4-vit-tagger-v2"             : "https://huggingface.co/bubbliiiing/simple-tagger/resolve/main/SmilingWolf-wd-v1-4-vit-tagger-v2.onnx",
}

csv_url = "https://huggingface.co/bubbliiiing/simple-tagger/resolve/main/selected_tags.csv"

class WD14_Predictor(object):
    def __init__(self, model_name:str, cache_dir:str=None) -> None:
        model_dir = get_tagger_home(cache_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        url             = wd_model_name_2_url[model_name]
        cached_file     = check_and_download_url(model_dir, url)
        csv_url_file    = check_and_download_url(model_dir, csv_url)

        self.model          = rt.InferenceSession(cached_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.label_names    = pd.read_csv(csv_url_file)

    def __call__(self, image, dim:int=448, threshold:float=0.3228, **kwargs) -> str:
        image_np = np.array(image, np.uint8)

        img = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
        img = smart_24bit(img)
        img = make_square(img, dim)
        img = smart_resize(img, dim)
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: img})[0]

        self.label_names["probs"] = probs[0]
        tags_list = self.label_names[self.label_names["probs"] > threshold][["tag_id", "name", "probs"]]
        tags_list = list(tags_list["name"])
        tags_list = [tag.replace("_", " ") for tag in tags_list]
        sentence  = ", ".join(tags_list)
        
        return {
            "sentence"  : sentence,
            "tags_list" : tags_list,
            "probs"     : probs[0],
        }
