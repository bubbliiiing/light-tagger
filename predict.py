import requests
from PIL import Image

from simple_tagger import TaggerPredictor

# "SmilingWolf/wd-v1-4-convnext-tagger-v2"
# "SmilingWolf/wd-v1-4-convnextv2-tagger-v2.onnx"
# "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
# "SmilingWolf/wd-v1-4-vit-tagger-v2"
model_name  = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
# 图片地址
img_url     = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 

raw_image   = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
model       = TaggerPredictor(model_name, cache_dir="model_data")
print(model(raw_image)['probs'])
print(model(raw_image)['tags_list'])