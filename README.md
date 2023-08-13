## simple-tagger简单的打标工具，可以用于实现各类打标并合并成一句话。
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [所需环境 Environment](#所需环境)
3. [预测步骤 How2predict](#预测步骤)
4. [输入输出格式 Format](#输入输出格式)
5. [参考资料 Reference](#Reference)

## Top News
**`2023-08`**:**仓库创建，更新wd14打标。**  

## 所需环境
按需要的requirements.txt配置即可   
```
pip install -r requirements.txt
```

(Optional) 为了方便调用，可使用   
```
pip install simple-tagger
```
或者   
```
git clone https://github.com/bubbliiiing/simple-tagger.git
cd simple-tagger
python setup.py install
```
快速安装。  

然后，我们就可以在别的project里面使用simple-tagger。  

## 预测步骤
### a、Demo
1. 下载完库后解压，运行predict.py即可。   
```python
python predict.py
```  
### b、在别的项目中使用simple-tagger**。  
1. 首先导入TaggerPredictor。  
```python
from simple_tagger import TaggerPredictor
```
2. 根据模型名称创建模型。   
```
# "SmilingWolf/wd-v1-4-convnext-tagger-v2"
# "SmilingWolf/wd-v1-4-convnextv2-tagger-v2.onnx"
# "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
# "SmilingWolf/wd-v1-4-vit-tagger-v2"
model_name  = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
model       = TaggerPredictor(model_name)
```
3. 传入图片并且进行预测
```
model(raw_image)
```

## 输入输出格式
### a、inputs
当前必要输入为image。   
无论是numpy格式还是PIL均可，使用RGB色域。

### b、outputs
当前输出均为字典，通过不同的key调用不同的输出。
#### 1.wd14
sentence代表打标结果汇总的一句话；  
probs代表相对于总标签集合，每个标签的得分；  
tags_list代表打标结果对应的list；
```
{
    "sentence": "general, sensitive, 1girl, long hair, smile, shirt, sitting, outdoors, barefoot, water, plaid, ocean, animal, beach, dog, leash, sand, plaid shirt",
    "tags_list": ['general', 'sensitive', '1girl', 'long hair', 'smile', 'shirt', 'sitting', 'outdoors', 'barefoot', 'water', 'plaid', 'ocean', 'animal', 'beach', 'dog', 'leash', 'sand', 'plaid shirt'], 
    "probs": [3.5740888e-01, 6.6963732e-01, 4.4209361e-03, ... 4.7683716e-06, 7.1525574e-07, 4.7683716e-07],
}
```

## Reference
https://github.com/SmilingWolf/SW-CV-ModelZoo   
