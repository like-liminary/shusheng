# InternVL 多模态模型部署微调实践
---
# 1 写在前面（什么是InternVL）
InternVL 是一种用于多模态任务的深度学习模型，旨在处理和理解多种类型的数据输入，如图像和文本。它结合了视觉和语言模型，能够执行复杂的跨模态任务，比如图文匹配、图像描述生成等。通过整合视觉特征和语言信息，InternVL 可以在多模态领域取得更好的表现
![image](https://github.com/user-attachments/assets/8b935ddc-e72e-46fc-9eb3-0ac22eb8ed22)
# 2 InternVL 部署微调实践
让我们来一起完成一个用VLM模型进行冷笑话生成，让你的模型说出很逗的冷笑话吧。在这里，我们微调InterenVL使用xtuner。部署InternVL使用lmdeploy。
## 准备InternVL模型
```cd /root
mkdir -p model

# cp 模型

cp -r /root/share/new_models/OpenGVLab/InternVL2-2B /root/model/
```
## 准备环境
配置虚拟环境

```conda create --name xtuner python=3.10 -y

# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner

# 安装一些必要的库
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
apt install libaio-dev
pip install transformers==4.39.3
pip install streamlit==1.36.0
```
安装xtuner
```
# 创建一个目录，用来存放源代码
mkdir -p /root/InternLM/code

cd /root/InternLM/code

git clone -b v0.1.23  https://github.com/InternLM/XTuner
```
```
cd /root/InternLM/code/XTuner
pip install -e '.[deepspeed]'
pip install lmdeploy==0.5.3
xtuner version

##命令

xtuner help
```
![S)KG{HEY5$%5GGRYIB2RTUT](https://github.com/user-attachments/assets/4ff41597-a445-4c82-93cc-46fa7d2303c4)
## 准备微调数据集
```
## 首先让我们安装一下需要的包
pip install datasets matplotlib Pillow timm

## 让我们把数据集挪出来
cp -r /root/share/new_models/datasets/CLoT_cn_2000 /root/InternLM/datasets/
```
![4%32)CF`C9H_${N2L))WCNC](https://github.com/user-attachments/assets/21859597-b9c2-47d0-b1db-c90b93a7674e)
## InternVL 推理部署攻略
```
touch /root/InternLM/code/test_lmdeploy.py
cd /root/InternLM/code/
```
粘贴代码
```
from lmdeploy import pipeline
from lmdeploy.vl import load_image

pipe = pipeline('/root/model/InternVL2-2B')

image = load_image('/root/InternLM/007aPnLRgy1hb39z0im50j30ci0el0wm.jpg')
response = pipe(('请你根据这张图片，讲一个脑洞大开的梗', image))
print(response.text)
```
运行：
![4 }M DV21EN2Y8}}URLLC5D](https://github.com/user-attachments/assets/1e2bdb11-e40a-4e82-9e1b-7dbf57c1e73e)
## InternVL 微调攻略
数据集格式：
```


# 为了高效训练，请确保数据格式为：
{
    "id": "000000033471",
    "image": ["coco/train2017/000000033471.jpg"], # 如果是纯文本，则该字段为 None 或者不存在
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      }
    ]
  }
```
### 微调：：：：
![image](https://github.com/user-attachments/assets/68b597f9-f930-4a1e-9498-5bae1a2c2a11)
### 合并权重&&模型转换
```
cd XTuner
# transfer weights
python3 xtuner/configs/internvl/v1_5/convert_to_official.py xtuner/configs/internvl/v2/internvl_v2_internlm2_2b_qlora_finetune.py /root/InternLM/work_dir/internvl_ft_run_8_filter/iter_3000.pth /root/InternLM/InternVL2-2B/
```
### 结果测试：
![image](https://github.com/user-attachments/assets/146f5777-a727-41d2-80f4-033d389707d0)
