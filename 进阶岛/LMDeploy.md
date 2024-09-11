# LMDeploy 量化部署进阶实践
------------------------------------
------------------------------------

# 1 配置LMDeploy环境
## 创建虚拟环境
``` conda create -n lmdeploy  python=3.10 -y
conda activate lmdeploy
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install timm==1.0.8 openai==1.40.3 lmdeploy[all]==0.5.3
```
## InternStudio环境获取模型
```
mkdir /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat /root/models
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat /root/models
ln -s /root/share/new_models/OpenGVLab/InternVL2-26B /root/models
```
## LMDeploy验证启动模型文件
```
conda activate lmdeploy
lmdeploy chat /root/models/internlm2_5-7b-chat
```
![7c6bbaac7fb55a2490c32a9f719a051](https://github.com/user-attachments/assets/1bd16c6c-0227-4983-ac27-de2abc29478d)
# 2 LMDeploy与InternLM2.5
## 启动API服务器
```
conda activate lmdeploy
lmdeploy serve api_server \
    /root/models/internlm2_5-7b-chat \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
转发端口
```
 ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的ssh端口号
```
然后打开浏览器，访问http://127.0.0.1:23333看到如下界面即代表部署成功。
![c50e01bf2c73d3f21a4fb3f1fafb5b2](https://github.com/user-attachments/assets/7a1a49bd-ea38-4863-928c-1edd800a279d)
## 以命令行形式连接API服务器
```
conda activate lmdeploy
lmdeploy serve api_client http://localhost:23333
```
![54e17880c4b0ccb6b3d9ae54db84529](https://github.com/user-attachments/assets/42e708ca-113a-4c73-b4cd-6132e9f73b14)
![0bd9f5ba6f24a8d7551c9c307ea91f2](https://github.com/user-attachments/assets/1f3f98e4-67ff-40d1-831a-eb7fc4a1c866)
## 以Gradio网页形式连接API服务器
```
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```
关闭之前的cmd/powershell窗口，重开一个，再次做一下ssh转发(因为此时端口不同)
```ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的ssh端口号>```
![3245a17a99ed955a6db3ed5186853a8](https://github.com/user-attachments/assets/6ad326a0-2c72-402f-bc98-37782eed8f14)
## LMDeploy Lite
### 设置最大kv cache缓存大小
kv cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。在大规模训练和推理中，kv cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，kv cache全部存储于显存，以加快访存速度。

模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、kv cache占用的显存，以及中间运算结果占用的显存。LMDeploy的kv cache管理器可以通过设置--cache-max-entry-count参数，控制kv缓存占用剩余显存的最大比例。默认的比例为0.8。

首先我们先来回顾一下InternLM2.5正常运行时占用显存。
![19e4e28fb79112476d458d8b2ee74c9](https://github.com/user-attachments/assets/0b06d9ef-244c-40cd-9a12-f39b8512b45d)
占用了23GB，那么试一试执行以下命令，再来观看占用显存情况。
```lmdeploy chat /root/models/internlm2_5-7b-chat --cache-max-entry-count 0.4```
![e0f158083fe7f6594e9a0405e33d6d9](https://github.com/user-attachments/assets/35d87cde-0d5b-4449-b237-c3b932c17d73)
### 设置在线 kv cache int4/int8 量化
自 v0.4.0 起，LMDeploy 支持在线 kv cache int4/int8 量化，量化方式为 per-head per-token 的非对称量化。此外，通过 LMDeploy 应用 kv 量化非常简单，只需要设定 quant_policy 和cache-max-entry-count参数。目前，LMDeploy 规定 quant_policy=4 表示 kv int4 量化，quant_policy=8 表示 kv int8 量化。

我们通过2.1 LMDeploy API部署InternLM2.5的实践为例，输入以下指令，启动API服务器。
```
lmdeploy serve api_server \
    /root/models/internlm2_5-7b-chat \
    --model-format hf \
    --quant-policy 4 \
    --cache-max-entry-count 0.4\
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```

![image](https://github.com/user-attachments/assets/ea54ecf3-f67e-40aa-87ef-331edc6d5a41)
### W4A16 模型量化和部署
准确说，模型量化是一种优化技术，旨在减少机器学习模型的大小并提高其推理速度。量化通过将模型的权重和激活从高精度（如16位浮点数）转换为低精度（如8位整数、4位整数、甚至二值网络）来实现。

那么标题中的W4A16又是什么意思呢？

W4：这通常表示权重量化为4位整数（int4）。这意味着模型中的权重参数将从它们原始的浮点表示（例如FP32、BF16或FP16，Internlm2.5精度为BF16）转换为4位的整数表示。这样做可以显著减少模型的大小。
A16：这表示激活（或输入/输出）仍然保持在16位浮点数（例如FP16或BF16）。激活是在神经网络中传播的数据，通常在每层运算之后产生。
因此，W4A16的量化配置意味着：

权重被量化为4位整数。
激活保持为16位浮点数。
```
lmdeploy lite auto_awq \
   /root/models/internlm2_5-1_8b-chat \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --batch-size 1 \
  --search-scale False \
  --work-dir /root/models/internlm2_5-1_8b-chat-w4a16-4bit
```

![70e8b55786be7463031ce1242cb95c5](https://github.com/user-attachments/assets/835b5750-5985-4b47-8236-8255756e26fe)
我们可以输入如下指令查看在当前目录中显示所有子目录的大小。

```cd /root/models/
du -sh *```

![ee2dc1f71948d6e235566fa088a9656](https://github.com/user-attachments/assets/8460418d-8899-4a37-bac1-9f008262738d)
### W4A16 量化+ KV cache+KV cache 量化
输入以下指令，让我们同时启用量化后的模型、设定kv cache占用和kv cache int4量化
```
lmdeploy serve api_server \
    /root/models/internlm2_5-7b-chat-w4a16-4bit/ \
    --model-format awq \
    --quant-policy 4 \
    --cache-max-entry-count 0.4\
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
    ```
# 3LMDeploy与InternVL2
InternVL2-26B需要约70+GB显存，但是为了让我们能够在30%A100上运行，需要先进行量化操作，这也是量化本身的意义所在——即降低模型部署成本。
## W4A16 模型量化和部署
```
conda activate lmdeploy
lmdeploy lite auto_awq \
   /root/models/InternVL2-26B \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --batch-size 1 \
  --search-scale False \
  --work-dir /root/models/InternVL2-26B-w4a16-4bit
```

![5a0c3b380cf4316ade4b911c643f6cb](https://github.com/user-attachments/assets/ec42f74f-d17e-40d4-8b79-dd6b2109e4e8)
## W4A16 量化+ KV cache+KV cache 量化
```
lmdeploy serve api_server \
    /root/models/InternVL2-26B-w4a16-4bit \
    --model-format awq \
    --quant-policy 4 \
    --cache-max-entry-count 0.1\
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```

![image](https://github.com/user-attachments/assets/f364db45-d475-4790-a743-96a505fb1b9f)
此时只需要约23.8GB的显存，已经是一张30%A100即可部署的模型了。
## LMDeploy API部署InternVL2
```
lmdeploy serve api_server \
    /root/models/InternVL2-26B-w4a16-4bit/ \
    --model-format awq \
    --quant-policy 4 \
    --cache-max-entry-count 0.1 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
其余步骤与2.1.1 启动API服务器剩余内容一致。

![e65a1e1e0c27555ea3b03838b1cda92](https://github.com/user-attachments/assets/4ad531d0-7924-4b5e-b4ee-bf948a6ad556)
## LMDeploy之FastAPI与Function call
### API开发
```
conda activate lmdeploy
lmdeploy serve api_server \
    /root/models/internlm2_5-1_8b-chat-w4a16-4bit \
    --model-format awq \
    --cache-max-entry-count 0.4 \
    --quant-policy 4 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
创建python文件
```touch /root/internlm2_5.py```
```
# 导入openai模块中的OpenAI类，这个类用于与OpenAI API进行交互
from openai import OpenAI


# 创建一个OpenAI的客户端实例，需要传入API密钥和API的基础URL
client = OpenAI(
    api_key='YOUR_API_KEY',  
    # 替换为你的OpenAI API密钥，由于我们使用的本地API，无需密钥，任意填写即可
    base_url="http://0.0.0.0:23333/v1"  
    # 指定API的基础URL，这里使用了本地地址和端口
)

# 调用client.models.list()方法获取所有可用的模型，并选择第一个模型的ID
# models.list()返回一个模型列表，每个模型都有一个id属性
model_name = client.models.list().data[0].id

# 使用client.chat.completions.create()方法创建一个聊天补全请求
# 这个方法需要传入多个参数来指定请求的细节
response = client.chat.completions.create(
  model=model_name,  
  # 指定要使用的模型ID
  messages=[  
  # 定义消息列表，列表中的每个字典代表一个消息
    {"role": "system", "content": "你是一个友好的小助手，负责解决问题."},  
    # 系统消息，定义助手的行为
    {"role": "user", "content": "帮我讲述一个关于狐狸和西瓜的小故事"},  
    # 用户消息，询问时间管理的建议
  ],
    temperature=0.8,  
    # 控制生成文本的随机性，值越高生成的文本越随机
    top_p=0.8  
    # 控制生成文本的多样性，值越高生成的文本越多样
)

# 打印出API的响应结果
print(response.choices[0].message.content)
```
运行代码
```
conda activate lmdeploy
python /root/internlm2_5.py
```

![image](https://github.com/user-attachments/assets/486397d1-f5ae-477f-9a7d-dae7e165cdef)
### Function call
```
conda activate lmdeploy
lmdeploy serve api_server \
    /root/models/internlm2_5-7b-chat \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
新建internlm2_5_func.py。
```touch /root/internlm2_5_func.py```
```
from openai import OpenAI


def add(a: int, b: int):
    return a + b


def mul(a: int, b: int):
    return a * b


tools = [{
    'type': 'function',
    'function': {
        'name': 'add',
        'description': 'Compute the sum of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}, {
    'type': 'function',
    'function': {
        'name': 'mul',
        'description': 'Calculate the product of two numbers',
        'parameters': {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'int',
                    'description': 'A number',
                },
                'b': {
                    'type': 'int',
                    'description': 'A number',
                },
            },
            'required': ['a', 'b'],
        },
    }
}]
messages = [{'role': 'user', 'content': 'Compute (3+5)*2'}]

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func1_name = response.choices[0].message.tool_calls[0].function.name
func1_args = response.choices[0].message.tool_calls[0].function.arguments
func1_out = eval(f'{func1_name}(**{func1_args})')
print(func1_out)

messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content
})
messages.append({
    'role': 'environment',
    'content': f'3+5={func1_out}',
    'name': 'plugin'
})
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools)
print(response)
func2_name = response.choices[0].message.tool_calls[0].function.name
func2_args = response.choices[0].message.tool_calls[0].function.arguments
func2_out = eval(f'{func2_name}(**{func2_args})')
print(func2_out)
```

现在让我们输入以下指令运行python代码。
![image](https://github.com/user-attachments/assets/44a40dfa-06f0-4bda-88f2-f2eb0f3bf226)
