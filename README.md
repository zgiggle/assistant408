# assistant408
基于Internlm2-chat-7b模型的408考研小助手
## 📖 简介
assistant408（考研408计算机学科专业基础综合的小助手）是一个集成 408 考研知识及解答能力的大语言模型。
> 408全称计算机专业基础综合，共包括四门专业课的考察，分别是数据结构、计算机组成原理、操作系统和计算机网络。
> |  考试科目  |   数据结构   |   计算机组成原理   | 操作系统 | 计算机网络 |
>|:------:|:------:|:-------:|:-------:|:-------:|
> |分数|45分|45分|35分|25分|
> |考试占比|30%|30%|23%|17%|

该模型基于Internlm2-chat-7b基座模型，使用模型容易出错的概念性题目进行微调，再使用GPT生成的400道题目进行评测，得到了较好的效果。

## 🚀 快速开始
### 📝 知识库
在本次知识库搭建中用到了如下知识，有需要的可以自行前往下载，在本仓库 data_base/vector_db/chroma 目录下已经为大家搭建好了训练后的知识库，可以将本仓库 clone 到本地。
- [x] 25王道考研数据结构
- [x] 25王道考研计算机组成原理
- [x] 25王道考研操作系统
- [x] 25王道考研计算机网络
- [x] ...
```bash
git clone https://github.com/zgiggle/assistant408.git
```

如果需要自定义知识库，可以参考create_db.py，首先将本地知识库通过Unstructed Loader转化为纯文本格式，纯文本再经过Text Splitter分为Chunks，通过Sentence Transformer转化为向量格式并存放在Chroma向量数据库中。对于用户的输入同样通过Sentence Transformer转为向量格式，再与向量数据库进行相似度匹配，找到与用户输入相关的文本段，最后将文本段嵌入到写好的Prompt Template中并交给InternLM回答。这整个过程都会被封装在检索问答流中，我们可以将个性化配置引入检索问答流中，就能搭建属于自己的RAG应用。

目录下的LLM.py文件用于自定义LLM类，在终端使用如下命令即可运行食用（注意修改run_internlm.py中的模型地址）。
```bash
python run_internlm.py
```

### 📝 微调
__1. 准备数据集__  
在本仓库的 data/xtuner_data 目录下已经准备好了训练的数据集，微调数据集结构如下。
```text
input: 计算机中，浮点数的指数部分通常采用什么方式进行编码？
output: 移码。
input: 哪种类型的存储器通常被用作计算机的主存储器？
output: SRAM。
input: 在计算机系统中，一个32位的指令长度意味着什么？
output: 指令的长度是32位。
input: 什么是衡量CPU性能的一个重要指标？
output: 时钟频率、核心数、缓存大小。
input: 请做一下自我介绍。
output: 你好,我是考研408计算机学科专业基础综合的小助手哦。
```

__2. 准备配置文件__
```bash
mkdir /root/personal_assistant/config/question && cd /root/personal_assistant/config/question
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
```

__3. 修改配置文件__
```bash
# 修改模型为本地路径
pretrained_model_name_or_path = '/root/share/module_repos/internlm2-chat-7b'

# 修改训练数据集为本地路径
data_path = '/root/personal_assistant/data/xtuner_data/answer2.json'

# 修改配置文件PART 1参数
max_length = 2048
max_epochs = 12
evaluation_freq = 90
SYSTEM = '您是考研408计算机学科专业基础综合小助手，您始终根据提问者的问题提供准确、全面和详细的答案。'
evaluation_inputs = [
    '请做一下自我介绍',
    '计算机中，浮点数的指数部分通常采用什么方式进行编码', 
    '在计算机系统中，中断的作用是什么？',
    '在操作系统中，死锁发生的必要条件之一是：',
    '哪种设备不适合使用缓冲区（Buffer）技术？',
    '在计算机系统中，什么决定了计算机执行指令的速度？'
]

# 修改配置文件PART 3参数
dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
dataset_map_fn=None

```

__4. 开始训练__
```bash
cd /root/personal_assistant/config/question
xtuner train internlm2_chat_7b_qlora_oasst1_e3_copy.py
```
__5. pth格式转换为hugging face格式__
```bash
mkdir /root/personal_assistant/config/question/work_dirs/hf
export MKL_SERVICE_FORCE_INTEL=1

# 配置文件存放的位置
export CONFIG_NAME_OR_PATH=/root/personal_assistant/config/question/internlm2_chat_7b_qlora_oasst1_e3_copy.py

# 模型训练后得到的pth格式参数存放的位置
export PTH=/root/personal_assistant/config/question/work_dirs/internlm2_chat_7b_qlora_oasst1_e3_copy/iter_384.pth

# pth文件转换为Hugging Face格式后参数存放的位置
export SAVE_PATH=/root/personal_assistant/config/question/work_dirs/hf

# 执行参数转换
xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH
```
__6. 模型合并__
```bash
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER='GNU'

# 原始模型参数存放的位置
export NAME_OR_PATH_TO_LLM=/root/share/model_repos/internlm2-chat-7b

# Hugging Face格式参数存放的位置
export NAME_OR_PATH_TO_ADAPTER=/root/personal_assistant/config/question/work_dirs/hf

# 最终Merge后的参数存放的位置
mkdir /root/personal_assistant/config/question/work_dirs/hf_merge
export SAVE_PATH=/root/personal_assistant/config/question/work_dirs/hf_merge

# 执行参数Merge
xtuner convert merge \
    $NAME_OR_PATH_TO_LLM \
    $NAME_OR_PATH_TO_ADAPTER \
    $SAVE_PATH \
--max-shard-size 2GB
```


## ‍‍‍‍‍🙂 项目成员
- 张丰瑞、杨阳、周殷稷、曹一凡

## 💕 致谢
<div align="center">

**感谢上海人工智能实验室组织的 书生·浦语实战营 学习活动 与 算力支持~**

</div>











