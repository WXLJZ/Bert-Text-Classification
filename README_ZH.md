# 🤖 BERT 文本分类

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFCE56?logo=huggingface&logoColor=black)


✨ **BERT** - 基于Transformers和pytorch构建的bert文本分类模型。注意，本项目基于GitHub开源项目[**BERT-TextClassification**](https://github.com/songyingxin/Bert-TextClassification)修改完成。

---

## 🚀 亮点
- ⚡ **七类BERT基线**：共包含七条BERT文本分类基线，包括原始BERT、BERTATT、BERTLSTM、BERTCNN、BERTCNN+、BERTRCNN、BERTDPCNN
- 🌐 **第三方包支持**：当前许多BERT文本分类基线可能都是基于比较老的pytorch-pretrained-bert包，本项目基于transformers包以及pytorch构建，环境的搭建更为方便
- 🎯 **模型训练**：支持模型训练、验证、测试、预测等功能

## 📦 功能列表
| 模型            | 模型描述                               |
|---------------|------------------------------------|
| `BertOrigin`  | 原始BERT模型                           |
| `BertATT`     | 使用注意力机制动态选择重要Token特征               |
| `BertCNN`     | 使用CNN进行局部特征提取                      |
| `BertCNNPlus` | 在BertCNN的基础上，将CNN提取的局部特征与全局句级表示拼接  |
| `BertDPCNN`   | 使用2D卷积，残差连接+区域嵌入（Deep Pyramid CNN） |
| `BertRCNN`    | RNN+CNN混合架构，使用LSTM捕获上下文信息          |
| `BertLSTM`    | BERT+LSTM架构                        |
>CNN模型对比：
> ```bash
> # BertCNN: 
> BERT → 卷积 → 池化 → 分类
> 
> # BertCNNPlus
> BERT → 卷积 → 池化 → 拼接CLS向量 → 分类
> 
> # BertDPCNN 
> BERT → 区域卷积 → 金字塔池化 → 分类
>
> # BertRCNN
> BERT → LSTM上下文增强 → 特征融合 → 类卷积变换 → 全局池化 → 分类```

## 🛠️ 快速开始

### 环境要求
- Python 3.8
- 依赖安装：`pip install -r requirements.txt`

### 数据准备
- 文本分类数据集，包含训练集、验证集、测试集
- 单条数据格式：`label \t text`，如：
- ```bash
  0	我爱北京天安门
  1	这个电影真好看
  ```
- 数据集存放路径：`data/`，数据集文件名：`train.tsv`、`dev.tsv`、`test.tsv`

### 模型训练和评估
0️⃣ 修改相应的运行文件的配置信息，如`run_STT2.py`：
```python
# 数据集路径
data_dir = "./data/SST2"
# 标签列表
label_list = ["0", "1"] # SST2数据集标签列表（SST2是情感二分类数据集）
# 所需运行的模型名称列表
model_name_list = ["BertOrigin", "BertATT", "BertCNN", "BertCNNPlus", "BertDPCNN", "BertRCNN", "BertLSTM"] # 运行七条基线模型
# 模型保存路径、缓存保存路径、日志保存路径
output_dir = "./sst2_output/"
cache_dir = "./sst2_cache/"
log_dir = "./sst2_log/"
# BERT预训练模型路径，中文数据集使用"bert-base-chinese"，英文数据集使用"bert-base-uncased"
model_name_or_path = "XXXX/XXX/bert-base-uncased"
```

1️⃣ 运行训练脚本：
```bash
# 注意修改执行的数据集脚本名称，如运行SST2数据集时，run.sh中应该是python3 run_SST2.py
CUDA_VISIBLE_DEVICES=0 bash run.sh

# run.sh中的参数说明：
max_seq_length：句子截断长度
num_train_epochs：训练轮数
do_train：是否训练
gpu_ids：使用的GPU编号，注意单卡训练时，gpu_ids为0
gradient_accumulation_steps：梯度累积步数
print_step：打印训练信息的步数（验证频率）
early_stop：早停步数，即验证集准确率连续early_stop次不再提升时，停止训练。当设置很大时，相当于关闭了早停功能。
train_batch_size：训练批次大小
```

2️⃣ 测试模型：
```bash
# 训练完成后，模型保存在`output_dir`目录下，如`./sst2_output/BertOrigin/`，包含模型文件、词表文件、配置文件等。
# 移除`run.sh`中的`do_train`参数，运行测试脚本
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

> **实现说明**：
默认配置下，模型会在每个训练周期（epoch）结束后自动在测试集上进行评估。如需禁用此功能，请在`train_evaluate.py`文件中设置`do_test_after_per_epoch=False`。
>
> 🔧 **高级配置**  
> - 在`BertXXX/args.py`中调整超参数进行定制实验
> - 在`BertXXX/BertXXX.py`文件中修改模型架构
>
> 📊 **性能监控**  
> - 可通过TensorBoard查看训练日志，日志路径示例：`./sst2_log/`
> - 每一训练轮次测试后的详细指标将导出至：`./sst2_output/BertXXX/BertXXX/metric_info_for_test.json`
>
> 🛑 **已知限制**：  
> - 当前版本仅支持`单标签`分类任务
> - 最大序列长度受GPU显存限制
> - 早停机制基于验证集的`加权F1分数`




