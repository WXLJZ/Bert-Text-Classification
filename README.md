# 🤖 BERT Text Classification

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFCE56?logo=huggingface&logoColor=black)

✨ **BERT** - A Transformer-based text classification framework built with PyTorch.  
📝 *Note: This project is modified from the GitHub open-source repository [BERT-TextClassification](https://github.com/songyingxin/Bert-TextClassification).*
---
📚 Multilingual Docs: [English](https://github.com/WXLJZ/Bert-Text-Classification/blob/main/README.md) | [简体中文](https://github.com/WXLJZ/Bert-Text-Classification/blob/main/README_ZH.md)

---

## 🚀 Key Features
- ⚡ **7 BERT Variants**: Comprehensive baseline implementations including Original BERT, BERTATT, BERTLSTM, BERTCNN, BERTCNN+, BERTRCNN, BERTDPCNN
- 🌐 **Modern Framework Support**: Migrated from legacy `pytorch-pretrained-bert` to contemporary `transformers` library for streamlined environment setup
- 🎯 **Full Training Lifecycle**: Supports model training, validation, testing, and prediction

## 📦 Model Architecture
| Model           | Architecture Description                          |
|-----------------|---------------------------------------------------|
| `BertOrigin`    | Original BERT implementation                      |
| `BertATT`       | Dynamic token selection via attention mechanisms  |
| `BertCNN`       | Local feature extraction with CNN                 |
| `BertCNNPlus`   | Enhanced CNN with global CLS vector concatenation |
| `BertDPCNN`     | Deep Pyramid CNN with residual connections        |
| `BertRCNN`      | Hybrid RNN-CNN architecture with LSTM             |
| `BertLSTM`      | Sequential modeling with LSTM layer               |

> **Architecture Comparison**:
> ```bash
> # BertCNN: 
> BERT → Conv → Pooling → Classification
> 
> # BertCNNPlus
> BERT → Conv → Pooling → CLS Concatenation → Classification
> 
> # BertDPCNN 
> BERT → Regional Conv → Pyramid Pooling → Classification
>
> # BertRCNN
> BERT → LSTM Context → Feature Fusion → Conv-like Transform → Global Pooling → Classification
> ```

## 🛠️ Getting Started

### Requirements
- Python 3.8
- Dependencies: `pip install -r requirements.txt`

### Data Preparation
- Text classification dataset in TSV format:
  - File structure: `label \t text` (per line)
  - Example:
  ```bash
  0	I love Beijing Tiananmen
  1	This movie is fantastic
  ```
- Dataset paths: `data/`，dataset name: `train.tsv`、`dev.tsv`、`test.tsv`

### Training & Evaluation
0️⃣ Configure parameters in execution file (e.g., `run_SST2.py`):
```python
# Dataset path
data_dir = "./data/SST2"

# Label list (for classification)
label_list = ["0", "1"]

# Model selection (choose from available architectures)
model_name_list = ["BertOrigin", "BertATT", "BertCNN", "BertCNNPlus", "BertDPCNN", "BertRCNN", "BertLSTM"]

# System paths
output_dir = "./sst2_output/"    # Model checkpoints
cache_dir = "./sst2_cache/"      # Preprocessing cache
log_dir = "./sst2_log/"          # Training logs

# Model path (BERT pre-trained model)
model_name_or_path = "XXXX/XXX/bert-base-uncased"
```
1️⃣ Start training:
```BASH
# Execute with GPU (modify dataset script name as needed)
CUDA_VISIBLE_DEVICES=0 bash run.sh

# Key parameters in run.sh:
max_seq_length        # Sequence truncation length
num_train_epochs      # Training epochs
do_train              # Enable training mode
gpu_ids               # GPU device ID(s)
gradient_accumulation_steps  # Gradient accumulation steps
print_step            # Validation frequency (steps)
early_stop            # Early stopping patience
train_batch_size      # Batch size
```
2️⃣ Model evaluation:

```bash
# Trained models are saved under output_dir, e.g.:
# ./sst2_output/BertOrigin/ (contains model weights, configs, vocab)

# Remove 'do_train' parameter in run.sh for evaluation mode
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

> Implementation Note:
By default, models are evaluated on the test set after each epoch. To disable this behavior, set `do_test_after_per_epoch=False` in `train_evaluate.py`.
>
> 🔧 Advanced Configuration
> - Adjust hyperparameters in `BertXXX/args.py` for custom experiments
> - Modify model architectures in `BertXXX/BertXXX.py`
>
> 📊 Performance Monitoring
> - TensorBoard logs available in `log_dir`, e.g.: `./sst2_log/`
> - Detailed metrics in doing test after per epoch are exported to `./sst2_output/BertXXX/BertXXX/metric_info_for_test.json`
>
> 🛑 Known Limitations:
> - Current implementation supports `single-label` classification only
> - Maximum sequence length constrained by GPU memory
> - Early stopping based on validation `weighted_f1 score`
