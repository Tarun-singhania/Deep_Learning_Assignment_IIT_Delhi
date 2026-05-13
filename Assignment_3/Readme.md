
# 🧠 Deep Learning Assignment 3 Submission

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Implemented-red)
![NLP](https://img.shields.io/badge/Task-NLP-green)
![Transformer](https://img.shields.io/badge/Model-Transformer-purple)
![GCN](https://img.shields.io/badge/Model-GCN-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This repository contains the submission for **Deep Learning Assignment 3**.

The assignment covers three important deep learning problems:

| Task | Problem Type | Main File |
|---|---|---|
| Task 1 | Sentiment Analysis using RNN models | `sentimentAnalysis.py` |
| Task 2 | Story/Text Generation using Transformer | `transformer.py` |
| Task 3 | Multi-label Node Classification using GCN | `train.py`, `model.py` |

The submission includes model implementations, training logic, inference logic, saved model weights, and the final report.

---

## 📁 Folder Structure

```text
JRB252023/
│
├── sentimentAnalysis.py
├── transformer.py
├── train.py
├── model.py
│
├── gcn_ppi_model.npz
├── README_Task3.md
├── Report.pdf
└── README.md
```

---

## 📄 File Description

| File | Description |
|---|---|
| `sentimentAnalysis.py` | Implements sentiment classification using Scratch RNN, PyTorch RNN, and BiLSTM |
| `transformer.py` | Implements Transformer-based text generation |
| `train.py` | Training and evaluation pipeline for NumPy-based GCN on PPI dataset |
| `model.py` | Contains the GCN architecture implemented from scratch using NumPy |
| `gcn_ppi_model.npz` | Saved GCN model checkpoint |
| `README_Task3.md` | Detailed README for Task 3 GCN implementation |
| `Report.pdf` | Final assignment report |
| `README.md` | Main repository overview and running guide |

---

# 📝 Task 1: Sentiment Analysis

## 🎯 Objective

The goal of this task is to classify text sentiment into one of three classes:

```text
negative
positive
neutral
```

This is a **multi-class text classification** problem.

The input is a tweet/text sentence, and the output is the predicted sentiment label.

---

## 🧠 Implemented Models

The sentiment analysis file supports three model options:

| Model Argument | Model Type | Description |
|---|---|---|
| `rnn_scratch` | Scratch RNN | RNN implemented manually using NumPy |
| `rnn` | PyTorch RNN | RNN implemented using `torch.nn.RNN` |
| `rnn_comp` | BiLSTM | Comparative BiLSTM model using PyTorch |

---

## 🔤 Text Preprocessing

The preprocessing pipeline includes:

```text
1. Convert text to lowercase
2. Remove URLs
3. Remove user mentions
4. Remove hashtag symbols
5. Remove special characters
6. Remove extra spaces
7. Tokenize text
8. Convert words to integer sequences
9. Apply padding
10. Use GloVe embeddings
```

---

## 📊 Evaluation Metrics

The sentiment model is evaluated using:

```text
Accuracy
Macro F1 Score
Micro F1 Score
Average F1 Score
```

The main score is:

```text
Average F1 = (Macro F1 + Micro F1) / 2
```

---

## 🚀 Train Scratch RNN

```bash
python sentimentAnalysis.py \
  --mode train \
  --model rnn_scratch \
  --dataset_path ./data/train.csv \
  --model_save_path ./models/rnn_scratch.pt
```

---

## 🚀 Train PyTorch RNN

```bash
python sentimentAnalysis.py \
  --mode train \
  --model rnn \
  --dataset_path ./data/train.csv \
  --model_save_path ./models/rnn_model.pt
```

---

## 🚀 Train BiLSTM Model

```bash
python sentimentAnalysis.py \
  --mode train \
  --model rnn_comp \
  --dataset_path ./data/train.csv \
  --model_save_path ./models/bilstm_model.pt
```

---

## ✅ Inference using Scratch RNN

```bash
python sentimentAnalysis.py \
  --mode inference \
  --model rnn_scratch \
  --dataset_path ./data/test.csv \
  --model_path ./models/rnn_scratch.pt \
  --output_path ./outputs/rnn_scratch_predictions.csv
```

---

## ✅ Inference using PyTorch RNN

```bash
python sentimentAnalysis.py \
  --mode inference \
  --model rnn \
  --dataset_path ./data/test.csv \
  --model_path ./models/rnn_model.pt \
  --output_path ./outputs/rnn_predictions.csv
```

---

## ✅ Inference using BiLSTM

```bash
python sentimentAnalysis.py \
  --mode inference \
  --model rnn_comp \
  --dataset_path ./data/test.csv \
  --model_path ./models/bilstm_model.pt \
  --output_path ./outputs/bilstm_predictions.csv
```

---

## 📂 Expected Sentiment Dataset Format

Training CSV should contain:

```text
tweet_id,text,sentiment
```

Example:

```text
1,I love this product,positive
2,This is not good,negative
3,It is okay,neutral
```

Test CSV should contain:

```text
tweet_id,text
```

If the test file also contains `sentiment`, the script can print evaluation metrics.

---

# 📖 Task 2: Transformer-Based Text Generation

## 🎯 Objective

The goal of this task is to generate story/text continuation using a Transformer-based language model.

The model learns the sequence pattern from story text and predicts the next token.

This is a **language modeling and text generation** task.

---

## 🧠 Transformer Architecture

The implemented Transformer contains:

```text
Token Embedding
        ↓
GloVe Embedding Projection
        ↓
Sinusoidal Positional Encoding
        ↓
Masked Multi-Head Self-Attention
        ↓
Residual Connection
        ↓
Output Linear Layer
        ↓
Vocabulary Prediction
```

---

## 🔥 Key Features

The Transformer implementation includes:

```text
Multi-head self-attention
Causal attention mask
Sinusoidal positional embeddings
GloVe embeddings
Beam search decoding
Cross-entropy loss
Perplexity calculation
Training loss curve
Training perplexity curve
```

---

## 📊 Evaluation Metric

The model tracks:

```text
Training Loss
Training Perplexity
```

Perplexity is calculated as:

```text
Perplexity = exp(loss)
```

Lower perplexity means the model is better at predicting the next word.

---

## 🚀 Train Transformer Model

```bash
python transformer.py \
  --mode train \
  --dataset_path ./data/stories.csv \
  --model_save_path ./models/transformer_model.pt
```

After training, the script saves:

```text
train_curves.png
```

This plot contains:

```text
Training Loss
Training Perplexity
```

---

## ✅ Run Transformer Inference

```bash
python transformer.py \
  --mode inference \
  --dataset_path ./data/stories_test.csv \
  --model_path ./models/transformer_model.pt \
  --output_path ./outputs/generated_stories.txt
```

---

## 📂 Expected Transformer Dataset Format

The CSV file should contain a column named:

```text
story
```

Example:

```text
story
Once upon a time there was a small village
A young boy went into the forest
```

---

# 🧬 Task 3: GCN on PPI Dataset

## 🎯 Objective

The goal of this task is to perform **multi-label node classification** on the **PPI dataset** using a Graph Convolutional Network.

Each node can belong to multiple labels at the same time.

This task is implemented **from scratch using NumPy**, without using PyTorch Geometric or any high-level GNN library.

---

## 🧠 GCN Architecture

The GCN model is implemented in `model.py`.

The architecture follows this structure:

```text
Input Node Features
        ↓
Graph Convolution Layer 1
        ↓
ReLU Activation
        ↓
Graph Convolution Layer 2
        ↓
ReLU Activation
        ↓
Graph Convolution Layer 3
        ↓
ReLU Activation
        ↓
Graph Convolution Layer 4
        ↓
Output Logits
```

The final output uses logits for multi-label prediction.

---

## 🔥 GCN Features

The GCN implementation includes:

```text
Manual forward propagation
Manual backward propagation
Graph adjacency normalization
Residual connections
Dropout
Input feature standardization
Binary Cross-Entropy loss
Positive class weighting
Adam optimizer from scratch
Gradient clipping
Learning rate decay
Early stopping
Micro-F1 evaluation
Training curve generation
```

---

## 📊 Evaluation Metric

The main evaluation metric is:

```text
Micro-F1 Score
```

Predictions are obtained using sigmoid probability with threshold:

```text
threshold = 0.5
```

---

## 🏆 Final GCN Configuration

```text
learning rate        = 0.0015
final learning rate  = 0.0005
hidden_dim           = 256
num_layers           = 4
dropout              = 0.1
input_dropout        = 0.0
weight_decay         = 1e-4
normalization        = symmetric
activation           = relu
seed                 = 42
patience             = 45
feature standardization = true
use residual         = true
use pos weight       = true
max_pos_weight       = 8.0
pos_weight_power     = 0.5
grad_clip_value      = 5.0
```

---

## 📈 Final GCN Result

The selected model is the best validation checkpoint.

```text
Best Validation Micro-F1: 0.5503
Best Epoch: 111
Final Training Loss: 0.2249
Final Validation Loss: 0.1807
Final Training Micro-F1: 0.5464
Final Validation Micro-F1: 0.5317
```

---

## 📂 Expected PPI Dataset Structure

Place the PPI dataset files inside:

```text
data/ppi/
```

Expected files:

```text
data/ppi/
│
├── ppi-G.json
├── ppi-feats.npy
├── ppi-class_map.json
└── ppi-id_map.json
```

---

## 🚀 Train GCN Model

```bash
python train.py \
  --mode train \
  --dataset_path ./data/ppi \
  --model_save_path ./gcn_ppi_model.npz \
  --artifacts_dir ./outputs/final_submission_run \
  --epochs 180 \
  --patience 45 \
  --lr 0.0015 \
  --lr_decay_factor 0.5 \
  --lr_decay_patience 12 \
  --min_lr 0.0005 \
  --hidden_dim 256 \
  --num_layers 4 \
  --dropout 0.1 \
  --input_dropout 0.0 \
  --weight_decay 1e-4 \
  --normalization symmetric \
  --max_pos_weight 8 \
  --pos_weight_power 0.5 \
  --seed 42
```

---

## ✅ Evaluate GCN Model

```bash
python train.py \
  --mode eval \
  --dataset_path ./data/ppi \
  --model_path ./gcn_ppi_model.npz \
  --artifacts_dir ./outputs/final_submission_eval \
  --eval_split all
```

---

## 📁 Generated GCN Output Files

After training, the following files are generated:

```text
outputs/final_submission_run/
│
├── logs/
│   ├── run_config.json
│   ├── summary.json
│   └── training_history.json
│
└── plots/
    ├── loss_curve.png
    └── micro_f1_curve.png
```

After evaluation:

```text
outputs/final_submission_eval/
│
└── logs/
    └── summary.json
```

---

# 📦 Requirements

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn matplotlib torch seaborn
```

Main libraries used:

| Library | Purpose |
|---|---|
| `numpy` | Scratch GCN implementation and numerical operations |
| `pandas` | CSV loading for NLP tasks |
| `torch` | RNN, BiLSTM, and Transformer implementation |
| `scikit-learn` | F1 score, accuracy, class weights, train-validation split |
| `matplotlib` | Training and validation plots |
| `seaborn` | Plot visualization support |

---

# 🔤 GloVe Embedding Requirement

For the NLP tasks, the code uses:

```text
glove.twitter.27B.100d.txt
```

Keep this file in the main working directory before running:

```text
sentimentAnalysis.py
transformer.py
```

Recommended structure:

```text
JRB252023/
│
├── glove.twitter.27B.100d.txt
├── sentimentAnalysis.py
├── transformer.py
└── ...
```

---

# ⚙️ Device Support

The NLP models automatically use GPU if available:

```text
cuda if available else cpu
```

The GCN task is implemented using NumPy and runs on CPU.

---

# ✅ Submission Checklist

| Item | Status |
|---|---|
| Sentiment analysis code | ✅ Included |
| Transformer code | ✅ Included |
| GCN training code | ✅ Included |
| GCN model architecture | ✅ Included |
| Saved GCN checkpoint | ✅ Included |
| Final report | ✅ Included |
| Task 3 README | ✅ Included |
| Main README | ✅ Included |

---

# 📌 Important Notes

- `sentimentAnalysis.py` supports three model modes: `rnn_scratch`, `rnn`, and `rnn_comp`.
- `transformer.py` expects a CSV file with a `story` column.
- `train.py` is used only for the GCN/PPI task.
- The PPI task uses Micro-F1 as the main metric.
- The GCN model is implemented from scratch using NumPy.
- Keep GloVe embeddings available before running NLP models.
- Keep dataset paths correct while running training or inference commands.

---

# 🏁 Final Summary

This repository demonstrates three deep learning pipelines:

```text
1. Sentiment Analysis using Scratch RNN, PyTorch RNN, and BiLSTM
2. Text Generation using Transformer with masked self-attention
3. Multi-label Node Classification using NumPy-based GCN on PPI
```

The submission includes complete source code, model checkpoint, report, and reproducible commands.

---

## 👨‍💻 Author

**Tarun Kumar**  
Deep Learning Assignment  
IIT Delhi
