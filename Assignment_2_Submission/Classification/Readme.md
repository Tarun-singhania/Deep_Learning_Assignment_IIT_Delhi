
# 🌲 Forest Cover Type Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-red)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-MLPClassifier-yellow)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This folder contains the implementation of the **Classification Task** for **Deep Learning Assignment 1**.

The objective of this task is to classify forest cover types using cartographic and geographical features such as elevation, slope, soil type, wilderness area, and distance-based measurements.

The problem is a **multi-class classification task** where the model predicts one of the **7 forest cover classes**.

---

## 🎯 Problem Statement

Given a set of input features describing geographical and environmental conditions, the goal is to predict the correct forest cover type.

The target column is:

```text
Cover_Type
```

There are 7 output classes:

| Class Label | Forest Cover Type |
|-----------:|-------------------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## 🧠 Implemented Models

This classification task is implemented using three different approaches:

| Approach | Description |
|---------|-------------|
| **NumPy Neural Network** | Neural network implemented from scratch using only NumPy |
| **PyTorch Neural Network** | Framework-based deep learning implementation using PyTorch |
| **Scikit-learn MLPClassifier** | Baseline model using Scikit-learn's MLPClassifier |

The purpose of using these three approaches is to compare manual implementation with standard machine learning and deep learning frameworks.

---

## 📁 Folder Structure

```text
Classification/
│
├── model_numpy.py
├── train_numpy.py
├── eval_numpy.py
│
├── model_pytorch.py
├── train_pytorch.py
├── eval_torch.py
│
├── MLP_Classifier.py
│
├── forest_cover_model_numpy.pth
├── forest_cover_model_pytorch.pth
├── forest_cover_model_sklearn.pth
│
├── evaluation_results_numpy.txt
└── README.md
```

---

## 📄 File Description

| File Name | Purpose |
|---------|---------|
| `model_numpy.py` | Defines the neural network architecture using NumPy |
| `train_numpy.py` | Trains the NumPy-based neural network from scratch |
| `eval_numpy.py` | Evaluates the trained NumPy model |
| `model_pytorch.py` | Defines the PyTorch neural network model |
| `train_pytorch.py` | Trains the PyTorch neural network |
| `eval_torch.py` | Evaluates the trained PyTorch model |
| `MLP_Classifier.py` | Trains the Scikit-learn MLPClassifier baseline |
| `forest_cover_model_numpy.pth` | Saved NumPy model checkpoint |
| `forest_cover_model_pytorch.pth` | Saved PyTorch model checkpoint |
| `forest_cover_model_sklearn.pth` | Saved Scikit-learn model checkpoint |
| `evaluation_results_numpy.txt` | Evaluation output of the NumPy model |

---

## 🏗️ Model Architecture

The neural network architecture used in the NumPy and PyTorch implementations is:

```text
Input Layer: 54 features
        ↓
Hidden Layer 1: 256 neurons + ReLU
        ↓
Hidden Layer 2: 128 neurons + ReLU
        ↓
Output Layer: 7 neurons
```

The final output layer produces logits for the 7 forest cover classes.

---

## ⚙️ Activation Function

The hidden layers use the **ReLU activation function**.

```text
ReLU(x) = max(0, x)
```

### Why ReLU?

ReLU is used because:

- It introduces non-linearity.
- It is computationally simple.
- It helps reduce the vanishing gradient problem.
- It works well in deep neural networks.

---

## 📉 Loss Function

Since this is a multi-class classification problem, the model uses **Cross-Entropy Loss**.

For PyTorch:

```python
nn.CrossEntropyLoss()
```

For the NumPy model, both **Softmax** and **Cross-Entropy Loss** are implemented manually.

---

## 📊 Evaluation Metrics

The main evaluation metric used is:

```text
Macro F1 Score
```

Macro F1 Score is useful because it gives equal importance to all classes, even if the dataset is imbalanced.

Other evaluation metrics include:

| Metric | Purpose |
|------|---------|
| Accuracy | Overall percentage of correct predictions |
| Macro F1 Score | Average F1 score across all classes equally |
| Micro F1 Score | Global F1 score considering all samples |
| Weighted F1 Score | F1 score weighted by class support |
| Confusion Matrix | Shows correct and incorrect class-wise predictions |
| Classification Report | Precision, recall, and F1 score for each class |

---

## 🧹 Data Preprocessing

The following preprocessing steps are used before training:

```text
1. Load the dataset.
2. Separate features and target labels.
3. Perform train-validation split.
4. Standardize the input features.
5. Clip standardized values to avoid extreme outliers.
6. Convert labels from 1-7 to 0-6 internally.
```

The features are standardized using:

```text
X_scaled = (X - mean) / standard_deviation
```

The scaler parameters are saved in the model checkpoint:

```text
scaler_mean
scaler_scale
```

This ensures that the same preprocessing is applied during evaluation.

---

## 📦 Requirements

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn torch matplotlib
```

Required packages:

| Library | Use |
|--------|-----|
| NumPy | Matrix operations and scratch implementation |
| Pandas | Dataset loading and preprocessing |
| Scikit-learn | Metrics, preprocessing, and MLPClassifier |
| PyTorch | Neural network training |
| Matplotlib | Plotting and visualization if required |

---

## 🗂️ Dataset Format

The dataset should be in CSV format.

Expected format:

```text
feature_1, feature_2, feature_3, ..., feature_54, Cover_Type
```

The target column should be named:

```text
Cover_Type
```

The class labels should be in the range:

```text
1 to 7
```

Example:

```text
Elevation,Aspect,Slope,...,Soil_Type40,Cover_Type
2596,51,3,...,0,5
```

---

## 🚀 How to Run

First, move inside the classification folder:

```bash
cd Assignment_1/Classification
```

---

## 🔵 Train NumPy Model

Run:

```bash
python train_numpy.py --data_set covtype_train.csv --epochs 100
```

This trains the neural network from scratch using NumPy.

The trained model will be saved as:

```text
forest_cover_model_numpy.pth
```

---

## 🔴 Train PyTorch Model

Run:

```bash
python train_pytorch.py --data_set covtype_train.csv --epochs 100
```

This trains the model using PyTorch.

The trained model will be saved as:

```text
forest_cover_model_pytorch.pth
```

---

## 🟡 Train Scikit-learn MLPClassifier

Run:

```bash
python MLP_Classifier.py --data_set covtype_train.csv --epochs 100
```

This trains the Scikit-learn baseline model.

The trained model will be saved as:

```text
forest_cover_model_sklearn.pth
```

---

## ✅ Evaluate NumPy Model

Run:

```bash
python eval_numpy.py --test_data covtype_test.csv --model_path forest_cover_model_numpy.pth
```

This will generate evaluation results and save them in:

```text
evaluation_results_numpy.txt
```

---

## ✅ Evaluate PyTorch Model

Run:

```bash
python eval_torch.py --test_data covtype_test.csv --model_path forest_cover_model_pytorch.pth
```

The evaluation script prints:

```text
Accuracy
Macro F1 Score
Micro F1 Score
Weighted F1 Score
Per-Class F1 Score
Classification Report
Confusion Matrix
```

---

## 💾 Checkpoint Details

Each saved model checkpoint contains important information required for evaluation.

```text
model_state_dict
input_dim
scaler_mean
scaler_scale
training_loss
```

The scaler values are important because the test data must be normalized in the same way as the training data.

---

## 🔬 NumPy Implementation Details

The NumPy model is implemented completely from scratch.

It includes:

```text
Forward propagation
ReLU activation
Softmax calculation
Cross-entropy loss
Backward propagation
Gradient calculation
Mini-batch training
L2 regularization
Early stopping
Validation using Macro F1 Score
```

This implementation helps in understanding the internal working of neural networks without relying on deep learning frameworks.

---

## 🔥 PyTorch Implementation Details

The PyTorch implementation uses:

```text
torch.nn.Module
Linear layers
ReLU activation
CrossEntropyLoss
Adam optimizer
Weight decay
Early stopping
GPU support
```

The device is automatically selected as:

```python
cuda if available else cpu
```

This makes the code compatible with both CPU and GPU environments.

---

## 🤖 Scikit-learn Implementation Details

The Scikit-learn baseline uses:

```python
MLPClassifier
```

Main configuration:

```text
Hidden Layers: 256, 128
Activation: ReLU
Solver: SGD
Batch Size: 128
Learning Rate: 0.002
Early Stopping: Enabled
```

This model acts as a reference baseline for comparing the custom NumPy and PyTorch implementations.

---

## 📈 Expected Output

During training, the scripts display training progress such as:

```text
Epoch number
Training loss
Validation score
Macro F1 score
```

During evaluation, the scripts display:

```text
FOREST COVER TYPE CLASSIFICATION - MODEL EVALUATION

Accuracy
Macro F1 Score
Micro F1 Score
Weighted F1 Score
Per-Class F1 Scores
Classification Report
Confusion Matrix
```

---

## 🧾 Important Notes

- The original class labels are from `1` to `7`.
- Internally, the labels are converted to `0` to `6` where required.
- The same scaler values from training must be used during testing.
- Do not rename checkpoint files unless the script paths are also updated.
- Keep the dataset files in the correct folder or provide the correct path.
- Macro F1 Score is the most important metric for this task.

---

## 🏁 Final Summary

This classification task demonstrates a complete deep learning pipeline for forest cover type prediction.

The project includes:

```text
Data preprocessing
Neural network implementation from scratch
PyTorch-based deep learning model
Scikit-learn baseline model
Model checkpoint saving
Evaluation using multiple metrics
Confusion matrix analysis
```

This folder provides a complete and reproducible solution for the classification part of **Deep Learning Assignment 1**.

---

## 👨‍💻 Author

**Tarun Kumar**  
Deep Learning Assignment  
IIT Delhi
