# Forest Cover Type Classification  
### NumPy | PyTorch | Scikit-learn (End-to-End Comparison)

This project implements **Forest Cover Type Classification** on the UCI Forest Cover dataset using **three different machine learning approaches**:

1. **Pure NumPy Neural Network (from scratch)**
2. **PyTorch Neural Network**
3. **Scikit-learn MLPClassifier**

The goal is to **compare implementations, training pipelines, and evaluation results** while keeping preprocessing and metrics consistent.

---

## Problem Statement

Given cartographic and environmental features, predict the **forest cover type** among **7 classes**:

| Label | Class Name |
|-----|-----------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## Project Structure
├── model_numpy.py # NumPy neural network definition
├── train_numpy.py # Training (NumPy from scratch)
├── eval_numpy.py # Evaluation (NumPy)
│
├── model_pytorch.py # PyTorch model definition
├── train_pytorch.py # Training (PyTorch)
├── eval_torch.py # Evaluation (PyTorch)
│
├── MLP_Classifier.py # sklearn MLPClassifier baseline
│
├── forest_cover_model_numpy.pth # Trained NumPy model checkpoint
├── forest_cover_model_pytorch.pth # Trained PyTorch model checkpoint
├── forest_cover_model_sklearn.pth # Trained sklearn model checkpoint
│
├── evaluation_results_numpy.txt # Detailed NumPy evaluation report
└── README.md


---

## 🧠 Model Architecture (All Implementations)

All three approaches use the **same neural architecture** for fair comparison:
Input (54 features)
↓
Dense (256) + ReLU
↓
Dense (128) + ReLU
↓
Dense (7) → Logits


- Initialization: **He Initialization**
- Output: **Raw logits**
- Task: **Multi-class classification**

---

## Preprocessing (Consistent Across Models)

- Standardization using training data statistics
- Feature clipping to range `[-5, 5]`
- Stratified train/validation split
- Label conversion:
  - Dataset labels: `1–7`
  - Internal model labels: `0–6`

Scaler parameters (`mean`, `scale`) are **saved in model checkpoints** to ensure consistent inference.

---

## Training Details

### NumPy (From Scratch)
- Manual forward & backward propagation
- Softmax + Cross-Entropy loss
- Mini-batch SGD
- L2 regularization
- Early stopping (Macro F1 based)

### PyTorch
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Early stopping with smoothed Macro F1

### Scikit-learn
- `MLPClassifier`
- Solver: SGD
- Early stopping enabled
- Parameters extracted and saved manually

---

## How to Train

### NumPy
```bash
python train_numpy.py --data_set covtype_train.csv --epochs 100


python train_pytorch.py --data_set covtype_train.csv --epochs 100
python MLP_Classifier.py --data_set covtype_train.csv --epochs 100

## Model Evaluation
python eval_numpy.py \
  --test_data covtype_test.csv \
  --model_path forest_cover_model_numpy.pth

python eval_torch.py \
  --test_data covtype_test.csv \
  --model_path forest_cover_model_pytorch.pth


