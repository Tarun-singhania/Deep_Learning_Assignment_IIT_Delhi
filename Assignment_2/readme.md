
# 🧠 Deep Learning Assignment 2 Submission

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Task-Computer%20Vision-green)
![Classification](https://img.shields.io/badge/Bird-Classification-orange)
![Segmentation](https://img.shields.io/badge/Cell-Segmentation-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview

This repository contains the complete submission for **Deep Learning Assignment 2**.

The assignment includes two major computer vision tasks:

| Task | Problem Type | Main File |
|---|---|---|
| 🐦 Bird Classification | Image Classification | `bird_classification.py` |
| 🧬 Cell Segmentation | Semantic Segmentation | `cell_segmentation.py` |

The submission includes training code, inference code, timestamp proof images, model path information, and final report.

---

## 📁 Folder Structure

```text
DL_A2/
│
├── bird_classification.py
├── cell_segmentation.py
│
├── bird_timestamp.jpg
├── cell_timestamp.jpg
│
├── modelpath.txt.txt
├── Report.pdf
└── README.md
```

---

## 📄 File Description

| File | Description |
|---|---|
| `bird_classification.py` | Complete training and inference code for bird image classification |
| `cell_segmentation.py` | Complete training and inference code for cell image segmentation |
| `bird_timestamp.jpg` | Timestamp proof image for bird classification task |
| `cell_timestamp.jpg` | Timestamp proof image for cell segmentation task |
| `modelpath.txt.txt` | Contains Google Drive link for saved trained model weights |
| `Report.pdf` | Final report explaining methodology, model design, results, and observations |
| `README.md` | Project overview and running instructions |

---

# 🐦 Task 1: Bird Classification

## 🎯 Objective

The goal of the bird classification task is to classify bird images into their correct categories using a deep learning model.

This is a **multi-class image classification** problem.

The model takes an RGB bird image as input and predicts the corresponding bird class label.

---

## 🧠 Model Used

A custom **ResNet-style Convolutional Neural Network** is implemented using PyTorch.

The model contains:

```text
Initial Convolution Layer
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Residual Block 1
        ↓
Residual Block 2
        ↓
Residual Block 3
        ↓
Residual Block 4
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Fully Connected Classification Layer
```

---

## 🔥 Key Features

The classification model includes:

```text
Custom Residual Blocks
Batch Normalization
ReLU Activation
Dropout Regularization
Class Weighted Loss
Label Smoothing
Data Augmentation
Macro F1 Score Evaluation
Micro F1 Score Evaluation
Best Model Saving
```

---

## 🧹 Data Augmentation

The following augmentation techniques are used during training:

```text
Random Horizontal Flip
Random Rotation
Random Resized Crop
Color Jitter
```

These augmentations help improve generalization and reduce overfitting.

---

## 📊 Evaluation Metric

The validation score is calculated using:

```text
Final Score = (Macro F1 Score + Micro F1 Score) / 2
```

This ensures that both overall accuracy and class-wise balance are considered.

---

## 🚀 Train Bird Classification Model

```bash
python bird_classification.py \
  --mode train \
  --dataset_path /path/to/bird_dataset \
  --model_save_path bird_model.pth
```

Expected dataset should contain:

```text
bird_dataset/
│
├── train.csv
├── image_1.jpg
├── image_2.jpg
├── image_3.jpg
└── ...
```

The `train.csv` file should contain:

```text
image_name,label
```

---

## ✅ Run Bird Classification Inference

```bash
python bird_classification.py \
  --mode inference \
  --dataset_path /path/to/bird_test_dataset \
  --model_path bird_model.pth \
  --output_path bird_predictions.csv
```

The output file will be:

```text
bird_predictions.csv
```

Output format:

```text
image_name,label
```

---

# 🧬 Task 2: Cell Segmentation

## 🎯 Objective

The goal of the cell segmentation task is to perform pixel-level classification of cell images.

This is a **semantic segmentation** problem, where every pixel in the image is assigned a class label.

The output is a colored segmentation mask.

---

## 🧠 Model Used

A custom **U-Net architecture** is implemented using PyTorch.

The model follows an encoder-decoder structure:

```text
Input Image
    ↓
Encoder Block 1
    ↓
Encoder Block 2
    ↓
Encoder Block 3
    ↓
Encoder Block 4
    ↓
Bottleneck
    ↓
Decoder Block 1
    ↓
Decoder Block 2
    ↓
Decoder Block 3
    ↓
Decoder Block 4
    ↓
Output Segmentation Mask
```

---

## 🔥 Key Features

The segmentation model includes:

```text
U-Net Encoder-Decoder Architecture
Skip Connections
Batch Normalization
ReLU Activation
Transposed Convolutions
Class Weighted Loss
Cross Entropy Loss
Focal Loss
Dice Loss
Combined Loss Function
Mean IoU Evaluation
Mean Dice Score Evaluation
```

---

## 🎨 Segmentation Classes

The masks use color-coded labels.

| RGB Color | Class ID |
|---|---:|
| `(0, 0, 0)` | 0 |
| `(255, 255, 0)` | 1 |
| `(255, 0, 0)` | 2 |
| `(0, 255, 0)` | 3 |
| `(0, 0, 255)` | 4 |

The background class is excluded while calculating final mean IoU and Dice score.

---

## 🧹 Data Augmentation

The following augmentation techniques are used for segmentation:

```text
Horizontal Flip
Vertical Flip
Rotation
Brightness Change
Gaussian Noise
Zoom
Gaussian Blur
```

These augmentations are applied to both image and mask so that pixel-level alignment is preserved.

---

## 📉 Loss Function

The final training loss is a combined loss:

```text
Combined Loss = Cross Entropy Loss + Focal Loss + 2 × Dice Loss
```

This helps the model learn both pixel-level accuracy and object shape overlap.

---

## 📊 Evaluation Metrics

The segmentation model is evaluated using:

```text
Mean IoU
Mean Dice Score
Final Score
```

Final score:

```text
Final Score = (Mean IoU + Mean Dice Score) / 2
```

---

## 🚀 Train Cell Segmentation Model

```bash
python cell_segmentation.py \
  --mode train \
  --dataset_path /path/to/cell_dataset \
  --model_save_path cell_model.pth
```

Expected dataset structure:

```text
cell_dataset/
│
├── train_images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
│
├── train_masks/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
```

---

## ✅ Run Cell Segmentation Inference

```bash
python cell_segmentation.py \
  --mode inference \
  --dataset_path /path/to/cell_dataset \
  --model_path cell_model.pth \
  --output_path segmentation_outputs
```

Expected test structure:

```text
cell_dataset/
│
├── test_images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
```

The output folder will contain predicted segmentation masks:

```text
segmentation_outputs/
│
├── image_1.png
├── image_2.png
└── ...
```

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install torch torchvision numpy pandas scikit-learn pillow opencv-python
```

Main libraries used:

| Library | Purpose |
|---|---|
| `torch` | Model building and training |
| `torchvision` | Image transformations |
| `numpy` | Numerical operations |
| `pandas` | CSV handling |
| `scikit-learn` | Train-validation split and F1 score |
| `Pillow` | Image loading for classification |
| `OpenCV` | Image and mask processing for segmentation |

---

## 💾 Model Weights

The trained model path is provided in:

```text
modelpath.txt.txt
```

This file contains the Google Drive link for saved trained models.

Download the required model weights and pass the path using:

```bash
--model_path
```

during inference.

---

## 🧪 Training Summary

| Task | Model | Main Metric |
|---|---|---|
| Bird Classification | Custom ResNet CNN | Macro F1 + Micro F1 |
| Cell Segmentation | Custom U-Net | Mean IoU + Mean Dice |

Both models are implemented in PyTorch and support GPU acceleration if CUDA is available.

---

## ⚙️ Device Support

The code automatically selects GPU if available:

```python
cuda if available else cpu
```

This allows the same code to run on:

```text
CPU
GPU
Google Colab
Kaggle
Local System
```

---

## 📌 Important Notes

- Keep the dataset path correct while running training or inference.
- For bird classification, `train.csv` is required for label mapping.
- For cell segmentation, image-mask names should be properly aligned.
- During segmentation, masks should use the predefined RGB color format.
- Download trained model weights before inference.
- Use the correct model path with `--model_path`.

---

## ✅ Submission Checklist

| Item | Status |
|---|---|
| Bird classification code | ✅ Included |
| Cell segmentation code | ✅ Included |
| Training mode | ✅ Implemented |
| Inference mode | ✅ Implemented |
| Timestamp images | ✅ Included |
| Model path file | ✅ Included |
| Final report | ✅ Included |
| README file | ✅ Included |

---

## 🏁 Final Summary

This assignment demonstrates two important computer vision pipelines:

```text
1. Image Classification using a custom ResNet-style CNN
2. Semantic Segmentation using a custom U-Net model
```
---

## 👨‍💻 Author

**Tarun Kumar**  
Deep Learning Assignment 2  
IIT Delhi
