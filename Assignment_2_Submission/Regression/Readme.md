# 🚕 NYC Taxi Trip Duration Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Neural%20Network%20From%20Scratch-orange)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-MLPRegressor-yellow)
![Task](https://img.shields.io/badge/Task-Regression-green)
![Loss](https://img.shields.io/badge/Loss-MSE-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This folder contains the implementation of the **Regression Task** for **Deep Learning Assignment 1**.

The objective of this task is to predict the **NYC taxi trip duration** using trip-related information such as pickup time, pickup location, dropoff location, passenger count, store-and-forward flag, distance-based features, and traffic-related features.

This is a **supervised regression problem**, where the model predicts a continuous numerical value:

```text
Trip Duration
```

---

## 🎯 Problem Statement

Given taxi trip details, the goal is to predict how long the taxi trip will take.

The target variable is:

```text
trip_duration
```

The prediction output is a continuous value representing the estimated duration of the trip.

---

## 🧠 Implemented Models

This regression task is implemented using two main approaches:

| Approach | Description |
|---------|-------------|
| **NumPy Neural Network** | Neural network implemented from scratch using only NumPy |
| **Scikit-learn MLPRegressor** | Baseline model using Scikit-learn's MLPRegressor |

The main focus of this task is to understand how a regression neural network works internally by manually implementing forward propagation, loss computation, backpropagation, and model training.

---

## 📁 Folder Structure

```text
Regression/
│
├── nn_from_scratch.py
├── train_model.py
├── inference.py
├── MLP_Regressor.py
├── preprocessing_params.json
│
├── models/
│   ├── model_hidden_1_relu.npz
│   ├── model_hidden_5_relu.npz
│   ├── model_hidden_10_relu.npz
│   ├── model_hidden_50_relu.npz
│   ├── model_hidden_100_relu.npz
│   ├── model_hidden_256_relu.npz
│   ├── model_hidden_256_128_relu.npz
│   ├── model_hidden_256_128_64_relu.npz
│   └── model_hidden_256_128_64_32_relu.npz
│
├── pre_processing/
│   ├── pre_process_hidden_256_128_64_relu.json
│   └── ...
│
├── loss_logs/
│   ├── logs_hidden_256_128_64_relu.npz
│   └── ...
│
├── Images_Plot_Scratch/
│   ├── loss_curve_hidden_256_128_64_relu.png
│   └── ...
│
├── Images_Plot_MLP_Regressor/
│   ├── train_loss_relu.png
│   └── val_loss_relu.png
│
└── README.md
```

---

## 📄 File Description

| File Name | Purpose |
|---------|---------|
| `nn_from_scratch.py` | Defines the custom neural network regressor using NumPy |
| `train_model.py` | Training pipeline for the NumPy-based regression model |
| `inference.py` | Generates predictions on unseen test data |
| `MLP_Regressor.py` | Trains Scikit-learn MLPRegressor baseline models |
| `preprocessing_params.json` | Stores preprocessing information |
| `models/` | Contains trained NumPy model checkpoints in `.npz` format |
| `pre_processing/` | Contains saved feature scaling parameters for different models |
| `loss_logs/` | Stores training and validation loss logs |
| `Images_Plot_Scratch/` | Contains loss curve plots for scratch models |
| `Images_Plot_MLP_Regressor/` | Contains train and validation loss plots for MLPRegressor |

---

## 🏗️ Neural Network Architecture

The regression model is a fully connected feedforward neural network.

Example architecture:

```text
Input Layer
     ↓
Hidden Layer 1: 256 neurons + ReLU
     ↓
Hidden Layer 2: 128 neurons + ReLU
     ↓
Hidden Layer 3: 64 neurons + ReLU
     ↓
Output Layer: 1 neuron
```

The output layer has only **one neuron** because this is a regression problem.

```text
Output = Predicted Trip Duration
```

---

## 🔬 Architecture Experiments

Multiple hidden-layer configurations were tested.

```text
[1]
[5]
[10]
[50]
[100]
[256]
[256, 128]
[256, 128, 64]
[256, 128, 64, 32]
```

Both activation functions were tested:

```text
ReLU
Sigmoid
```

This helps compare the effect of model depth, width, and activation function on regression performance.

---

## ⚙️ Activation Functions

Two activation functions are used in the hidden layers.

### ReLU Activation

```text
ReLU(x) = max(0, x)
```

ReLU is useful because:

- It is simple and fast.
- It helps reduce the vanishing gradient problem.
- It works well for deeper neural networks.

### Sigmoid Activation

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

Sigmoid is useful for comparison, but it can suffer from vanishing gradients when the network becomes deep.

---

## 📉 Loss Function

Since this is a regression task, the model uses **Mean Squared Error Loss**.

```text
MSE = mean((y_true - y_pred)^2)
```

MSE penalizes large prediction errors more strongly because the error term is squared.

---

## 📊 Evaluation Metric

The main evaluation objective is to minimize prediction error.

Common regression metrics include:

| Metric | Meaning |
|------|---------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| Validation Loss | Loss on validation data during training |

In this project, training and validation loss curves are used to compare different models.

---

## 🧹 Feature Engineering

The raw taxi dataset is transformed into useful numerical features before training.

The preprocessing pipeline includes:

```text
1. Convert pickup_datetime into datetime format.
2. Extract day of week.
3. Extract pickup hour.
4. Identify weekend trips.
5. Identify rush-hour trips.
6. Convert longitude and latitude values into radians.
7. Calculate Haversine distance between pickup and dropoff points.
8. Encode store_and_fwd_flag.
9. Detect traffic-related conditions.
10. Detect anomalous trip days.
11. Create pickup/dropoff location range indicators.
12. Remove unnecessary columns.
13. Apply log transformation on trip_duration.
14. Standardize numerical features.
```

---

## 🌍 Haversine Distance Feature

The Haversine formula is used to calculate the approximate distance between pickup and dropoff coordinates.

This is important because trip distance is one of the strongest factors affecting trip duration.

```text
Distance = distance between pickup point and dropoff point on Earth
```

The distance is calculated in kilometers.

---

## ⏰ Time-Based Features

The model uses pickup time information to create important features.

| Feature | Meaning |
|------|---------|
| `day_of_week` | Day number from Monday to Sunday |
| `day_of_hours` | Hour of the day |
| `Weekend` | Whether the trip happened on weekend |
| `rush_hour` | Whether the trip happened during rush hour |
| `is_traffic` | Traffic condition indicator |
| `is_anomaly` | Unusual traffic day indicator |

These features help the model understand time-dependent traffic behavior.

---

## 🗽 Location-Based Features

Pickup and dropoff coordinates are used to create geographical features.

| Feature | Meaning |
|------|---------|
| `pickup_in_range` | Whether pickup point lies inside selected NYC range |
| `dropoff_in_range` | Whether dropoff point lies inside selected NYC range |
| `trip_distance(KM)` | Approximate trip distance in kilometers |

These features help identify whether the taxi trip is happening within a common city region or outside the expected range.

---

## 🔁 Target Transformation

The target value `trip_duration` is transformed using:

```text
log1p(trip_duration)
```

This means:

```text
y_train = log(1 + trip_duration)
```

### Why log transformation?

Trip duration can have very large values due to outliers or unusually long trips.  
Log transformation helps make the target distribution more stable and easier for the neural network to learn.

During inference, predictions are converted back using:

```text
expm1(prediction)
```

This means:

```text
trip_duration = exp(prediction) - 1
```

---

## 📦 Requirements

Install the required libraries using:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Required packages:

| Library | Use |
|--------|-----|
| NumPy | Neural network implementation and matrix operations |
| Pandas | Dataset loading and preprocessing |
| Scikit-learn | MLPRegressor baseline and metrics |
| Matplotlib | Loss curve visualization |

---

## 🗂️ Dataset Format

The dataset should be in CSV format.

Expected columns include:

```text
id
vendor_id
pickup_datetime
dropoff_datetime
passenger_count
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
store_and_fwd_flag
trip_duration
```

For test data, the `trip_duration` column may be absent.

---

## 🚀 How to Run

First, move inside the regression folder:

```bash
cd Assignment_1/Regression
```

---

## 🔵 Train NumPy Model From Scratch

Example command:

```bash
python train_model.py \
  --train_data train.csv \
  --val_data val.csv \
  --hidden_layers 256 128 64 \
  --activation relu \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 300 \
  --model_dir models \
  --logs_dir loss_logs \
  --pre_processing_params_dir pre_processing
```

This command trains a NumPy neural network with architecture:

```text
Input → 256 → 128 → 64 → Output
```

The trained model will be saved as:

```text
models/model_hidden_256_128_64_relu.npz
```

The preprocessing parameters will be saved as:

```text
pre_processing/pre_process_hidden_256_128_64_relu.json
```

The loss logs will be saved as:

```text
loss_logs/logs_hidden_256_128_64_relu.npz
```

---

## 🟢 Train Model With Sigmoid Activation

Example command:

```bash
python train_model.py \
  --train_data train.csv \
  --val_data val.csv \
  --hidden_layers 256 128 64 \
  --activation sigmoid \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 300 \
  --model_dir models \
  --logs_dir loss_logs \
  --pre_processing_params_dir pre_processing
```

This is useful for comparing Sigmoid with ReLU.

---

## 🟡 Train Scikit-learn MLPRegressor

Run:

```bash
python MLP_Regressor.py
```

This script trains multiple Scikit-learn MLPRegressor architectures and saves training/validation loss plots inside:

```text
Images_Plot_MLP_Regressor/
```

Generated plots:

```text
train_loss_relu.png
val_loss_relu.png
```

---

## ✅ Run Inference

Use the trained model and corresponding preprocessing file.

Example:

```bash
python inference.py \
  --model_path models/model_hidden_256_128_64_relu.npz \
  --test_data test.csv \
  --pre_processing_params_path pre_processing/pre_process_hidden_256_128_64_relu.json \
  --predictions_path predictions
```

The output file will be saved as:

```text
predictions/predictions.csv
```

The prediction file contains a single column without header, as required for submission.

---

## 💾 Saved Model Details

Each trained NumPy model is saved in `.npz` format.

Example:

```text
model_hidden_256_128_64_relu.npz
```

The filename stores the architecture information:

```text
model_hidden_<hidden_layer_sizes>_<activation>.npz
```

Example:

```text
model_hidden_256_128_64_relu.npz
```

means:

```text
Hidden layers: 256, 128, 64
Activation: ReLU
```

---

## 🧾 Preprocessing Parameter Details

For every trained model, a corresponding preprocessing file is saved.

Example:

```text
pre_process_hidden_256_128_64_relu.json
```

This file stores:

```text
feature_mean
feature_std
target_transform
```

These values are required during inference because test data must be normalized in the same way as training data.

---

## 📈 Loss Logs

Training and validation loss values are saved in `.npz` files.

Example:

```text
logs_hidden_256_128_64_relu.npz
```

These logs contain:

```text
training_loss
val_loss
total_time
```

They are useful for analyzing convergence, overfitting, and model comparison.

---

## 📊 Generated Plots

The folder `Images_Plot_Scratch/` contains loss curves for different NumPy neural network experiments.

Example plots:

```text
loss_curve_hidden_256_128_64_relu.png
loss_curve_hidden_256_128_64_sigmoid.png
```

The folder `Images_Plot_MLP_Regressor/` contains plots for the Scikit-learn MLPRegressor baseline.

```text
train_loss_relu.png
val_loss_relu.png
```

---

## 🔬 NumPy Implementation Details

The custom neural network includes:

```text
Forward propagation
Linear layers
ReLU activation
Sigmoid activation
Mean Squared Error loss
Backpropagation
Gradient calculation
Mini-batch training
Gradient clipping
Early stopping
Model saving and loading
Prediction on unseen data
```

This implementation is useful for understanding the internal working of neural networks without using high-level deep learning libraries.

---

## 🤖 Scikit-learn Implementation Details

The Scikit-learn baseline uses:

```python
MLPRegressor
```

Experimental architectures include:

```text
1 hidden layer:  256
2 hidden layers: 256, 128
3 hidden layers: 256, 128, 64
4 hidden layers: 256, 128, 64, 32
```

Main configuration:

```text
Activation: ReLU
Solver: SGD
Batch Size: 32
Learning Rate: 0.00001
Epochs: 30
Warm Start: Enabled
```

This baseline is used to compare the custom NumPy implementation with a standard machine learning library.

---

## 🧪 Experiment Summary

The project compares different neural network configurations based on validation loss.

Important comparisons include:

```text
Small network vs large network
Shallow network vs deep network
ReLU activation vs Sigmoid activation
Scratch NumPy model vs Scikit-learn MLPRegressor
```

The final model can be selected based on the lowest validation loss.

---

## 🧾 Important Notes

- This is a regression task, so the output is a continuous value.
- The output layer uses linear activation.
- The loss function is Mean Squared Error.
- The target variable is transformed using `log1p`.
- Predictions are converted back using `expm1`.
- The same preprocessing parameters must be used during training and inference.
- The model filename and preprocessing filename should match the same architecture.
- Keep the dataset files in the correct folder or provide the correct path in the command.

---

## 🛠️ Resubmission Fix Note

The updated `inference.py` correctly loads the model architecture from the model filename.

For example, from:

```text
model_hidden_256_128_64_relu.npz
```

it identifies:

```text
Hidden layers: 256, 128, 64
Activation: ReLU
```

This ensures that the inference model architecture matches the saved weights.

The inference script also saves predictions properly as:

```text
predictions.csv
```

inside the given prediction output folder.

---

## 🏁 Final Summary

This regression task demonstrates a complete neural network pipeline for NYC taxi trip duration prediction.

The project includes:

```text
Data preprocessing
Feature engineering
Distance calculation using Haversine formula
Target log transformation
Neural network implementation from scratch
Scikit-learn baseline comparison
Model checkpoint saving
Loss logging
Inference script
Prediction file generation
```

This folder provides a complete and reproducible solution for the regression part of **Deep Learning Assignment 1**.

---

## 👨‍💻 Author

**Tarun Kumar**  
Deep Learning Assignment  
IIT Delhi
