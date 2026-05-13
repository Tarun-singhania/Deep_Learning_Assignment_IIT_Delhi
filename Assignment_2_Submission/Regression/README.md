# NYC Taxi Trip Duration Prediction (Neural Network From Scratch)

This project implements an **end-to-end machine learning pipeline** to predict **NYC taxi trip duration** using a **custom-built neural network written entirely in NumPy** (no TensorFlow / PyTorch).

The system includes:
- Feature engineering on raw NYC Taxi data
- Manual implementation of a deep neural network regressor
- Training with mini-batch gradient descent & early stopping
- Model saving/loading
- Inference on unseen test data via CLI

---

## Project Highlights

**Neural Network from Scratch**
  - Arbitrary depth & width
  - ReLU / Sigmoid activations
  - Xavier & He initialization
  - Gradient clipping
  - Mini-batch SGD
  - Early stopping

**Advanced Feature Engineering**
  - Temporal features (weekday, rush hour, weekend)
  - Haversine distance calculation
  - Traffic & anomaly detection
  - NYC boundary filtering
  - Log-transformed target (`log1p`)

**Production-Ready Design**
  - Command-line interface (CLI)
  - Saved preprocessing parameters
  - Separate training & inference scripts

---

## Project Structure
├── nn_from_scratch.py # Custom Neural Network implementation
├── train_model.py # Training pipeline
├── inference.py # Model inference script
├── data/
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
├── models/ # Saved model files (.npz)
├── logs/ # Training logs
├── preprocessing_params/ # Feature scaling parameters
└── README.md

## Neural Network Architecture

- Fully Connected Feedforward Network
- Configurable hidden layers
- Output layer: **Linear activation** (Regression)
- Loss Function: **Mean Squared Error (MSE)**

Example:
Input → [256] → [128] → [64] → Output


---

## Feature Engineering Overview

| Category | Features |
|--------|---------|
| Time-based | day_of_week, hour, weekend, rush_hour |
| Distance | Haversine trip distance (KM) |
| Traffic | rush hour traffic flag |
| Anomalies | unusual daily traffic detection |
| Geography | pickup/dropoff NYC boundary flags |

---

## How to Train the Model

```bash
python train_model.py \
  --train_data data/train.csv \
  --val_data data/val.csv \
  --hidden_layers 256 128 64 \
  --activation relu \
  --learning_rate 0.001 \
  --batch_size 32 \
  --epochs 300 \
  --model_dir models \
  --logs_dir logs \
  --pre_processing_params_dir preprocessing_params

**How to run inference.py**
python inference.py \
  --model_path models/model_hidden_256_128_64_relu.npz \
  --test_data data/test.csv \
  --pre_processing_params_path preprocessing_params/pre_process_hidden_256_128_64_relu.json \
  --predictions_path predictions
