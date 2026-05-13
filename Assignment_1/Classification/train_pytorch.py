"""
train.py - Training Script for Forest Cover Classification

Students: Implement your complete training pipeline in this file.

REQUIRED OUTPUT:
Your script must save a model checkpoint named 'forest_cover_model.pth' containing:
- 'model_state_dict': model.state_dict()
- 'input_dim': number of input features
- 'scaler_mean': mean values used for standardization (numpy array)
- 'scaler_scale': scale values used for standardization (numpy array)

Example:
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, 'forest_cover_model.pth')
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_pytorch import ForestCoverNet

# Some Utility Functions used by me
def stratified_split(X, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)

    X_train, X_val = [], []
    y_train, y_val = [], []

    for cls in range(1, 8):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        split = int(len(idx) * (1 - val_ratio))
        X_train.append(X[idx[:split]])
        y_train.append(y[idx[:split]])
        X_val.append(X[idx[split:]])
        y_val.append(y[idx[split:]])

    return (
        np.vstack(X_train),
        np.vstack(X_val),
        np.hstack(y_train),
        np.hstack(y_val),
    )


def macro_f1(y_true, y_pred):
    f1s = []
    for cls in range(1, 8):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1s.append(2 * p * r / (p + r + 1e-8))
    return np.mean(f1s)


@torch.no_grad()
def macro_f1_from_model(model, loader, device):
    model.eval()
    preds, targets = [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = torch.argmax(logits, dim=1) + 1

        preds.extend(pred.cpu().numpy())
        targets.extend(y.cpu().numpy())

    return macro_f1(np.array(targets), np.array(preds))

# TODO: Implement your complete training pipeline
# - Load data from 'covtype_train.csv'
# - Preprocess and split data
# - Create model, define loss and optimizer
# - Train the model
# - Save model checkpoint with required keys

# Training Function
def train(args):
    # 1. Load data
    df = pd.read_csv(args.data_set)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    input_dim = X.shape[1]
    num_classes = 7

    # 2. Stratified split
    X_train, X_val, y_train, y_val = stratified_split(X, y)

    # 3. StandardScaler
    scaler_mean = np.mean(X_train, axis=0)
    scaler_scale = np.std(X_train, axis=0) + 1e-8

    X_train = (X_train - scaler_mean) / scaler_scale
    X_val   = (X_val   - scaler_mean) / scaler_scale

    X_train = np.clip(X_train, -5, 5)
    X_val   = np.clip(X_val, -5, 5)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train - 1, dtype=torch.long)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val - 1, dtype=torch.long)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val + 1),
        batch_size=512,
        shuffle=False
    )

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForestCoverNet(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3,
        weight_decay=5e-4  # L2 regularization
    )

    # 5. Training loop with Early Stopping
    best_f1 = 0.0
    patience = 15
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses =[]

        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_f1 = macro_f1_from_model(model, val_loader, device)
        smoothed_f1 = 0.9 * best_f1 + 0.1 * val_f1

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {epoch_loss:.4f} | Val Macro-F1: {val_f1:.4f}"
        )

        if smoothed_f1 > best_f1:
            best_f1 = smoothed_f1
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break

    # 6. Save checkpoint
    torch.save({
        "model_state_dict": best_state,
        "training_loss":train_losses,
        "input_dim": input_dim,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale
    }, "forest_cover_model_pytorch.pth")

    print("Model saved as forest_cover_model_pytorch.pth")

# Main function
if __name__ == "__main__":
    # Your training code here
    parser = argparse.ArgumentParser(description="Train Forest Cover Type using PyTorch")
    parser.add_argument("--data_set", type=str, required=True, help="Path to training csv")
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    train(args)
