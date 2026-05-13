import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

def train(args):

    # 1. Load data
    df = pd.read_csv(args.data_set)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # 2. Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

    # 3. Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    X_train = np.clip(X_train, -5, 5)
    X_val   = np.clip(X_val, -5, 5)

    # 4. Model definition
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.002,
        batch_size=128,
        alpha=5e-4,
        max_iter=args.epochs,
        shuffle=True,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.2,
        verbose=False,
        random_state=42
    )

    # 5. Training
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_val_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    print("\nValidation Macro F1:", macro_f1)
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_val_pred))

    # 7. SAVE sklearn weights and biases
    best_model_parameters = {
        "W1": model.coefs_[0].copy(),
        "b1": model.intercepts_[0].copy(),
        "W2": model.coefs_[1].copy(),
        "b2": model.intercepts_[1].copy(),
        "W3": model.coefs_[2].copy(),
        "b3": model.intercepts_[2].copy(),
    }

    checkpoint = {
        "model_state_dict": best_model_parameters,
        "training_loss":model.loss_,
        "input_dim": X_train.shape[1],
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy()
    }

    with open("forest_cover_model_sklearn.pth", "wb") as f:
        pickle.dump(checkpoint, f)

    print("\nSaved checkpoint: forest_cover_model_slearn.pth")

    # 8. Plotting
    plt.plot(model.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("MLPClassifier Training Loss")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    train(args)