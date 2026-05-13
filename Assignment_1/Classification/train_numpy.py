import numpy as np
import pandas as pd
import argparse
import pickle
from model_numpy import ForestCoverNet

"""
REQUIRED OUTPUT:
Your script must save a model checkpoint named 'forest_cover_model_numpy.pth' containing:
- 'model_state_dict': dictionary of numpy arrays (weights/biases)
- 'input_dim': number of input features
- 'scaler_mean': mean values used for standardization
- 'scaler_scale': scale values used for standardization
"""

# Extra UTILS Functions :

#(i). Softmax Function
def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

#(ii).Cross_Entropy_Loss
def cross_entropy_loss(probs, y):
    y_idx = y - 1
    N = y.shape[0]
    return np.mean(-np.log(probs[np.arange(N), y_idx] + 1e-8))

#(iii). For splitting the datset because we want to use Early stopping and in Early Stopping,we want validation loss also 
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

#(iv). For Validation dataset,compute macro_f1 score
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

def macro_f1_from_model(model, X, y, batch_size=512):
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model.forward(X[i:i+batch_size])
        preds.extend(np.argmax(logits, axis=1) + 1)
    return macro_f1(y, np.array(preds))


def train(args):
    # TODO: Implement complete training pipeline
    # 1. Load data from args.data_set
    # 2. Preprocess (StandardScaler) - Store mean/scale for saving
    # 3. Initialize ForestCoverNet(input_dim=54)
    # 4. Implement manual Backpropagation and SGD
    # 5. Save the dictionary using pickle

    # 1. Load data from args.data_set
    df = pd.read_csv(args.data_set)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64) 

    input_dim = X.shape[1]
    num_classes = 7

    # Train / Validation split
    X_train, X_val, y_train, y_val = stratified_split(X, y)

    # 2. Preprocess (StandardScaler) - Store mean/scale for saving
    scaler_mean = np.mean(X_train, axis=0)
    scaler_scale = np.std(X_train, axis=0) + 1e-8

    X_train = (X_train - scaler_mean) / scaler_scale
    X_val   = (X_val   - scaler_mean) / scaler_scale

    X_train = np.clip(X_train, -5, 5)
    X_val   = np.clip(X_val, -5, 5)

    # 3. Initialize ForestCoverNet
    model = ForestCoverNet(input_dim=input_dim, num_classes=num_classes)

    # 4. Implement manual Backpropagation and SGD
    # (i). Hyperparameters initialization
    learning_rate = 2e-3
    batch_size = 128
    epochs = args.epochs
    l2_lambda = 5e-4  # For regularization

    #(iii). For Early Stopping
    best_f1_score = 0.0
    patience = 15
    wait = 0
    best_model_parameters = None

    # (iv). Training loop
    for epoch in range(epochs):

        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        train_losses = []
        epoch_loss = 0.0

        for j in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            y_idx = y_batch - 1
            N = X_batch.shape[0]

            # Forward Propagation
            logits = model.forward(X_batch)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss * N

            # Backward Propagation
            dlogits = probs  
            dlogits[np.arange(N), y_idx] -= 1 
            dlogits /= N

            # Layer 3
            dW3 = model.a2.T @ dlogits + l2_lambda * model.W3
            db3 = np.sum(dlogits, axis=0, keepdims=True)

            da2 = dlogits @ model.W3.T
            da2[model.z2 <= 0] = 0

            # Layer 2
            dW2 = model.a1.T @ da2 + l2_lambda * model.W2
            db2 = np.sum(da2, axis=0, keepdims=True)

            da1 = da2 @ model.W2.T
            da1[model.z1 <= 0] = 0

            # Layer 1
            dW1 = X_batch.T @ da1 + l2_lambda * model.W1
            db1 = np.sum(da1, axis=0, keepdims=True)

            # SGD Update
            model.W3 -= learning_rate * dW3
            model.b3 -= learning_rate * db3
            model.W2 -= learning_rate * dW2
            model.b2 -= learning_rate * db2
            model.W1 -= learning_rate * dW1
            model.b1 -= learning_rate * db1

        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        val_f1 = macro_f1_from_model(model, X_val, y_val)
        val_f1_score = 0.9 * best_f1_score + 0.1 * val_f1

        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            wait = 0
            best_model_parameters = {
                "W1": model.W1.copy(), "b1": model.b1.copy(),
                "W2": model.W2.copy(), "b2": model.b2.copy(),
                "W3": model.W3.copy(), "b3": model.b3.copy()
            }
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    # 5. Save checkpoint
    checkpoint = {
        "model_state_dict": best_model_parameters,
        "training_loss":train_losses,
        "input_dim": input_dim,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale
    }

    with open("forest_cover_model_numpy.pth", "wb") as f:
        pickle.dump(checkpoint, f)

    print("Model saved as forest_cover_model_numpy.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ForestCoverNet using NumPy")
    parser.add_argument("--data_set", type=str, required=True, help="Path to training csv")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    args = parser.parse_args()
    train(args)
