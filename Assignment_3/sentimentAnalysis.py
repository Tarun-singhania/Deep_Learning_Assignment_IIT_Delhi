# Import all libaries
import argparse
import numpy as np
import pandas as pd
import re
import os

import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


#  Data Preprocessing
# (i). Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# (ii). Build New Tokenizer
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for word in counter:
        word2idx[word] = len(word2idx)

    return word2idx

# (iii). Convert Text to Sequences
def texts_to_sequences(texts, word2idx):
    sequences = []
    for text in texts:
        seq = [word2idx.get(word, word2idx["<OOV>"]) for word in text.split()]
        sequences.append(seq)
    return sequences

# (iv). Apply Padding
def pad_sequences_manual(sequences, maxlen):
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq = seq + [0] * (maxlen - len(seq))
        else:
            seq = seq[:maxlen]
        padded.append(seq)
    return np.array(padded)


# Create Scratch Model(Part A)
class RNN_Scratch:
    def __init__(self, input_size, hidden_size, output_size):

        self.Wxh = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def forward_step(self, x, h_prev):
        return np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)

    def forward(self, inputs, embedding_matrix):
        h_prev = np.zeros((1, self.Whh.shape[0]))
        hs, xs = [], []

        for t in range(len(inputs)):
            x = embedding_matrix[inputs[t]].reshape(1, -1)
            xs.append(x)
            h_prev = self.forward_step(x, h_prev)
            hs.append(h_prev)

        h_mean = np.mean(hs, axis=0)
        logits = h_mean @ self.Why + self.by
        return logits, hs, xs

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss_fn(self, logits, target, class_weights):
        probs = self.softmax(logits)

        # Label smoothing
        num_classes = probs.shape[1]
        smooth = 0.1

        target_dist = np.full((1, num_classes), smooth / (num_classes - 1))
        target_dist[0, target] = 1 - smooth

        loss = -np.sum(target_dist * np.log(probs + 1e-9))

        return loss * class_weights[target]

    def compute_output_gradients(self, logits, target):
        probs = self.softmax(logits)
        probs[0, target] -= 1
        return probs

    def bptt(self, dh, hs, xs):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(len(hs))):
            dh_raw = (1 - hs[t] ** 2) * dh
            dWxh += xs[t].T @ dh_raw
            dWhh += (hs[t-1] if t > 0 else np.zeros_like(hs[0])).T @ dh_raw
            dbh += dh_raw
            dh = dh_raw @ self.Whh.T

        return dWxh, dWhh, dbh

    def update_params(self, grads, lr):
        for dparam in grads:
            np.clip(dparam, -5, 5, out=dparam)

        self.Wxh -= lr * grads[0]
        self.Whh -= lr * grads[1]
        self.Why -= lr * grads[2]
        self.bh  -= lr * grads[3]
        self.by  -= lr * grads[4]

    def backward(self, logits, hs, xs, target, class_weights):
        dy = self.compute_output_gradients(logits, target)

        h_mean = np.mean(hs, axis=0)

        # Dropout (during training only)
        drop_prob = 0.5
        mask = (np.random.rand(*h_mean.shape) > drop_prob).astype(float)
        h_mean = h_mean * mask / (1 - drop_prob)
        dWhy = h_mean.T @ dy
        dby = dy

        dh = dy @ self.Why.T / len(hs)
        dWxh, dWhh, dbh = self.bptt(dh, hs, xs)

        return dWxh, dWhh, dWhy, dbh, dby

# Create RNN Model by using nn Module (Part B)
class Torch_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True
        )

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        h = torch.mean(out, dim=1)
        h = self.dropout(h)
        return self.fc(h)

# Training(Part A) 
def train_scratch(args):

    # Load Data
    df = pd.read_csv(args.dataset_path)
    df['clean_text'] = df['text'].apply(clean_text)

    label_map = {"negative":0,"positive":1,"neutral":2}
    df["label"] = df['sentiment'].map(label_map)

    # Apply new Tokenizer
    word2idx = build_vocab(df['clean_text'])

    sequences = texts_to_sequences(df['clean_text'], word2idx)
    MAX_LEN = max(len(s) for s in sequences)
    sequences = pad_sequences_manual(sequences, MAX_LEN)

    X = sequences
    y = df['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use GLOVE for embedding
    embedding_index = {}
    glove_path = "glove.twitter.27B.100d.txt"

    if not os.path.exists(glove_path):
        glove_path = os.path.join(args.dataset_path, "glove.twitter.27B.100d.txt")

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

    embedding_dim = 100

    # Changed (tokenizer → word2idx)
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word2idx.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]

    #  Model Creation 
    model = RNN_Scratch(100, 256, 3)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {i: w for i, w in zip(classes, weights)}

    # Track the results
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_f1 = -1
    patience = 5
    patience_counter = 0

    epochs = 30
    batch_size = 8
    lr = 0.0007

    # Training Loop
    for epoch in range(epochs):

        total_loss = 0
        preds_train, targets_train = [], []

        # Shuffle the data
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Batch Train
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            batch_loss = 0

            dWxh_sum = np.zeros_like(model.Wxh)
            dWhh_sum = np.zeros_like(model.Whh)
            dWhy_sum = np.zeros_like(model.Why)
            dbh_sum  = np.zeros_like(model.bh)
            dby_sum  = np.zeros_like(model.by)

            for seq, target in zip(X_batch, y_batch):

                logits, hs, xs = model.forward(seq, embedding_matrix)
                loss = model.loss_fn(logits, target, class_weights)
                batch_loss += loss

                pred = np.argmax(logits)
                preds_train.append(pred)
                targets_train.append(target)

                dWxh, dWhh, dWhy, dbh, dby = model.backward(
                    logits, hs, xs, target, class_weights
                )

                dWxh_sum += dWxh
                dWhh_sum += dWhh
                dWhy_sum += dWhy
                dbh_sum  += dbh
                dby_sum  += dby

            # Average gradients
            bs = len(X_batch)
            dWxh_sum /= bs
            dWhh_sum /= bs
            dWhy_sum /= bs
            dbh_sum  /= bs
            dby_sum  /= bs

            model.update_params(
                (dWxh_sum, dWhh_sum, dWhy_sum, dbh_sum, dby_sum),
                lr
            )

            total_loss += batch_loss

        train_loss = total_loss / len(X_train)
        train_acc = accuracy_score(targets_train, preds_train)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation loop
        val_loss_total = 0
        val_preds, val_targets = [], []

        for i in range(0, len(X_val), batch_size):
            X_batch = X_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]

            for seq, target in zip(X_batch, y_batch):
                logits, _, _ = model.forward(seq, embedding_matrix)
                loss = model.loss_fn(logits, target, class_weights)

                val_loss_total += loss

                pred = np.argmax(logits)
                val_preds.append(pred)
                val_targets.append(target)

        val_loss = val_loss_total / len(X_val)
        val_acc = accuracy_score(val_targets, val_preds)

        macro_f1 = f1_score(val_targets, val_preds, average='macro')
        micro_f1 = f1_score(val_targets, val_preds, average='micro')
        avg_f1 = (macro_f1 + micro_f1) / 2

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        print(f"Macro F1={macro_f1:.4f}, Micro F1={micro_f1:.4f}, Avg F1={avg_f1:.4f}")

        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

        # Save Best Model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0

            torch.save({
                "Wxh": model.Wxh,
                "Whh": model.Whh,
                "Why": model.Why,
                "bh": model.bh,
                "by": model.by,
                "word2idx": word2idx,
                "max_len": MAX_LEN,
                "embedding_matrix": embedding_matrix
            }, args.model_save_path)

            print("Best model saved!")

        else:
            patience_counter += 1

        # Using Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Save Graphs
    os.makedirs("plots", exist_ok=True)

    # loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("plots/Task_1_rnn_scratch_loss.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(train_accs, label="Train Acc", marker='o')
    plt.plot(val_accs, label="Val Acc", marker='o')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("plots/Task_1_rnn_scratch_accuracy.png")
    plt.close()

# Train (Part B) 
def train_torch(args):

    df = pd.read_csv(args.dataset_path)
    df['clean_text'] = df['text'].apply(clean_text)

    label_map = {"negative":0,"positive":1,"neutral":2}
    df["label"] = df['sentiment'].map(label_map)

    # Usinn Tokenizer
    word2idx = build_vocab(df['clean_text'])
    sequences = texts_to_sequences(df['clean_text'], word2idx)

    MAX_LEN = max(len(s) for s in sequences)
    X = pad_sequences_manual(sequences, MAX_LEN)
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # USing Glove for Embedding
    embedding_index = {}
    glove_path = "glove.twitter.27B.100d.txt"

    if not os.path.exists(glove_path):
        glove_path = os.path.join(args.dataset_path, "glove.twitter.27B.100d.txt")

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

    embedding_dim = 100
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word2idx.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]

    # Model Creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Torch_RNN(100, 256, 3, embedding_matrix).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0007,weight_decay=1e-4)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    def loss_fn(logits, targets):
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)(logits, targets)

    # Training Loop
    best_f1 = -1

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(30):
        model.train()

        X_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = loss_fn(logits, y_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        acc = accuracy_score(y_train, preds)

        train_losses.append(loss.item())
        train_accs.append(acc)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.long).to(device))
            val_loss = loss_fn(val_logits, torch.tensor(y_val, dtype=torch.long).to(device)).item()

            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        macro = f1_score(y_val, val_preds, average='macro')
        micro = f1_score(y_val, val_preds, average='micro')
        avg_f1 = (macro + micro) / 2

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")
        print(f"Train Acc={acc:.4f}, Val Acc={val_acc:.4f}")
        print(f"Macro F1={macro:.4f}, Micro F1={micro:.4f}, Avg F1={avg_f1:.4f}")

        # Save best Model
        if avg_f1 > best_f1:
            best_f1 = avg_f1

            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

            torch.save(model.state_dict(), args.model_save_path)

            torch.save({
                "word2idx": word2idx,
                "max_len": MAX_LEN,
                "embedding_matrix": embedding_matrix
            }, args.model_save_path + "_meta.pt")

            print("Model Saved")

    # Save Plots
    os.makedirs("plots", exist_ok=True)

    # Loss graph
    plt.figure()
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("plots/rnn_loss.png")
    plt.close()

    # Accuracy graph
    plt.figure()
    plt.plot(train_accs, label="Train Acc", marker='o')
    plt.plot(val_accs, label="Val Acc", marker='o')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("plots/rnn_accuracy.png")
    plt.close()

    print("Graphs saved in 'plots/' folder")

# Train BILSTM Model (Part C)
def train_bilstm(args):

    df = pd.read_csv(args.dataset_path)
    df['clean_text'] = df['text'].apply(clean_text)

    label_map = {"negative":0,"positive":1,"neutral":2}
    df["label"] = df['sentiment'].map(label_map)

    # Using Tokenizer
    word2idx = build_vocab(df['clean_text'])
    sequences = texts_to_sequences(df['clean_text'], word2idx)

    MAX_LEN = max(len(s) for s in sequences)
    X = pad_sequences_manual(sequences, MAX_LEN)
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use GLOVE for Embedding
    embedding_index = {}
    glove_path = "glove.twitter.27B.100d.txt"

    if not os.path.exists(glove_path):
        glove_path = os.path.join(args.dataset_path, "glove.twitter.27B.100d.txt")

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embedding_index[values[0]] = np.asarray(values[1:], dtype='float32')

    embedding_dim = 100
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word2idx.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]

    # Model Creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float32),
        freeze=True
    ).to(device)

    lstm = nn.LSTM(
        input_size=100,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.5
    ).to(device)

    dropout = nn.Dropout(0.6)

    fc = nn.Linear(256, 3).to(device)

    # Include embedding params
    params = list(embedding.parameters()) + list(lstm.parameters()) + list(fc.parameters())

    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    def loss_fn(logits, targets):
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)(logits, targets)

    # Training Loop
    best_f1 = -1
    batch_size = 8

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(30):

        embedding.train()
        lstm.train()
        fc.train()

        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0
        preds_train, targets_train = [], []

        # Use Batch Training
        for i in range(0, len(X_train), batch_size):

            X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.long).to(device)
            y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.long).to(device)

            optimizer.zero_grad()

            emb = embedding(X_batch)
            out, _ = lstm(emb)

            avg_pool = torch.mean(out, dim=1)
            max_pool, _ = torch.max(out, dim=1)

            h = torch.cat([avg_pool, max_pool], dim=1)
            h = dropout(h)

            logits = fc(h)

            loss = loss_fn(logits, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)

            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_train.extend(preds)
            targets_train.extend(y_batch.cpu().numpy())

        train_loss = total_loss / len(X_train)
        train_acc = accuracy_score(targets_train, preds_train)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Loop
        embedding.eval()
        lstm.eval()
        fc.eval()

        val_preds = []
        val_loss_total = 0

        patience = 3
        counter = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):

                X_batch = torch.tensor(X_val[i:i+batch_size], dtype=torch.long).to(device)
                y_batch = torch.tensor(y_val[i:i+batch_size], dtype=torch.long).to(device)

                emb = embedding(X_batch)
                out, _ = lstm(emb)

                avg_pool = torch.mean(out, dim=1)
                max_pool, _ = torch.max(out, dim=1)

                h = torch.cat([avg_pool, max_pool], dim=1)
                logits = fc(h)

                loss = loss_fn(logits, y_batch)
                val_loss_total += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_loss = val_loss_total / len(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        macro = f1_score(y_val, val_preds, average='macro')
        micro = f1_score(y_val, val_preds, average='micro')
        avg_f1 = (macro + micro) / 2

        scheduler.step(avg_f1)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        print(f"Macro F1={macro:.4f}, Micro F1={micro:.4f}, Avg F1={avg_f1:.4f}")

        # Save Model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            counter = 0

            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

            torch.save({
                "lstm": lstm.state_dict(),
                "fc": fc.state_dict(),
                "embedding": embedding.state_dict(),
                "word2idx": word2idx,
                "max_len": MAX_LEN,
                "embedding_matrix": embedding_matrix
            }, args.model_save_path)

            print("BiLSTM Model Saved")
        else :
            counter +=1

        if counter >= patience:
            print("Early stopping triggered")
            break

    # Save Plots
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("plots/bilstm_loss.png")
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="Train Acc", marker='o')
    plt.plot(val_accs, label="Val Acc", marker='o')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("plots/bilstm_accuracy.png")
    plt.close()

    print("Graphs saved in 'plots/' folder")

# Create Inference Function for Scratch RNN (Part A)
def inference_scratch(args):
    checkpoint = torch.load(args.model_path, weights_only=False)

    model = RNN_Scratch(100, 256, 3)
    model.Wxh = checkpoint["Wxh"]
    model.Whh = checkpoint["Whh"]
    model.Why = checkpoint["Why"]
    model.bh = checkpoint["bh"]
    model.by = checkpoint["by"]

    word2idx = checkpoint["word2idx"]
    MAX_LEN = checkpoint["max_len"]
    embedding_matrix = checkpoint["embedding_matrix"]

    test = pd.read_csv(args.dataset_path)
    test['clean_text'] = test['text'].apply(clean_text)

    sequences = texts_to_sequences(test['clean_text'], word2idx)
    sequences = pad_sequences_manual(sequences, MAX_LEN)

    preds = []
    for seq in sequences:
        valid_len = np.count_nonzero(seq)
        seq = seq[:valid_len]
        if len(seq) == 0:
            preds.append(0)
            continue
        out, _, _ = model.forward(seq, embedding_matrix)
        probs = model.softmax(out)
        preds.append(np.argmax(probs))

    label_map_rev = {0:"negative",1:"positive",2:"neutral"}
    output = pd.DataFrame({
        "tweet_id": test["tweet_id"],
        "sentiment": [label_map_rev[p] for p in preds]
    })
    output.to_csv(args.output_path, index=False)

    # Evaluation if ground truth exists
    if "sentiment" in test.columns:
        label_map = {"negative":0,"positive":1,"neutral":2}
        y_true = test["sentiment"].map(label_map).values
        macro = f1_score(y_true, preds, average="macro")
        micro = f1_score(y_true, preds, average="micro")
        avg_f1 = (macro + micro) / 2
        acc = accuracy_score(y_true, preds)
        print(f"Accuracy={acc:.4f}, Macro F1={macro:.4f}, Micro F1={micro:.4f}, Avg F1={avg_f1:.4f}")

# Create Inference Function for RNN (Part B)
def inference_torch(args):

    meta = torch.load(args.model_path + "_meta.pt", weights_only=False)
    word2idx = meta["word2idx"]
    MAX_LEN = meta["max_len"]
    embedding_matrix = meta["embedding_matrix"]

    model = Torch_RNN(100, 256, 3, embedding_matrix)
    model.load_state_dict(torch.load(args.model_path, weights_only=False))
    model.eval()

    test = pd.read_csv(args.dataset_path)
    test['clean_text'] = test['text'].apply(clean_text)

    sequences = texts_to_sequences(test['clean_text'], word2idx)
    sequences = pad_sequences_manual(sequences, MAX_LEN)

    preds = []
    for seq in sequences:
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        out = model(seq)
        probs = torch.softmax(out, dim=1)
        preds.append(torch.argmax(probs).item())

    label_map = {0:"negative",1:"positive",2:"neutral"}

    output = pd.DataFrame({
        "tweet_id": test["tweet_id"],
        "sentiment": [label_map[p] for p in preds]
    })

    output.to_csv(args.output_path, index=False)

    # Evaluation if ground truth exists
    if "sentiment" in test.columns:

        label_map_eval = {"negative":0,"positive":1,"neutral":2}
        y_true = test["sentiment"].map(label_map_eval).values

        macro_f1 = f1_score(y_true, preds, average='macro')
        micro_f1 = f1_score(y_true, preds, average='micro')
        avg_f1 = (macro_f1 + micro_f1) / 2
        acc = accuracy_score(y_true, preds)

        print("\nEvaluation Results")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro F1     : {macro_f1:.4f}")
        print(f"Micro F1     : {micro_f1:.4f}")
        print(f"Average F1   : {avg_f1:.4f}")

# Create Inference Function for BILSTM (Part C)
def inference_bilstm(args):

    checkpoint = torch.load(args.model_path, weights_only=False)

    word2idx = checkpoint["word2idx"]
    MAX_LEN = checkpoint["max_len"]
    embedding_matrix = checkpoint["embedding_matrix"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float32),
        freeze=True
    ).to(device)

    lstm = nn.LSTM(
        input_size=100,
        hidden_size=64,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.3
    ).to(device)

    fc = nn.Linear(256, 3).to(device)

    lstm.load_state_dict(checkpoint["lstm"])
    fc.load_state_dict(checkpoint["fc"])

    test = pd.read_csv(args.dataset_path)
    test['clean_text'] = test['text'].apply(clean_text)

    sequences = texts_to_sequences(test['clean_text'], word2idx)
    sequences = pad_sequences_manual(sequences, MAX_LEN)

    preds = []

    for seq in sequences:
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = embedding(seq)
            out, _ = lstm(emb)

            avg_pool = torch.mean(out, dim=1)
            max_pool, _ = torch.max(out, dim=1)
            h = torch.cat([avg_pool, max_pool], dim=1)

            logits = fc(h)

        preds.append(torch.argmax(logits).item())

    label_map = {0:"negative",1:"positive",2:"neutral"}

    output = pd.DataFrame({
        "tweet_id": test["tweet_id"],
        "sentiment": [label_map[p] for p in preds]
    })

    output.to_csv(args.output_path, index=False)

    # Evaluation if ground truth exists
    if "sentiment" in test.columns:
        label_map_eval = {"negative":0,"positive":1,"neutral":2}
        y_true = test["sentiment"].map(label_map_eval).values

        macro_f1 = f1_score(y_true, preds, average='macro')
        micro_f1 = f1_score(y_true, preds, average='micro')
        avg_f1 = (macro_f1 + micro_f1) / 2
        acc = accuracy_score(y_true, preds)

        print("\nEvaluation Results")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro F1     : {macro_f1:.4f}")
        print(f"Micro F1     : {micro_f1:.4f}")
        print(f"Average F1   : {avg_f1:.4f}")

# Main Function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True)
    parser.add_argument("--model", required=True) 
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_save_path")
    parser.add_argument("--model_path")
    parser.add_argument("--output_path")

    args = parser.parse_args()

    # Training Part
    if args.mode == "train":

        if args.model == "rnn_scratch":
            train_scratch(args)

        elif args.model == "rnn":
            train_torch(args)

        elif args.model == "rnn_comp":  
            train_bilstm(args)

        else:
            raise ValueError("Invalid model type")

    # Inference Part
    elif args.mode == "inference":

        if args.model == "rnn_scratch":
            inference_scratch(args)

        elif args.model == "rnn":
            inference_torch(args)

        elif args.model == "rnn_comp":  
            inference_bilstm(args)

        else:
            raise ValueError("Invalid model type")