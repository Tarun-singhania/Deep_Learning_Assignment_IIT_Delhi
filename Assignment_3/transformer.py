# Imports all required libaries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
import os
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configuration
MAX_LEN = 64
GLOVE_DIM = 100
NUM_HEADS = 2
HEAD_DIM = 64
EMBED_DIM = HEAD_DIM * NUM_HEADS
BATCH_SIZE = 32
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build Vocab
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<PAD>":0, "<UNK>":1}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

# Load Glove
def load_glove(path, word2idx):
    embeddings = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            embeddings[word] = vector

    matrix = np.random.normal(0, 0.6, (len(word2idx), GLOVE_DIM))

    for word, idx in word2idx.items():
        if word in embeddings:
            matrix[idx] = embeddings[word]

    return matrix

# Encoding
def encode(text):
    tokens = text.lower().split()
    ids = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids

# Dataset
class StoryDataset(Dataset):
    def __init__(self, texts):
        self.data = [encode(t) for t in texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1])
        y = torch.tensor(seq[1:])
        return x, y

# Positional Encodding
def get_sinusoidal_embeddings(max_len, embed_dim):
    position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))

    embeddings = torch.zeros(max_len, embed_dim)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)

    return embeddings

# Model
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.W_q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_v = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_o = nn.Linear(EMBED_DIM, EMBED_DIM)

        for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight)
              if m.bias is not None:
                  nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1,2)
        K = K.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1,2)
        V = V.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(HEAD_DIM)

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)

        out = attn @ V
        out = out.transpose(1,2).contiguous().view(B, T, C)

        return self.W_o(out), attn


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out, attn = self.attn(x)
        return x + self.dropout(out), attn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True
        )

        self.proj = nn.Linear(GLOVE_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(p=0.1)

        # Sinusoidal positional embedding (fixed, non-trainable)
        pos_emb = get_sinusoidal_embeddings(MAX_LEN, EMBED_DIM)
        self.pos_embedding = nn.Embedding(MAX_LEN, EMBED_DIM)
        self.pos_embedding.weight = nn.Parameter(pos_emb, requires_grad=False)


        self.block = TransformerBlock()
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T).unsqueeze(0).to(x.device)

        x = self.embedding(x)
        x = self.proj(x)
        x = self.dropout(x)
        x = x + self.pos_embedding(pos)

        x, attn = self.block(x)

        logits = self.fc_out(x)
        return logits, attn

def beam_search(model, start_tokens, beam_size=8):
    model.eval()
    sequences = [(start_tokens, 0)]

    for _ in range(MAX_LEN - len(start_tokens)):
        candidates = []

        for seq, score in sequences:
            x = torch.tensor(seq[-MAX_LEN:]).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(x)
                probs = F.log_softmax(logits[0, -1], dim=-1)

            topk = torch.topk(probs, beam_size)

            for i in range(beam_size):
                token = topk.indices[i].item()
                prob = topk.values[i].item()
                candidates.append((seq + [token], score + prob))

        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

    return sequences[0][0]

def train_epoch():
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=0   # PAD token
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Main Function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_save_path")
    parser.add_argument("--model_path")
    parser.add_argument("--output_path")

    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    texts = df["story"].astype(str).tolist()

    word2idx = build_vocab(texts)
    idx2word = {i:w for w,i in word2idx.items()}

    # GloVe path
    glove_path = "glove.twitter.27B.100d.txt"
    if not os.path.exists(glove_path):
        glove_path = os.path.join(os.path.dirname(args.dataset_path), "glove.twitter.27B.100d.txt")

    embedding_matrix = load_glove(glove_path, word2idx)

    train_dataset = StoryDataset(texts)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerModel(len(word2idx), embedding_matrix).to(device)

    if args.mode == "train":

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

        # Store metrics
        train_losses = []
        train_ppls = []

        for epoch in range(EPOCHS):
            train_loss = train_epoch()

            train_ppl = torch.exp(torch.tensor(train_loss)).item()

            train_losses.append(train_loss)
            train_ppls.append(train_ppl)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train PPL={train_ppl:.2f}")

        # Save Model
        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), args.model_save_path)


        # Save Plot
        plt.figure(figsize=(10,5))

        # Loss
        plt.subplot(1,2,1)
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Perplexity
        plt.subplot(1,2,2)
        plt.plot(train_ppls, label="Train PPL")
        plt.title("Training Perplexity")
        plt.xlabel("Epoch")
        plt.ylabel("PPL")
        plt.legend()

        plt.tight_layout()
        plt.savefig("train_curves.png") 
        print("Training curves saved as train_curves.png")

    elif args.mode == "inference":

        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        outputs = []

        for text in texts:
            start = encode(text)
            generated = beam_search(model,start)

            sentence = " ".join([idx2word.get(token, "<UNK>") for token in generated])
            outputs.append(sentence)

        with open(args.output_path, "w") as f:
            for line in outputs:
                f.write(line + "\n")