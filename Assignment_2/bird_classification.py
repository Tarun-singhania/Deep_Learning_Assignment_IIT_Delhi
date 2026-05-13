# Import all required libaries
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image

# Define DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True if device.type == "cuda" else False


# MODEL Definition
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)


class ResNetCustom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            ResidualBlock(out_c, out_c, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# Use the DATASET
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_dir = image_dir
        self.image_names = dataframe['image_name'].values
        self.labels = dataframe['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


class TestDataset(Dataset):
    def __init__(self, test_dir, transform):
        self.test_dir = test_dir
        self.image_names = os.listdir(test_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_name


# TRAINING Period
def train_model(dataset_path, model_save_path):

    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 40
    LR = 1e-3

    csv_path = os.path.join(dataset_path, "train.csv")
    image_dir = dataset_path

    df = pd.read_csv(csv_path)

    label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_to_index)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )

    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Data Augmentation
    data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.1 * 180),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ColorJitter(contrast=0.1)
    ]) 

    train_transform = transforms.Compose([
        data_augmentation,
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        CustomDataset(train_df, image_dir, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        CustomDataset(val_df, image_dir, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ResNetCustom(len(label_to_index)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    best_score = -1

    for epoch in range(EPOCHS):

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        final_score = (macro_f1 + micro_f1) / 2

        print(f"Epoch {epoch+1} | Final Score: {final_score:.4f}")

        if final_score > best_score:
            best_score = final_score
            torch.save(model.state_dict(), model_save_path)

    print("Training completed. Model saved.")


# INFERENCE Period
def inference(dataset_path, model_path, output_path):

    IMG_SIZE = 224

    csv_path = os.path.join(dataset_path, "train.csv")
    test_dir = dataset_path

    df = pd.read_csv(csv_path)
    label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
    index_to_label = {v: k for k, v in label_to_index.items()}

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    model = ResNetCustom(len(label_to_index))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        TestDataset(test_dir, transform),
        batch_size=64,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()
            for name, pred in zip(names, preds):
                predictions.append([name, index_to_label[pred]])

    submission = pd.DataFrame(predictions, columns=["image_name", "label"])
    submission.to_csv(output_path, index=False)

    print("Inference completed. Predictions saved.")


# MAIN Function
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "inference"])
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_save_path")
    parser.add_argument("--model_path")
    parser.add_argument("--output_path")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.dataset_path, args.model_save_path)

    elif args.mode == "inference":
        inference(args.dataset_path, args.model_path, args.output_path)