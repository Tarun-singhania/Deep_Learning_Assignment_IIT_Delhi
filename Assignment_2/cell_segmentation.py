import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 320
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 120

COLOR_MAP = {
    (0,0,0):0,
    (255,255,0):1,
    (255,0,0):2,
    (0,255,0):3,
    (0,0,255):4
}

LABEL_TO_COLOR = {v:k for k,v in COLOR_MAP.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
def load_image_mask(image_path, mask_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    image = image/255.0

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)

    class_mask = np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.uint8)

    for rgb,label in COLOR_MAP.items():
        matches = np.all(mask==rgb,axis=-1)
        class_mask[matches] = label

    return image.astype(np.float32), class_mask

# Data Augmentation
class SegmentationDataset(Dataset):

    def __init__(self, image_list, mask_list=None, augment=False):
        self.image_list = image_list
        self.mask_list = mask_list
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path = self.image_list[idx]

        if self.mask_list is None:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
            image = np.transpose(image, (2,0,1))

            image = torch.tensor(image, dtype=torch.float32)

            return image, img_path

        mask_path = self.mask_list[idx]

        image, mask = load_image_mask(img_path, mask_path)

        if self.augment:

            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if np.random.rand() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

            # rotation
            k = np.random.randint(0,4)
            image = np.rot90(image,k)
            mask = np.rot90(mask,k)

            # brightness
            if np.random.rand() > 0.5:
                image = image * np.random.uniform(0.8,1.2)
                image = np.clip(image,0,1)

            # gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0,0.05,image.shape)
                image = np.clip(image+noise,0,1)

            # zoom
            if np.random.rand() > 0.5:

                scale = np.random.uniform(0.9,1.1)
                new_size = int(IMG_SIZE*scale)

                image = cv2.resize(image,(new_size,new_size))
                mask = cv2.resize(mask,(new_size,new_size),interpolation=cv2.INTER_NEAREST)

                image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
                mask = cv2.resize(mask,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_NEAREST)

            # blur
            if np.random.rand() > 0.5:
                image = cv2.GaussianBlur(image,(5,5),0)

        image = image.astype(np.float32)
        mask = mask.astype(np.uint8)

        image = np.transpose(image,(2,0,1))

        image = torch.tensor(image,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.long)

        return image, mask

# Model Creation
class UNet(nn.Module):

    def __init__(self,num_classes):

        super().__init__()

        self.eb1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2)

        self.eb2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.eb3 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.eb4 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.bottle = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec1 = nn.Conv2d(512,256,3,padding=1)

        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec2 = nn.Conv2d(256,128,3,padding=1)

        self.up3 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec3 = nn.Conv2d(128,64,3,padding=1)

        self.up4 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec4 = nn.Conv2d(64,32,3,padding=1)

        self.out = nn.Conv2d(32,num_classes,1)


    def forward(self,x):

        e1 = self.eb1(x)
        p1 = self.pool(e1)

        e2 = self.eb2(p1)
        p2 = self.pool(e2)

        e3 = self.eb3(p2)
        p3 = self.pool(e3)

        e4 = self.eb4(p3)
        p4 = self.pool(e4)

        b = self.bottle(p4)

        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)

        out = self.out(d4)

        return out

# Loss Calculation
def compute_class_weights(mask_list):

    class_counts = np.zeros(NUM_CLASSES)

    for mask_path in mask_list:

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        for rgb, label in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[label] += np.sum(matches)

    total_pixels = np.sum(class_counts)

    class_weights = total_pixels / (NUM_CLASSES * class_counts + 1e-6)

    return class_weights

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):

    y_true_onehot = F.one_hot(y_true.long(), NUM_CLASSES)
    y_true_onehot = y_true_onehot.permute(0,3,1,2).float()

    y_pred = F.softmax(y_pred, dim=1)

    intersection = torch.sum(y_true_onehot * y_pred, dim=(2,3))
    union = torch.sum(y_true_onehot + y_pred, dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - torch.mean(dice[:,1:])


def focal_loss(y_true, y_pred, gamma=2.0):

    y_true = y_true.long()

    y_true_onehot = F.one_hot(y_true, NUM_CLASSES)
    y_true_onehot = y_true_onehot.permute(0,3,1,2).float()

    y_pred_soft = F.softmax(y_pred, dim=1)

    ce = -torch.sum(y_true_onehot * torch.log(y_pred_soft + 1e-6), dim=1)

    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce

    weights = torch.sum(
        class_weights_tensor.view(1,-1,1,1) * y_true_onehot,
        dim=1
    )
    return focal * weights

def combined_loss(y_pred, y_true):

    y_true_onehot = F.one_hot(y_true.long(), NUM_CLASSES)
    y_true_onehot = y_true_onehot.permute(0,3,1,2).float()

    y_pred_soft = F.softmax(y_pred, dim=1)

    ce = -torch.sum(y_true_onehot * torch.log(y_pred_soft + 1e-6), dim=1).mean()

    focal = focal_loss(y_true, y_pred).mean()

    dice = dice_loss(y_true, y_pred)

    return ce + focal + 2 * dice

# Compute mIoU and Dice score:
def compute_metrics(model, dataloader):

    model.eval()

    total_iou = np.zeros(NUM_CLASSES)
    total_dice = np.zeros(NUM_CLASSES)
    total_count = np.zeros(NUM_CLASSES)

    with torch.no_grad():

        for images, true_masks in dataloader:

            images = images.to(device)
            true_masks = true_masks.to(device)

            preds = model(images)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            true_masks = true_masks.cpu().numpy()

            for cls in range(NUM_CLASSES):

                pred_cls = (preds == cls)
                true_cls = (true_masks == cls)

                intersection = np.sum(pred_cls & true_cls)
                union = np.sum(pred_cls | true_cls)

                if union == 0:
                    continue

                total_iou[cls] += intersection / union

                total_dice[cls] += (2 * intersection) / (
                    np.sum(pred_cls) + np.sum(true_cls)
                )

                total_count[cls] += 1

    mean_iou = total_iou / np.maximum(total_count,1)
    mean_dice = total_dice / np.maximum(total_count,1)

    return mean_iou, mean_dice

# Training loop
def train(dataset_path, model_save_path):

    train_img = sorted(glob(os.path.join(dataset_path,"train_images","*.png")))
    train_mask = sorted(glob(os.path.join(dataset_path,"train_masks","*.png")))

    # Compute class weights
    class_weights = compute_class_weights(train_mask)

    global class_weights_tensor
    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    tr_img,val_img,tr_mask,val_mask = train_test_split(
        train_img,train_mask,test_size=0.2,random_state=2
    )

    train_dataset = SegmentationDataset(tr_img,tr_mask,augment=True)
    val_dataset = SegmentationDataset(val_img,val_mask,augment=False)

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE)

    model = UNet(NUM_CLASSES).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    best_loss = 1e10

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = combined_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            train_correct += (preds == masks).sum().item()
            train_total += masks.numel()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        print(f"Epoch {epoch+1} Train Loss {train_loss:.4f} Train Acc {train_acc:.4f}")

        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = combined_loss(outputs, masks)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                val_correct += (preds == masks).sum().item()
                val_total += masks.numel()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f}")

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(model.state_dict(),model_save_path)

            print("Model saved")

    print("\nEvaluating model...\n")

    iou_scores, dice_scores = compute_metrics(model, val_loader)

    print("IoU per class:", iou_scores)
    print("Dice per class:", dice_scores)

    mean_iou = np.mean(iou_scores[1:])
    mean_dice = np.mean(dice_scores[1:])

    print("Mean IoU (excluding background):", mean_iou)
    print("Mean Dice (excluding background):", mean_dice)

    final_score = (mean_iou + mean_dice) / 2

    print("Final Score:", final_score)

# Inference loop
def label_to_color(mask):

    color_mask=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)

    for label,color in LABEL_TO_COLOR.items():
        color_mask[mask==label]=color

    return color_mask


def inference(dataset_path,model_path,output_path):

    os.makedirs(output_path,exist_ok=True)

    test_images = sorted(glob(os.path.join(dataset_path,"test_images","*.png")))

    model = UNet(NUM_CLASSES).to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))

    model.eval()

    for img_path in test_images:

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))/255.0

        image = np.transpose(image,(2,0,1))

        image = torch.tensor(image,dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():

            pred = model(image)

        pred = torch.argmax(pred,dim=1).squeeze().cpu().numpy()

        color_mask = label_to_color(pred)

        save_path = os.path.join(output_path,os.path.basename(img_path))

        cv2.imwrite(save_path,cv2.cvtColor(color_mask,cv2.COLOR_RGB2BGR))

# Main
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",required=True,choices=["train","inference"])

    parser.add_argument("--dataset_path",required=True)

    parser.add_argument("--model_save_path")

    parser.add_argument("--model_path")

    parser.add_argument("--output_path")

    args = parser.parse_args()

    if args.mode=="train":

        train(args.dataset_path,args.model_save_path)

    elif args.mode=="inference":

        inference(args.dataset_path,args.model_path,args.output_path)


if __name__=="__main__":
    main()