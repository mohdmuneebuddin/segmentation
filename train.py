import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.model import UNetAutoencoder   
from src.dataset import MVTecDataset
from src.config import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------ DATA ------------------
    print("Loading Train Dataset...")
    train_dataset = MVTecDataset("/content/drive/MyDrive/datasets/bottle/train/good")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading Test Dataset...")
    test_dataset = MVTecDataset("/content/drive/MyDrive/datasets/bottle/test")
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Debug labels
    labels = [label for _, label in test_dataset.images]
    print("Unique test labels:", set(labels))
    print("Count of 0:", labels.count(0))
    print("Count of 1:", labels.count(1))

    # ------------------ MODEL ------------------
    model = UNetAutoencoder(in_channels=3, out_channels=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"Optimizer: AdamW (lr={LEARNING_RATE})")
    print(f"Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    print("Loss: L1Loss")

    # ------------------ TRAINING ------------------
    print("Training Started...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, _ in train_bar:   
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.6f}")

    # ------------------ THRESHOLD ------------------
    print("\nComputing threshold...")
    model.eval()
    train_errors = []

    with torch.no_grad():
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            recon = model(images)

            error = torch.mean((images - recon) ** 2, dim=[1,2,3])
            train_errors.extend(error.cpu().numpy())

    threshold = np.percentile(train_errors, 95)
    print("Threshold:", threshold)

    # ------------------ VALIDATION ------------------
    model.eval()
    all_scores = []
    all_labels = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            recon = model(images)
            error = torch.mean((images - recon) ** 2, dim=[1,2,3])

            all_scores.extend(error.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------------ DEBUG ------------------
    print("\nDEBUG:")
    print("Min score:", np.min(all_scores))
    print("Max score:", np.max(all_scores))
    print("Threshold:", threshold)

    # ------------------ METRICS ------------------
    preds = [1 if s > threshold else 0 for s in all_scores]

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = float("nan")

    print("\nRESULTS:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc)


if __name__ == "__main__":
    main()
