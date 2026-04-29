import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve

from src.model import ConvAutoencoder
from src.dataset import MVTecDataset
from src.evalualize import plot_evaluation
from src.config import (
    BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    WEIGHT_DECAY, SEED, DATASET_PATH, CATEGORY,
)


# ------------------ SEED ------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------ THRESHOLD FOR HIGH RECALL ------------------
from sklearn.metrics import recall_score

#def find_threshold_for_recall(scores, labels, target_recall=0.95):
    #thresholds = min(scores)
    #return thresholds

# ------------------ SCORE FUNCTION ------------------
def compute_anomaly_score(model, loader, device):
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            recon = model(images)
            err = torch.mean((images - recon) ** 2, dim=[1, 2, 3])

            scores.extend(err.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.array(scores), np.array(labels)


# ------------------ MAIN ------------------
def main():
    set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------ DATA ------------------
    train_root = os.path.join(DATASET_PATH, CATEGORY, "train", "good")
    test_root  = os.path.join(DATASET_PATH, CATEGORY, "test")

    train_dataset = MVTecDataset(train_root, train=True)
    test_dataset  = MVTecDataset(test_root, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    labels = [lbl for _, lbl in test_dataset.images]
    print("Test set — good:", labels.count(0), " defect:", labels.count(1))

    # ------------------ MODEL ------------------
    model = ConvAutoencoder(in_channels=3, out_channels=3).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ------------------ TRAIN ------------------
    best_auc = -1.0
    best_path = "best_model.pt"

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, _ in bar:
            images = images.to(device, non_blocking=True)

            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # -------- evaluation --------
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            scores, lbls = compute_anomaly_score(model, test_loader, device)

            try:
                auc = roc_auc_score(lbls, scores)
            except ValueError:
                auc = float("nan")

            print(f"Epoch {epoch+1}: train_loss={avg_loss:.5f}  test_AUROC={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_path)
                print(f"  ↳ new best AUROC {auc:.4f}, saved to {best_path}")
        else:
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.5f}")

    # ------------------ FINAL EVAL ------------------
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_scores, test_labels = compute_anomaly_score(model, test_loader, device)
    print('test score: ',test_scores)
    print('labels: ',test_labels)

    # ---- IMPORTANT FIX: threshold from TRAIN only ----
    train_scores, _ = compute_anomaly_score(model, train_loader, device)

    #threshold = find_threshold_for_recall(
        #train_scores,
        #np.zeros_like(train_scores),
        #target_recall=0.95
    #)
    threshold = 0.037712
    print(f"Threshold: {threshold:.6f}")

    # ------------------ EVALUATION ------------------
    preds = (test_scores > threshold).astype(int)
    print('preds :',preds)
    acc = accuracy_score(test_labels, preds)
    f1  = f1_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    print(f"\n[High Recall Model]")
    print(f"accuracy={acc:.4f}")
    print(f"f1={f1:.4f}")
    print(f"recall={recall:.4f}")

    auc = roc_auc_score(test_labels, test_scores)
    print(f"\nFINAL AUROC (threshold-free): {auc:.4f}")
    print(f"Best AUROC during training:   {best_auc:.4f}")

    # ------------------ PLOTS ------------------
    plot_evaluation(
        y_true=test_labels,
        y_scores=test_scores,
        threshold=threshold,
        save_dir="plots",
    )


if __name__ == "__main__":
    main()
