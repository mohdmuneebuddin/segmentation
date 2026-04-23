import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.autoencmodel import AutoEncoder
from src.dataset import MVTecDataset
from src.config import *

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. Available GPUs: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    print("Loading Train Dataset..." )
    train_dataset = MVTecDataset("/content/drive/MyDrive/datasets/bottle/train/good")
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    
    print("Loading Test Dataset..." )
    test_dataset = MVTecDataset("/content/drive/MyDrive/datasets/bottle/test")
    test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, shuffle = False)
    
    model = AutoEncoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max = NUM_EPOCHS )
    print(f"Optimizer: AdamW (lr={LEARNING_RATE}),  weight_decay={WEIGHT_DECAY})")
    print(f"Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    print(f"Loss: MSELoss")
    print("Training Started...")
    train_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f" Epoch {epoch +1}/{NUM_EPOCHS}")
        for images, _ in train_bar:
            images = images.to(device)

            output = model(images)
            loss = criterion(output, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss}")

        #Calculating Threshold
    print("Computing threshold...")
    model.eval()
    train_errors = []
    with torch.no_grad():
        threshold_bar = tqdm(train_loader, desc=f" Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, _ in threshold_bar:
            images = images.to(device)
            recon = model(images)
            error = torch.mean((recon - images)**2, dim = [1,2,3])
            train_errors.extend(error.cpu().numpy())

    threshold = np.percentile(train_errors, 99)
    print(f"Threshold: {threshold}")
    model.eval()
    all_labels = []
    all_scores = []

    val_bar = tqdm(test_loader, desc=f" Epoch {epoch +1}/{NUM_EPOCHS} [Validation]")
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_bar):

            images, labels = images.to(device), labels.to(device)
            recon = model(images)
            error = torch.mean((images - recon) ** 2, dim = [1,2,3])
            
            all_scores.extend(error.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
                        error_map = (images - recon)**2
            heatmap = error_map.mean(dim=1)   # shape: (B, H, W)
            heatmap = heatmap[0].cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            img = images[0].cpu().permute(1, 2, 0).numpy()
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12,4))
            
            plt.subplot(1,3,1)
            plt.title("Original")
            plt.imshow(img)
            plt.axis("off")
            
            plt.subplot(1,3,2)
            plt.title("Reconstruction")
            recon_img = recon[0].cpu().permute(1,2,0).numpy()
            plt.imshow(recon_img)
            plt.axis("off")
            
            plt.subplot(1,3,3)
            plt.title("Anomaly Heatmap")
            plt.imshow(img)
            plt.imshow(heatmap, cmap='jet', alpha=0.5)  # overlay
            plt.axis("off")
            
            plt.show()
            break
    preds = [1 if s > threshold else 0 for s in all_scores]

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_scores)


    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc)



if __name__ == "__main__":
    main()
