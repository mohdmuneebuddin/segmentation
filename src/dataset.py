from src.config import *
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from src.config import *

class MVTecDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    path = os.path.join(root, file)
                    label = 0 if 'good' in root else 1
                    self.images.append((path, label))
        print("TOTAL IMAGES:", len(self.images))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        return img, label

