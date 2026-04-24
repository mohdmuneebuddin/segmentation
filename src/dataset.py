import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.config import IMG_SIZE

# ImageNet stats — standard choice even when not using a pretrained backbone,
# because it gives well-behaved zero-centered inputs.
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.images = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(root, file)
                    folder_name = os.path.basename(os.path.dirname(path))
                    label = 0 if folder_name == "good" else 1
                    self.images.append((path, label))

        print(f"[{'train' if train else 'test'}] total images:", len(self.images))

        if train:
            # Mild augmentation only — we must NOT invent defects.
            # Bottles are roughly rotationally symmetric around the vertical axis,
            # so horizontal flips are safe. Rotations/crops can also help for
            # categories without strong orientation. Tune per category.
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label
