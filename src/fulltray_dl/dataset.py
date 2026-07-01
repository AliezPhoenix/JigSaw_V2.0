"""满盘 cell 数据集与 DataLoader。"""

import os

import cv2 as cv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


class CellDataset(Dataset):
    """train/val 目录下 has_product、empty 二分类数据集。"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        has_product_dir = os.path.join(data_dir, "has_product")
        if os.path.isdir(has_product_dir):
            for filename in os.listdir(has_product_dir):
                if _is_image_file(filename):
                    self.images.append(os.path.join(has_product_dir, filename))
                    self.labels.append(1)

        empty_dir = os.path.join(data_dir, "empty")
        if os.path.isdir(empty_dir):
            for filename in os.listdir(empty_dir):
                if _is_image_file(filename):
                    self.images.append(os.path.join(empty_dir, filename))
                    self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(is_train=True, input_size=150):
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def count_class_images(data_dir: str) -> dict:
    counts = {}
    for category in ("has_product", "empty"):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            counts[category] = 0
            continue
        counts[category] = sum(
            1 for name in os.listdir(cat_dir) if _is_image_file(name)
        )
    return counts


def get_dataloaders(
    train_dir,
    val_dir,
    batch_size=16,
    input_size=150,
    num_workers=0,
):
    train_dataset = CellDataset(
        train_dir,
        transform=get_transforms(is_train=True, input_size=input_size),
    )
    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空: {train_dir}")

    val_dataset = CellDataset(
        val_dir,
        transform=get_transforms(is_train=False, input_size=input_size),
    )
    if len(val_dataset) == 0:
        raise ValueError(f"验证集为空: {val_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader
