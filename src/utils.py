"""
CropGuard AI — Preprocessing & Augmentation Utilities
======================================================
Provides:
  - Constants  : class names, ImageNet stats
  - Transforms : train / val-test pipelines
  - Dataset    : PlantVillageDataset, PotatoHealthyDataset
  - Helpers    : set_seed, laplacian_variance (blur filter), stratified split
  - Sampler    : build_weighted_sampler (handles class imbalance)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLASSES: List[str] = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
]

CLASS_DISPLAY: List[str] = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Pepper Bacterial Spot",
    "Pepper Healthy",
    "Corn Common Rust",
    "Corn Healthy",
]

CLASS_TO_DISPLAY = dict(zip(TARGET_CLASSES, CLASS_DISPLAY))
DISPLAY_TO_CLASS = dict(zip(CLASS_DISPLAY, TARGET_CLASSES))

NUM_CLASSES  = 10
IMG_SIZE     = 224
BATCH_SIZE   = 32
BLUR_THRESH  = 50.0          # Laplacian variance threshold

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python / NumPy / PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
#  Transform Pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    9-technique augmentation pipeline applied **only** to the training set.

    Techniques simulate real-world field imaging conditions:
      1. Resize + random crop
      2. Horizontal & vertical flip
      3. Random rotation ±30°
      4. Color jitter (brightness, contrast, saturation, hue)
      5. Random grayscale (p=0.05)
      6. Gaussian blur
      7. Random erasing / CutOut (p=0.20)
      8. Tensor conversion
      9. ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Minimal val/test pipeline (resize + normalise only).
    No augmentation — prevents data leakage.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_vae_transform(img_size: int = 64) -> transforms.Compose:
    """
    Transform pipeline for VAE training on Potato__healthy images.
    Outputs tensors in [-1, 1] (Tanh decoder range).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = IMAGENET_MEAN,
    std:  List[float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Reverse ImageNet normalisation for display.

    Args:
        tensor: (C, H, W) normalised image tensor.
    Returns:
        (H, W, C) NumPy array, clipped to [0, 1].
    """
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    return (tensor * std_t + mean_t).permute(1, 2, 0).numpy().clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_variance(img_path: str) -> float:
    """
    Detect blurry or corrupt images using Laplacian variance.

    A value below BLUR_THRESH (50.0) indicates the image is likely blurry.

    Args:
        img_path: Path to the image file.
    Returns:
        Laplacian variance (higher = sharper).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────────────────
#  Datasets
# ─────────────────────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    PlantVillage 10-class leaf disease dataset.

    Scans DATA_ROOT for the 10 TARGET_CLASSES subdirectories and optionally
    filters blurry images using Laplacian variance.

    Args:
        data_root:    Path to the PlantVillage *color* folder.
        transform:    Torchvision transform pipeline.
        blur_filter:  If True, exclude images with Laplacian variance < BLUR_THRESH.
        classes:      Override the default TARGET_CLASSES list.
    """

    def __init__(
        self,
        data_root: str | Path,
        transform=None,
        blur_filter: bool = True,
        classes: List[str] = TARGET_CLASSES,
    ) -> None:
        self.data_root  = Path(data_root)
        self.transform  = transform
        self.classes    = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples: List[Tuple[str, int]] = []
        for cls in classes:
            cls_dir = self.data_root / cls
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if blur_filter and laplacian_variance(str(img_path)) < BLUR_THRESH:
                    continue
                self.samples.append((str(img_path), self.class_to_idx[cls]))

        self.targets = [s[1] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class PotatoHealthyDataset(Dataset):
    """
    Single-class dataset containing only Potato__healthy images.

    Used exclusively for VAE training.  Applies augmentation to increase
    the effective diversity of the 152 available images.

    Args:
        data_root:  Path to the PlantVillage color folder.
        transform:  VAE-specific transform (outputs tensors in [-1, 1]).
    """

    def __init__(self, data_root: str | Path, transform=None) -> None:
        self.root = Path(data_root) / "Potato___healthy"
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Potato___healthy directory not found: {self.root}"
            )
        self.files = [
            str(p)
            for p in self.root.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        self.transform = transform
        print(f"PotatoHealthyDataset: {len(self.files)} images found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ─────────────────────────────────────────────────────────────────────────────
#  Weighted Sampler (class imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples minority classes.

    Particularly important for Potato__healthy (152 images vs 1,909 for the
    majority class — a 12.6× imbalance).

    Args:
        labels: List of integer class labels for the training split.
    Returns:
        WeightedRandomSampler instance.
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str | Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    seed: int = 42,
    test_size: float = 0.20,
    val_fraction: float = 0.50,
    blur_filter: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build stratified train / val / test DataLoaders (80 / 10 / 10 split).

    The split is reproducible via *seed* and uses the same random state
    as all CropGuard AI experiments.

    Args:
        data_root:      PlantVillage color folder.
        batch_size:     Batch size for all loaders.
        num_workers:    DataLoader worker processes.
        seed:           Random seed for stratified split.
        test_size:      Fraction of data for (val + test) combined.
        val_fraction:   Fraction of (val + test) assigned to val.
        blur_filter:    Exclude blurry images using Laplacian filter.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    set_seed(seed)

    # Full dataset (no transform yet — we apply per-split below)
    full_ds = PlantVillageDataset(data_root, transform=None, blur_filter=blur_filter)

    indices = list(range(len(full_ds)))
    labels  = full_ds.targets

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=val_fraction, stratify=temp_labels, random_state=seed
    )

    # Apply transforms per split
    from torch.utils.data import Subset
    from copy import deepcopy

    train_ds      = deepcopy(full_ds); 
    train_ds.transform = get_train_transform()
    val_test_tfm  = get_val_test_transform()
    val_ds        = deepcopy(full_ds); val_ds.transform   = val_test_tfm
    test_ds       = deepcopy(full_ds); test_ds.transform  = val_test_tfm

    train_sub = Subset(train_ds, train_idx)
    val_sub   = Subset(val_ds,   val_idx)
    test_sub  = Subset(test_ds,  test_idx)

    train_labels = [labels[i] for i in train_idx]
    sampler = build_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_sub, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(
        f"DataLoaders ready — Train: {len(train_sub):,}  "
        f"Val: {len(val_sub):,}  Test: {len(test_sub):,}"
    )
    return train_loader, val_loader, test_loader
