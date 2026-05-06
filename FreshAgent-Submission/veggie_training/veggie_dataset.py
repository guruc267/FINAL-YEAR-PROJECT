"""
dataset.py – Vegetable dataset loading, stratified splitting, and augmentation pipeline.

DATASET STRUCTURE:
  root_dir/
    Ginger/Fresh/Fresh/<images>
    Ginger/Rotten/Rotten/<images>
    Ginger/Adulterated/Adulterated/<images>

To add more vegetables in future:
  - Just add the folder (e.g. Tomato/Fresh/, Tomato/Rotten/, Tomato/Adulterated/)
  - Add "Tomato" to VEGGIE_CLASSES list below — that's it!
"""
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

# ─── Class Maps ───────────────────────────────────────────────────────────────
VEGGIE_CLASSES   = ["Ginger"]           # Add more vegetables here later
COND_CLASSES     = ["Adulterated", "Fresh", "Rotten"]
VEGGIE2IDX       = {v: i for i, v in enumerate(VEGGIE_CLASSES)}
COND2IDX         = {c: i for i, c in enumerate(COND_CLASSES)}
COMBINED_CLASSES = [f"{vg}_{co}" for vg in VEGGIE_CLASSES for co in COND_CLASSES]
COMBINED2IDX     = {c: i for i, c in enumerate(COMBINED_CLASSES)}
IDX2COMBINED     = {i: c for c, i in COMBINED2IDX.items()}

# User-friendly display names
COND_DISPLAY = {
    "Adulterated": "Adulterated (Acid-washed)",
    "Fresh":       "Fresh",
    "Rotten":      "Rotten",
}

IMG_SIZE = 224   # EfficientNetV2-B0 native resolution


# ─── Transforms ───────────────────────────────────────────────────────────────
def get_train_transforms():
    return T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── Dataset Discovery ────────────────────────────────────────────────────────
def _find_images(directory: Path) -> list:
    """Recursively find all image files, handling any nested subfolder depth."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(directory.rglob(ext))
    return images


def discover_dataset(root_dir: str) -> list:
    """
    Walk the dataset directory and return a list of sample dicts.
    Handles 'Pure' as 'Fresh' mapping, and nested custom splits (e.g. train/val/test).
    """
    root = Path(root_dir)
    samples = []

    # Map raw folder names to expected standard conditions
    cond_map = {
        "adulterated": "Adulterated",
        "fresh": "Fresh",
        "pure": "Fresh",
        "rotten": "Rotten"
    }

    # If the user passed e.g. "path/to/Ginger" directly, we search inside it
    # If they passed "path/to/dataset", we search inside "path/to/dataset/Ginger"
    for veggie in VEGGIE_CLASSES:
        veggie_dir = root / veggie
        if not veggie_dir.exists() and root.name.lower() == veggie.lower():
            veggie_dir = root
            
        if not veggie_dir.exists():
            print(f"[WARN] Dir not found for {veggie}: {veggie_dir}")
            continue

        images = _find_images(veggie_dir)
        for img_path in images:
            # Figure out the condition by looking at parent folder names
            condition = None
            for part in img_path.parts:
                if part.lower() in cond_map:
                    condition = cond_map[part.lower()]
                    break
            
            if condition:
                combined = f"{veggie}_{condition}"
                if combined not in COMBINED2IDX:
                    continue
                samples.append({
                    "path":           str(img_path),
                    "veggie":         veggie,
                    "condition":      condition,
                    "combined_label": COMBINED2IDX[combined],
                    "veggie_label":   VEGGIE2IDX[veggie],
                    "cond_label":     COND2IDX[condition],
                })
                
    return samples


def print_dataset_stats(samples: list, split_name: str = "Dataset"):
    counts = defaultdict(int)
    for s in samples:
        counts[f"{s['veggie']} / {s['condition']}"] += 1
    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  {split_name} ({len(samples)} total images)")
    print(sep)
    for cls, cnt in sorted(counts.items()):
        print(f"  {cls:<38} : {cnt:>5}")
    print(f"{sep}\n")


def split_dataset(samples: list, val_size=0.15, test_size=0.15, seed=42):
    """Stratified 70/15/15 train/val/test split."""
    labels      = np.array([s["combined_label"] for s in samples])
    all_indices = np.arange(len(samples))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(all_indices, labels))

    adjusted_val = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
    train_local, val_local = next(sss2.split(trainval_idx, labels[trainval_idx]))

    return (
        [samples[i] for i in trainval_idx[train_local]],
        [samples[i] for i in trainval_idx[val_local]],
        [samples[i] for i in test_idx],
    )


def mixup_data(x, y_veggie, y_cond, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y_veggie, y_veggie[idx], y_cond, y_cond[idx], lam


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class VeggieAdulterationDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image":          img,
            "veggie_label":   torch.tensor(s["veggie_label"],   dtype=torch.long),
            "cond_label":     torch.tensor(s["cond_label"],     dtype=torch.long),
            "combined_label": torch.tensor(s["combined_label"], dtype=torch.long),
            "path":           s["path"],
        }


def build_dataloaders(root_dir: str, batch_size=32, num_workers=0, seed=42):
    print(f"[Dataset] Scanning: {root_dir}")
    samples = discover_dataset(root_dir)
    if not samples:
        raise FileNotFoundError(
            f"No images found in '{root_dir}'.\n"
            f"Expected veggies : {VEGGIE_CLASSES}\n"
            f"Expected conds   : {COND_CLASSES}"
        )

    train_s, val_s, test_s = split_dataset(samples, seed=seed)
    print_dataset_stats(train_s, "Train")
    print_dataset_stats(val_s,   "Validation")
    print_dataset_stats(test_s,  "Test")

    # Class-balanced weights
    comb    = np.array([s["combined_label"] for s in train_s])
    unique, counts = np.unique(comb, return_counts=True)
    n_cls   = len(COMBINED_CLASSES)
    weights = np.ones(n_cls, dtype=np.float32)
    for c, n in zip(unique, counts):
        weights[c] = len(comb) / (n_cls * n)
    class_weights = torch.tensor(weights)

    train_ds = VeggieAdulterationDataset(train_s, transform=get_train_transforms())
    val_ds   = VeggieAdulterationDataset(val_s,   transform=get_val_transforms())
    test_ds  = VeggieAdulterationDataset(test_s,  transform=get_val_transforms())

    kw = dict(num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        class_weights,
    )


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r".\veggie_dataset"
    build_dataloaders(root, batch_size=16)
