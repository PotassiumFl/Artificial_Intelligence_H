from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

import config


def _sorted_class_dirs(root: Path):
    dirs = [p for p in root.iterdir() if p.is_dir()]

    def sort_key(p: Path):
        return 0, int(p.name)

    return sorted(dirs, key=sort_key)


def load_classification_bmps(root_dir):
    root = Path(root_dir)
    class_dirs = _sorted_class_dirs(root)
    num_classes = len(class_dirs)

    images = []
    labels = []
    h, w = config.CLASS_IMAGE_SIZE

    for class_idx, sub in enumerate(class_dirs):
        bmps = sorted(sub.glob("*.bmp"))
        for p in bmps:
            img = Image.open(p).convert("L")
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(arr[np.newaxis, ...])  # (1, H, W)
            labels.append(class_idx)

    X = np.stack(images, axis=0)  # (N, 1, H, W)
    y = np.array(labels, dtype=np.int64)
    return X, y, num_classes


def train_val_datasets(X, y, val_fraction, random_state):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y,
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return train_ds, val_ds
