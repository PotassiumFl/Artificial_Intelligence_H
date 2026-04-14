import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from data import load_classification_bmps, train_val_datasets
from model import ChineseCharCNN


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(image_root: str, random_state: int):
    _set_seed(random_state)

    X, y, num_classes = load_classification_bmps(image_root)
    train_ds, val_ds = train_val_datasets(
        X, y, val_fraction=config.VAL_FRACTION, random_state=random_state
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseCharCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)

        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = running_loss / max(n_batches, 1)

        if (
            epoch == 1
            or epoch % max(1, config.EPOCHS // 10) == 0
            or epoch == config.EPOCHS
        ):
            print(
                f"Epoch {epoch:4d}/{config.EPOCHS}  "
                f"loss={avg_loss:.4f}  "
                f"训练集准确率={train_acc:.4f}  验证集准确率={val_acc:.4f}"
            )

    print(f"\n最终 训练集准确率: {train_acc:.4f}  验证集准确率: {val_acc:.4f}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class-dir", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_seed = random.randint(1, int(1e9))
    train_classifier(args.class_dir, random_state=run_seed)
