import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config
from checkpoint import load_cnn_model, save_cnn_checkpoint
from data import load_classification_bmps, train_val_datasets
from model import ChineseCharCNN


def _val_loss_improved(val_loss, best_val_loss, rel_delta):
    if best_val_loss == float("inf"):
        return True
    if rel_delta > 0:
        if best_val_loss <= 0:
            return val_loss < best_val_loss
        return val_loss < best_val_loss * (1.0 - rel_delta)
    return val_loss < best_val_loss


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(image_root: str, random_state: int, model_out: str):
    _set_seed(random_state)

    X, y = load_classification_bmps(image_root)
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
    model = ChineseCharCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    use_early_stop = config.EARLY_STOP_PATIENCE > 0 and len(val_ds) > 0
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    last_epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        last_epoch = epoch
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
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item()
                val_batches += 1
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = running_loss / max(n_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)

        if use_early_stop:
            if _val_loss_improved(val_loss, best_val_loss, config.EARLY_STOP_MIN_DELTA):
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.EARLY_STOP_PATIENCE:
                    print(
                        f"早停于 epoch {epoch}/{config.EPOCHS}  "
                        f"验证集 loss={best_val_loss:.6f}"
                    )
                    break

        if (
            epoch == 1
            or epoch % max(1, config.EPOCHS // 10) == 0
            or epoch == config.EPOCHS
        ):
            print(
                f"Epoch {epoch:4d}/{config.EPOCHS}  "
                f"loss={avg_loss:.4f}  val_loss={val_loss:.6f}  "
                f"训练集准确率={train_acc:.4f}  验证集准确率={val_acc:.4f}"
            )

    if use_early_stop and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        train_correct = train_total = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                train_correct += (model(xb).argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_correct += (model(xb).argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

    print(
        f"\n最终（epoch {last_epoch}/{config.EPOCHS}）训练集准确率: {train_acc:.4f}  "
        f"验证集准确率: {val_acc:.4f}"
    )

    out_path = Path(model_out)
    save_cnn_checkpoint(
        out_path, model, num_classes=config.NUM_CLASSES, random_state=random_state
    )
    print(f"已保存模型: {out_path}")


def test_classifier_all(image_root: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_cnn_model(Path(model_path), device)

    X, y = load_classification_bmps(image_root)

    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

    acc = correct / max(total, 1)
    print(f"全量数据准确率（无划分）: {acc:.4f}")
