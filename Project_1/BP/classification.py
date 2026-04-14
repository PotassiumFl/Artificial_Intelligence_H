from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from checkpoint import load_bp_checkpoint, save_bp_checkpoint
from net import Net


def load_classification_bmps(root_dir):
    root = Path(root_dir)

    features = []
    labels = []
    h, w = config.CLASS_IMAGE_SIZE
    resample = Image.Resampling.LANCZOS

    for label in range(1, config.NUM_CLASSES + 1):
        sub = root / str(label)

        bmps = sorted(sub.glob("*.bmp")) + sorted(sub.glob("*.BMP"))

        for p in bmps:
            img = Image.open(p).convert("L")
            img = img.resize((w, h), resample)
            arr = np.asarray(img, dtype=np.float64) / 255.0
            features.append(arr.reshape(-1))
            labels.append(label - 1)

    X = np.stack(features, axis=0)
    y_idx = np.array(labels, dtype=np.int64)
    Y = np.eye(config.NUM_CLASSES)[y_idx]
    return X, Y


def classification(image_root, random_state, model_out):
    X, Y = load_classification_bmps(image_root)
    num_classes = config.NUM_CLASSES
    y_idx = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=config.CLASSIFICATION_TEST_SIZE,
        random_state=random_state,
        stratify=y_idx,
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    hidden_sizes = [config.HIDDEN_SIZE] * config.HIDDEN_LAYER_NUM
    net = Net(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=num_classes,
        task="classification",
        seed=random_state,
    )

    net.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    acc_train = net.score(X_train, y_train)
    acc_test = net.score(X_test, y_test)
    print(f"训练集准确率: {acc_train:.4f}  测试集准确率: {acc_test:.4f}")

    out_path = Path(model_out)
    save_bp_checkpoint(
        out_path, net, sc, random_state, task="classification", scaler_y=None
    )
    print(f"已保存模型: {out_path}")


def test_classification_all(image_root, model_path):
    ckpt = load_bp_checkpoint(Path(model_path))
    if ckpt["task"] != "classification":
        raise ValueError("检查点不是分类模型")
    X, Y = load_classification_bmps(image_root)
    X = ckpt["scaler_X"].transform(X)
    net = Net.from_state(ckpt["net_state"])
    acc = net.score(X, Y)
    print(f"全量数据准确率（无划分）: {acc:.4f}")
