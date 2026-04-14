from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from net import Net


def _sorted_class_dirs(root: Path):
    dirs = [p for p in root.iterdir() if p.is_dir()]

    def sort_key(p: Path):
        return 0, int(p.name)

    return sorted(dirs, key=sort_key)


def load_classification_bmps(root_dir):
    root = Path(root_dir)
    class_dirs = _sorted_class_dirs(root)
    num_classes = len(class_dirs)

    features = []
    labels = []
    h, w = config.CLASS_IMAGE_SIZE

    for class_idx, sub in enumerate(class_dirs):
        bmps = sorted(sub.glob("*.bmp"))
        for p in bmps:
            img = Image.open(p).convert("L")  # 图片转灰度
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float64) / 255.0
            features.append(arr.reshape(-1))
            labels.append(class_idx)

    X = np.stack(features, axis=0)
    y_idx = np.array(labels, dtype=np.int64)
    Y = np.eye(num_classes)[y_idx]
    return X, Y


def classification(image_root, random_state):
    X, Y = load_classification_bmps(image_root)
    num_classes = Y.shape[1]
    y_idx = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=random_state, stratify=y_idx
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

    net.fit(X_train, y_train)
    acc_train = net.score(X_train, y_train)
    acc_test = net.score(X_test, y_test)
    print(f"训练集准确率: {acc_train:.4f}  测试集准确率: {acc_test:.4f}")
