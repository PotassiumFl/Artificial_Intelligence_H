import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from checkpoint import load_bp_checkpoint, save_bp_checkpoint
from net import Net


def load_regression_csv(csv_path):
    path = Path(csv_path)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        idx_in = header.index("input")
        idx_out = header.index("output")
        for row in reader:
            rows.append((float(row[idx_in]), float(row[idx_out])))
    data = np.array(rows, dtype=np.float64)
    X = data[:, 0:1]
    y = data[:, 1:2]
    return X, y


def regression(csv_path, random_state, model_out):
    X, y = load_regression_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    scX, scY = StandardScaler(), StandardScaler()
    X_train = scX.fit_transform(X_train)
    X_test = scX.transform(X_test)
    y_train = scY.fit_transform(y_train)
    y_test = scY.transform(y_test)

    hidden_sizes = [config.HIDDEN_SIZE] * config.HIDDEN_LAYER_NUM
    net = Net(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=1,
        task="regression",
        seed=random_state,
    )

    net.fit(X_train, y_train)
    train_mse = net.score(X_train, y_train)
    test_mse = net.score(X_test, y_test)
    print(f"训练集 MSE: {train_mse:.6f}  测试集 MSE: {test_mse:.6f}")

    out_path = Path(model_out)
    save_bp_checkpoint(
        out_path, net, scX, random_state, task="regression", scaler_y=scY
    )
    print(f"已保存模型: {out_path}")


def test_regression_all(csv_path, model_path):
    ckpt = load_bp_checkpoint(Path(model_path))
    X, y = load_regression_csv(csv_path)
    X = ckpt["scaler_X"].transform(X)
    y = ckpt["scaler_y"].transform(y)
    net = Net.from_state(ckpt["net_state"])
    mse = net.score(X, y)
    print(f"全量数据 MSE（无划分）: {mse:.6f}")
