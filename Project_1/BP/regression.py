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
            if not row or all(not (c or "").strip() for c in row):
                continue
            x = float(row[idx_in])
            y_val = float(row[idx_out])
            rows.append((x, y_val))

    data = np.array(rows, dtype=np.float64)
    X = data[:, 0:1]
    y = data[:, 1:2]
    return X, y


def mae_on_original_y(net, X, y_scaled, scaler_y):
    """在 y 的原始量纲上计算 MAE（预测与标签均经 scaler_y 反标准化）。"""
    pred = net.predict(X)
    pred_o = scaler_y.inverse_transform(pred)
    y_o = scaler_y.inverse_transform(y_scaled)
    return float(np.mean(np.abs(pred_o - y_o)))


def regression(csv_path, random_state, model_out):
    X, y = load_regression_csv(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.REGRESSION_TEST_SIZE,
        random_state=random_state,
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

    net.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    train_mae = mae_on_original_y(net, X_train, y_train, scY)
    test_mae = mae_on_original_y(net, X_test, y_test, scY)

    print(
        f"训练集 MAE (y 原始尺度): {train_mae:.6f}  "
        f"测试集 MAE (y 原始尺度): {test_mae:.6f}"
    )

    out_path = Path(model_out)
    save_bp_checkpoint(
        out_path, net, scX, random_state, task="regression", scaler_y=scY
    )
    print(f"已保存模型: {out_path}")


def test_regression_all(csv_path, model_path):
    ckpt = load_bp_checkpoint(Path(model_path))
    X, y = load_regression_csv(csv_path)
    X = ckpt["scaler_X"].transform(X)
    y_s = ckpt["scaler_y"].transform(y)
    net = Net.from_state(ckpt["net_state"])
    mae = mae_on_original_y(net, X, y_s, ckpt["scaler_y"])
    print(f"全量数据 MAE（y 原始尺度，无划分）: {mae:.6f}")
