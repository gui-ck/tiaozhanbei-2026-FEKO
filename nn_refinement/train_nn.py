# -*- coding: utf-8 -*-
"""
脚本 3/4：训练“磁场 -> 偶极矩”神经网络
=======================================

功能：
- 读取 training_samples.npz；
- 用 StandardScaler 标准化输入输出；
- 默认用“单层线性神经网络”（最小二乘闭式解）训练回归网络；
- 也支持 --solver mlp 使用 sklearn 的 MLPRegressor；
- 保存模型、缩放器和训练指标。

用法：
    python nn_refinement/train_nn.py --solver linear
    python nn_refinement/train_nn.py --solver mlp --hidden 256,128 --max-iter 200
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import nn_common  # noqa: E402


class LinearNN:
    """
    单层线性神经网络。

    输入 X -> 线性层 -> 输出 Y，使用 MSE 最小二乘闭式解训练：

    .. math::

        W = (A^T A + \\alpha I)^{-1} A^T Y,  A = [X, 1]

    由于“磁场 -> 偶极矩”的物理前向模型本身就是线性的
    （场 = T_env @ 偶极矩），线性网络能稳定达到全局最优；
    sklearn 的 ReLU MLP 在该高维反演任务上经测试难以收敛。
    """

    def __init__(self, alpha=1e-6):
        self.alpha = alpha
        self.W = None

    def fit(self, X, Y):
        A = np.hstack([X, np.ones((len(X), 1))])
        ATA = A.T @ A + self.alpha * np.eye(A.shape[1])
        ATY = A.T @ Y
        self.W = np.linalg.solve(ATA, ATY)
        return self

    def predict(self, X):
        A = np.hstack([X, np.ones((len(X), 1))])
        return A @ self.W


def main():
    parser = argparse.ArgumentParser(description="训练磁场->偶极矩神经网络")
    parser.add_argument("--solver", choices=["linear", "mlp"],
                        default="linear",
                        help="linear=线性神经网络闭式解（默认）；mlp=sklearn MLP")
    parser.add_argument("--hidden", default="256,128",
                        help="隐藏层神经元数，逗号分隔")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = nn_common.ensure_output_dir()
    samples_path = os.path.join(out_dir, "training_samples.npz")
    data = np.load(samples_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    print(f"训练样本: X={X.shape}, Y={Y.shape}")

    # 随机划分训练/验证集
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(round(args.val_fraction * len(X))))
    val_idx = rng.choice(len(X), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 标准化输入输出，提升 MLP 训练稳定性
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val)
    Y_train_s = scaler_y.fit_transform(Y_train)
    Y_val_s = scaler_y.transform(Y_val)

    if args.solver == "linear":
        print("训练线性神经网络（最小二乘闭式解）...")
        model = LinearNN(alpha=args.alpha)
        model.fit(X_train_s, Y_train_s)
        pred_train = scaler_y.inverse_transform(model.predict(X_train_s))
        pred_val = scaler_y.inverse_transform(model.predict(X_val_s))
        r2_train = model.score(X_train_s, Y_train_s) if hasattr(model, "score") else None
        r2_val = None
        n_iter_used = 1
        loss_tail = []
    else:
        hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())
        print("训练 sklearn MLP 神经网络 ...")
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate_init=1e-3,
            max_iter=args.max_iter,
            early_stopping=True,
            n_iter_no_change=args.patience,
            validation_fraction=0.1,
            random_state=args.seed,
        )
        model.fit(X_train_s, Y_train_s)
        pred_train = scaler_y.inverse_transform(model.predict(X_train_s))
        pred_val = scaler_y.inverse_transform(model.predict(X_val_s))
        r2_train = model.score(X_train_s, Y_train_s)
        r2_val = model.score(X_val_s, Y_val_s)
        n_iter_used = int(model.n_iter_)
        loss_tail = [float(v) for v in model.loss_curve_[-10:]]

    # 训练与验证指标
    rel_train = np.linalg.norm(pred_train - Y_train) / np.linalg.norm(Y_train)
    rel_val = np.linalg.norm(pred_val - Y_val) / np.linalg.norm(Y_val)
    if r2_train is None:
        # 线性网络手动计算决定系数 R2
        ss_res = np.sum((Y_train - pred_train) ** 2)
        ss_tot = np.sum((Y_train - Y_train.mean(axis=0)) ** 2)
        r2_train = 1.0 - ss_res / ss_tot
    if r2_val is None:
        ss_res = np.sum((Y_val - pred_val) ** 2)
        ss_tot = np.sum((Y_val - Y_val.mean(axis=0)) ** 2)
        r2_val = 1.0 - ss_res / ss_tot

    metrics = {
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "feature_dim": X.shape[1],
        "target_dim": Y.shape[1],
        "solver": args.solver,
        "hidden_layers": list(args.hidden.split(",")),
        "max_iter": args.max_iter,
        "n_iter_used": n_iter_used,
        "r2_train": round(float(r2_train), 6),
        "r2_val": round(float(r2_val), 6),
        "rel_l2_error_train_percent": round(float(rel_train * 100.0), 4),
        "rel_l2_error_val_percent": round(float(rel_val * 100.0), 4),
        "loss_curve_tail": loss_tail,
    }
    print("训练完成:", json.dumps(metrics, ensure_ascii=False, indent=2))

    # 保存模型、缩放器与指标
    if args.solver == "linear":
        np.savez_compressed(
            os.path.join(out_dir, "linear_nn_weights.npz"),
            W=model.W,
            model_type="linear_nn",
        )
        # 删除旧版 sklearn MLP 模型，避免验证脚本误用
        old_mlp = os.path.join(out_dir, "nn_model.joblib")
        if os.path.exists(old_mlp):
            os.remove(old_mlp)
    else:
        joblib.dump(model, os.path.join(out_dir, "nn_model.joblib"))
    joblib.dump(scaler_x, os.path.join(out_dir, "scaler_x.joblib"))
    joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.joblib"))
    with open(os.path.join(out_dir, "training_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("模型与指标已保存到", out_dir)


if __name__ == "__main__":
    main()
