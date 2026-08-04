# -*- coding: utf-8 -*-
"""
脚本 2/4：生成神经网络训练样本
===============================

功能：
- 读取初步偶极子阵列（固定位置和数量）；
- 在初步偶极矩附近随机扰动（默认相对扰动 20%）；
- 通过 T_env 计算每个随机偶极子阵列在观测点上产生的磁场；
- 保存 “磁场特征 -> 偶极矩目标” 的训练样本为 npz。

用法：
    python nn_refinement/generate_training_samples.py \
        --n-samples 5000 --n-obs-points 500 --sigma 0.2
"""

import argparse
import datetime
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402
import nn_common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="生成神经网络训练样本")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="随机偶极子阵列数量（默认 5000）")
    parser.add_argument("--n-obs-points", type=int, default=500,
                        help="神经网络输入使用的观测点数量（默认 500）")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="偶极矩相对扰动标准差（默认 0.2）")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    out_dir = nn_common.ensure_output_dir()
    tenv = nn_common.load_tenv()
    positions, initial_moments = nn_common.load_initial_dipoles()
    print(f"初步偶极子: {len(positions)} 个")

    # 重新按主流程相同的抽稀规则读取扫描数据，校验与 T_env 的观测点一致
    planes = esr.load_all_planes()
    sub_coords_list = []
    sub_H_list = []
    plane_offsets = []
    offset = 0
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep_idx = esr.compute_subsample_indices(
            len(mag), mag, esr.OBS_SUBSAMPLE_STEP, esr.PEAK_KEEP_FRACTION
        )
        sub_coords_list.append(plane["coords"][keep_idx])
        sub_H_list.append(plane["H"][keep_idx])
        plane_offsets.append(offset)
        offset += len(keep_idx)
    sub_coords = np.vstack(sub_coords_list)

    if not np.allclose(sub_coords, tenv["obs_coords"]):
        raise RuntimeError("T_env 的观测点与当前抽稀结果不一致，请重新生成 T_env")

    # 选择神经网络输入观测点，并取出对应的 T_env 行
    plane_sizes = [len(c) for c in sub_coords_list]
    obs_indices = nn_common.compute_nn_obs_indices(
        plane_sizes, plane_offsets, args.n_obs_points
    )
    rows = nn_common.obs_indices_to_tenv_rows(obs_indices)
    # 初步偶极子是候选网格子集，这里统一建立观测行/偶极子列的子矩阵
    T_sub, src_cols, rows = nn_common.select_tenv_columns(
        tenv, positions, obs_indices
    )
    print(f"神经网络输入: {len(obs_indices)} 个观测点, "
          f"T 矩阵 {T_sub.shape}, 偶极子 {len(positions)} 个")

    n_dip = len(positions)
    rng = np.random.default_rng(args.seed)
    absm = np.abs(initial_moments)  # (N, 3)

    # 向量化生成随机偶极矩：在初始矩附近叠加相对扰动
    real_noise = rng.standard_normal((args.n_samples, n_dip, 3))
    imag_noise = rng.standard_normal((args.n_samples, n_dip, 3))
    random_moments = (
        initial_moments.real[None, :, :] + args.sigma * absm[None, :, :] * real_noise
        + 1j * (
            initial_moments.imag[None, :, :]
            + args.sigma * absm[None, :, :] * imag_noise
        )
    )

    # 初步阵列在所选观测点上产生的磁场，作为后续“残差输入”的基准
    base_field = T_sub @ initial_moments.ravel()

    # 一次矩阵乘法计算所有随机阵列的磁场：field = T_sub @ moment
    fields = random_moments.reshape(args.n_samples, -1) @ T_sub.T

    # 特征：复磁场归一化到单位范数 -> 实部+虚部；
    # 目标：偶极矩按同一范数同步缩放 -> 实部+虚部。
    # 因为场与矩是线性关系（field = T_env @ moment），
    # 归一化后网络学习的是“场分布形状 -> 矩相对比例”，
    # 推理时再把预测矩乘以真实场的范数，从而消除自由空间回退模型
    # 与真实屏蔽机箱之间约 4 个数量级的量纲差。
    X = np.empty((args.n_samples, T_sub.shape[0] * 2), dtype=np.float32)
    Y = np.empty((args.n_samples, n_dip * 6), dtype=np.float32)
    for i in range(args.n_samples):
        norm = float(np.linalg.norm(fields[i]))
        if norm < 1e-30:
            norm = 1.0
        X[i] = nn_common.fields_to_features(fields[i] / norm)
        Y[i] = nn_common.moments_to_targets(random_moments[i:i + 1] / norm)

    out_path = os.path.join(out_dir, "training_samples.npz")
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        positions=positions,
        initial_moments=initial_moments,
        base_field_sub=base_field,
        nn_obs_indices=obs_indices,
        plane_offsets=np.array(plane_offsets),
        plane_sizes=np.array(plane_sizes, dtype=int),
        src_col_indices=src_cols,
        sigma=args.sigma,
        n_samples=args.n_samples,
        feature_normalized=True,
        feature_dim=X.shape[1],
        target_dim=Y.shape[1],
        tenv_source=str(tenv["source"]),
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    print(f"训练样本已保存: {out_path}")
    print(f"X shape={X.shape}, Y shape={Y.shape}, "
          f"文件大小={os.path.getsize(out_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
