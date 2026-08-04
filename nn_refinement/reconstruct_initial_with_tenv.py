# -*- coding: utf-8 -*-
"""
脚本 1/5：基于 FEKO 环境感知 T_env 的传统稀疏重构
==================================================

功能：
- 读取真实扫描磁场数据（三个近场面的 .hfe）；
- 读取已经由 FEKO 单元偶极子仿真生成的环境感知传递矩阵 T_env；
- 使用组正交匹配追踪（Group OMP）在 T_env 支持的位置上做稀疏重构，
  得到初步偶极子阵列的数量、位置和复偶极矩；
- 对入选位置做合并/裁剪后，再用 T_env 最小二乘重拟合偶极矩；
- 输出 ``Output_ESR/equivalent_dipoles.txt``，供后续训练样本生成使用。

说明：
- 这里的“候选位置”来自 T_env.npz 中 FEKO 已经仿真过的偶极子位置；
- 机箱屏蔽、通风孔泄漏和周围设备的反射已包含在 T_env 中，
  因此重构结果比自由空间格林函数更接近真实环境；
- Mx/My 为磁偶极矩，Pz 为电偶极矩，与 FEKO 中 A6/A5 源卡片对应。

用法：
    python nn_refinement/reconstruct_initial_with_tenv.py
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402
import nn_common  # noqa: E402


def _columns_for_positions(tenv_src_coords, positions):
    """
    根据位置列表返回 T_env 中对应的列索引。

    T_env 的列顺序是“每个位置依次 3 列：Mx, My, Pz”，
    因此第 i 个位置的列是 [3*i, 3*i+1, 3*i+2]。
    """
    cols = []
    for pos in positions:
        match = np.where(
            np.all(np.isclose(tenv_src_coords, pos), axis=1)
        )[0]
        if len(match) != 1:
            raise RuntimeError(f"偶极子位置不在 T_env 候选位置中: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    return np.asarray(cols, dtype=int)


def main():
    parser = argparse.ArgumentParser(
        description="基于 FEKO T_env 的传统稀疏重构"
    )
    parser.add_argument("--max-support", type=int, default=60,
                        help="稀疏求解的最大偶极子数量（默认 60）")
    parser.add_argument("--cv-fraction", type=float, default=0.2,
                        help="每个扫描面随机留出的验证点比例（默认 0.2）")
    parser.add_argument("--prune-rel", type=float, default=0.001,
                        help="删除矩幅小于最大矩幅比例的分量（默认 0.001）")
    parser.add_argument("--merge-dist", type=float, default=0.005,
                        help="合并距离小于该值的偶极子，单位 m（默认 0.005）")
    parser.add_argument("--seed", type=int, default=12345,
                        help="交叉验证随机种子")
    args = parser.parse_args()

    # 1. 读取 FEKO 环境感知 T_env，并检查矩阵来源
    tenv = nn_common.load_tenv()
    if str(tenv["source"]) != "feko_unit_dipole":
        raise RuntimeError(
            "当前 T_env 不是 FEKO 环境感知矩阵，请等待 T_env 批处理完成"
        )
    tenv_src = np.asarray(tenv["src_coords"])
    T = np.asarray(tenv["T"])
    print(f"T_env: shape={T.shape}, 候选位置={len(tenv_src)} 个")

    # 2. 读取真实扫描数据，并按主流程相同的规则抽稀
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
    b = np.concatenate([H.ravel() for H in sub_H_list])
    if not np.allclose(sub_coords, np.asarray(tenv["obs_coords"])):
        raise RuntimeError("T_env 观测点与当前扫描数据抽稀结果不一致")

    # 3. 按扫描面做 80/20 留出切分，用验证误差选择偶极子数量
    rng = np.random.default_rng(args.seed)
    rows_train = []
    rows_test = []
    for plane_idx in range(len(planes)):
        n_plane = len(sub_coords_list[plane_idx])
        start = plane_offsets[plane_idx]
        idx = np.arange(n_plane)
        rng.shuffle(idx)
        n_test = max(1, int(round(args.cv_fraction * n_plane)))
        rows_test.append(start + idx[:n_test])
        rows_train.append(start + idx[n_test:])
    rows_train = np.concatenate(rows_train)
    rows_test = np.concatenate(rows_test)

    n_sites = T.shape[1] // 3
    max_support = min(args.max_support, n_sites)
    print(f"Group OMP 最大支持数: {max_support}")
    _, _, _, val_errors = esr.group_omp_run(
        T[rows_train], b[rows_train], max_support,
        T[rows_test], b[rows_test]
    )
    n_dipoles = esr.choose_support_size(
        val_errors,
        min_support=min(esr.MIN_SUPPORT, max_support),
        tolerance=0.02,
    )
    print(f"交叉验证选定初步偶极子数量: {n_dipoles}")

    # 4. 用全部观测点做最终稀疏求解
    selected_groups, x_all, train_errors, _ = esr.group_omp_run(
        T, b, n_dipoles
    )
    positions = tenv_src[selected_groups].copy()
    moments = x_all.reshape(-1, 3)
    final_train_error = train_errors[-1]
    print(f"支持集偶极子数: {len(positions)}, "
          f"训练相对误差: {final_train_error * 100:.4f}%")

    # 5. 合并近距离偶极子并删除弱偶极子，再用 T_env 重拟合
    positions, moments = esr.merge_prune_dipoles(
        positions, moments,
        merge_dist=args.merge_dist,
        rel_threshold=args.prune_rel,
    )
    if len(positions) > 0:
        cols = _columns_for_positions(tenv_src, positions)
        m_refit, _, _, _ = np.linalg.lstsq(T[:, cols], b, rcond=None)
        moments = m_refit.reshape(-1, 3)
    print(f"裁剪/合并后初步偶极子数量: {len(positions)}")

    # 6. 保存初步偶极子阵列与元数据
    txt_path = os.path.join(esr.OUTPUT_DIR, "equivalent_dipoles.txt")
    esr.save_equivalent_dipoles_txt(txt_path, positions, moments)
    print(f"初步偶极子阵列已保存: {txt_path}")

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frequency_hz": esr.FREQ_HZ,
        "method": "group_omp_on_feko_tenv + lstsq refit",
        "tenv_source": str(tenv["source"]),
        "tenv_shape": list(T.shape),
        "n_candidate_sites": n_sites,
        "selected_n_dipoles": len(positions),
        "final_train_error_percent": round(
            float(final_train_error * 100.0), 6
        ),
        "prune_rel": args.prune_rel,
        "merge_dist": args.merge_dist,
        "model": esr.MODEL_TAG,
    }
    esr.save_metadata(meta)
    with open(os.path.join(esr.OUTPUT_DIR, "tenv_initial_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("传统重构元数据已保存")


if __name__ == "__main__":
    main()
