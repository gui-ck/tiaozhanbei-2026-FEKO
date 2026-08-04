# -*- coding: utf-8 -*-
"""
新增候选偶极子位置筛选脚本
==========================

目的：
- 当前 T_env 只覆盖旧的 99 个位置，真实扫描场仍有约 16.95% 无法表示；
- 本脚本用“自由空间格林函数残差相关性”从较密的候选网格中挑选
  最能解释当前残差的新偶极子位置；
- 选出的位置后续会补做 FEKO T_env 列，用于扩大等效源支持集。

方法：
1. 读取现有 FEKO T_env 和初步偶极子阵列；
2. 计算当前重构在抽稀观测点上的残差 r = b - T_selected @ m；
3. 在机箱候选范围内生成较密网格（默认 4 cm）；
4. 用自由空间格林函数计算每个候选位置 3 个分量与残差的相关性；
5. 按组相关性强弱排序，贪心选出彼此距离足够远的前 N 个新位置。

用法：
    python nn_refinement/select_new_candidates.py --n-candidates 30
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


def main():
    parser = argparse.ArgumentParser(description="筛选新增偶极子候选位置")
    parser.add_argument("--n-candidates", type=int, default=30,
                        help="需要筛选的新候选位置数量（默认 30）")
    parser.add_argument("--spacing", type=float, default=0.04,
                        help="自由空间粗筛网格间距，单位 m（默认 0.04）")
    parser.add_argument("--min-dist", type=float, default=0.04,
                        help="新候选位置之间的最小距离，单位 m（默认 0.04）")
    args = parser.parse_args()

    tenv = nn_common.load_tenv()
    src = np.asarray(tenv["src_coords"])
    obs = np.asarray(tenv["obs_coords"])
    T = np.asarray(tenv["T"])

    positions, moments = nn_common.load_initial_dipoles()
    cols = []
    for pos in positions:
        match = np.where(np.all(np.isclose(src, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"位置不在 T_env 中: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    cols = np.asarray(cols, dtype=int)
    T_sel = T[:, cols]

    planes = esr.load_all_planes()
    b_parts = []
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep = esr.compute_subsample_indices(
            len(mag), mag, esr.OBS_SUBSAMPLE_STEP, esr.PEAK_KEEP_FRACTION
        )
        b_parts.append(plane["H"][keep].ravel())
    b = np.concatenate(b_parts)

    residual = b - T_sel @ moments.ravel()
    print(f"当前重构残差范数占比: "
          f"{np.linalg.norm(residual) / np.linalg.norm(b) * 100:.4f}%")

    # 生成较密候选网格，并去掉已经仿真过的位置
    dense = esr.build_candidate_grid(spacing=args.spacing)
    keep = np.ones(len(dense), dtype=bool)
    for pos in src:
        keep &= np.linalg.norm(dense - pos, axis=1) > 0.02
    dense = dense[keep]
    print(f"自由空间粗筛候选位置: {len(dense)} 个")

    # 自由空间格林函数列（Mx/My 磁偶极子，Pz 电偶极子）
    T_free = esr.build_free_space_T(obs, dense)
    corr = T_free.conj().T @ residual
    group_corr = np.linalg.norm(corr.reshape(-1, 3), axis=1)

    order = np.argsort(-group_corr)
    selected = []
    for idx in order:
        pos = dense[idx]
        if len(selected) > 0:
            dist = np.linalg.norm(np.asarray(selected) - pos, axis=1)
            if dist.min() < args.min_dist:
                continue
        selected.append(pos.tolist())
        if len(selected) >= args.n_candidates:
            break

    selected = np.asarray(selected)
    out_path = os.path.join(esr.OUTPUT_DIR, "new_candidate_positions.npy")
    np.save(out_path, selected)
    print(f"已筛选 {len(selected)} 个新候选位置: {out_path}")
    print(selected)

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_candidates": len(selected),
        "spacing": args.spacing,
        "min_dist": args.min_dist,
        "method": "free_space_residual_correlation",
        "file": out_path,
    }
    with open(os.path.join(esr.OUTPUT_DIR, "new_candidates_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
