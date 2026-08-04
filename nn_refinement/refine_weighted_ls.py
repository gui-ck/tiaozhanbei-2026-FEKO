# -*- coding: utf-8 -*-
"""
加权最小二乘重构脚本
====================

目的：
- 当前整体 L2 误差主要受点数最多的 xoz 面影响；
- 通过按扫描面等权重加权，让三个扫描面对重构目标的贡献更均衡，
  期望改善 FEKO 全平面验证的整体相对 L2 误差；
- 只使用现有 FEKO T_env，不需要重新跑单元偶极子仿真。

方法：
1. 读取 T_env 和初步偶极子位置；
2. 每个扫描面的观测行乘上 1/sqrt(该面点数)，使三个面总权重相同；
3. 在加权后的方程 W*T*m = W*b 上做最小二乘；
4. 保存为 Output_ESR/equivalent_dipoles_weighted.txt，供 FEKO 验证。

用法：
    python nn_refinement/refine_weighted_ls.py
"""

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
    tenv = nn_common.load_tenv()
    src = np.asarray(tenv["src_coords"])
    T = np.asarray(tenv["T"])
    offsets = np.asarray(tenv["plane_offsets"], dtype=int)

    positions, initial_moments = nn_common.load_initial_dipoles()
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

    # 每个扫描面等权：权重 = 1/sqrt(该面点数)
    weights = np.ones(len(b))
    for p_idx in range(len(offsets)):
        start = offsets[p_idx]
        end = offsets[p_idx + 1] if p_idx + 1 < len(offsets) else len(b) // 3
        n_plane = end - start
        if n_plane > 0:
            weights[start * 3:end * 3] = 1.0 / np.sqrt(n_plane)

    Tw = T_sel * weights[:, None]
    bw = b * weights
    m_wls, _, _, _ = np.linalg.lstsq(Tw, bw, rcond=None)
    moments = m_wls.reshape(-1, 3)

    out_txt = os.path.join(esr.OUTPUT_DIR, "equivalent_dipoles_weighted.txt")
    esr.save_equivalent_dipoles_txt(out_txt, positions, moments)
    print(f"加权最小二乘偶极子已保存: {out_txt}")
    print(f"与初步偶极矩的相对差: "
          f"{np.linalg.norm(moments - initial_moments) / np.linalg.norm(initial_moments):.6f}")

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "plane_equal_weighted_ls",
        "dipole_file": out_txt,
        "n_dipoles": len(positions),
    }
    with open(os.path.join(esr.OUTPUT_DIR, "weighted_ls_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
