# -*- coding: utf-8 -*-
"""
线性叠加验证脚本
================

目的：
- 验证“偶极子阵列的 FEKO 磁场是否等于各单元偶极子 T_env 列的线性叠加”；
- 如果成立，说明环境本身是线性的，T_env 建模没有问题；
- 如果不成立，则说明 FEKO 模型中存在非线性或建模不一致。

方法：
1. 读取 FEKO 环境感知 T_env.npz；
2. 读取传统重构得到的初步偶极子阵列 equivalent_dipoles.txt；
3. 用 T_env 的列按偶极矩线性组合，得到抽稀观测点上的线性叠加场；
4. 读取 FEKO 直接仿真该偶极子阵列得到的三个 .hfe；
5. 在相同抽稀观测点上逐点比较，输出整体相对 L2 误差。

用法：
    python nn_refinement/verify_superposition.py
"""

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


def coord_key(coord, ndigits=8):
    """把坐标转成可哈希的元组，用于精确匹配观测点。"""
    return tuple(round(float(v), ndigits) for v in coord)


def main():
    tenv = nn_common.load_tenv()
    src = np.asarray(tenv["src_coords"])
    obs = np.asarray(tenv["obs_coords"])
    T = np.asarray(tenv["T"])
    offsets = np.asarray(tenv["plane_offsets"], dtype=int)
    print(f"T_env: {T.shape}, obs={obs.shape}, plane_offsets={offsets}")

    positions, moments = nn_common.load_initial_dipoles(
        os.path.join(esr.OUTPUT_DIR, "equivalent_dipoles.txt")
    )
    print(f"初步偶极子数量: {len(positions)}")

    # 组装该阵列在 T_env 观测点上对应的列
    cols = []
    for pos in positions:
        match = np.where(np.all(np.isclose(src, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"偶极子位置不在 T_env 中: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    cols = np.asarray(cols, dtype=int)
    H_linear = T[:, cols] @ moments.ravel()

    # 读取 FEKO 直接仿真该阵列得到的三个全平面 .hfe
    hfe_dir = os.path.join(
        esr.OUTPUT_DIR, "nn_refinement", "feko_validation_initial"
    )
    plane_results = {}
    b_feko_all = []
    b_lin_all = []

    for p_idx, pdef in enumerate(esr.PLANE_DEFS):
        name = pdef["name"]
        hfe_path = os.path.join(
            hfe_dir, f"Model_pred_NearField_{name}.hfe"
        )
        if not os.path.exists(hfe_path):
            raise FileNotFoundError(hfe_path)
        coords_full, H_full, _ = esr.read_hfe(hfe_path)

        # T_env 观测点按面连续排列，取本面的行范围
        start = offsets[p_idx]
        end = offsets[p_idx + 1] if p_idx + 1 < len(offsets) else len(obs)
        obs_plane = obs[start:end]
        H_lin_plane = H_linear[start * 3:end * 3]

        # 建立完整 hfe 坐标 -> 行号 的映射
        key_to_row = {
            coord_key(c): r for r, c in enumerate(coords_full)
        }
        rows = np.asarray(
            [key_to_row[coord_key(c)] for c in obs_plane], dtype=int
        )
        H_feko_plane = H_full[rows]

        rel = np.linalg.norm(H_feko_plane - H_lin_plane.reshape(-1, 3)) / (
            np.linalg.norm(H_feko_plane) + 1e-30
        )
        plane_results[name] = round(float(rel * 100.0), 8)
        print(f"{name}: 匹配 {len(rows)} 点, 相对误差 {plane_results[name]}%")

        b_feko_all.append(H_feko_plane.ravel())
        b_lin_all.append(H_lin_plane)

    b_feko_all = np.concatenate(b_feko_all)
    b_lin_all = np.concatenate(b_lin_all)
    overall = np.linalg.norm(b_feko_all - b_lin_all) / (
        np.linalg.norm(b_feko_all) + 1e-30
    )
    result = {
        "method": "linear_superposition_check",
        "overall_rel_l2_error_percent": round(float(overall * 100.0), 8),
        "planes": plane_results,
        "dipole_file": "Output_ESR/equivalent_dipoles.txt",
        "n_dipoles": len(positions),
        "n_obs_matched": len(b_feko_all) // 3,
        "conclusion": (
            "误差接近 0 说明 FEKO 环境下偶极子阵列响应可线性叠加，"
            "T_env 建模有效；误差较大则说明存在非线性或模型不一致。"
        ),
    }
    out_path = os.path.join(esr.OUTPUT_DIR, "superposition_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"整体线性叠加相对误差: {result['overall_rel_l2_error_percent']}%")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
