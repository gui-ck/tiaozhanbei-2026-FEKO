# -*- coding: utf-8 -*-
"""
基于全观测点 T_env 的重构脚本（方案 1 + 方案 2）
================================================

目的：
- 使用全观测点 T_env（81903 个扫描点）替代原来的抽稀 T_env；
- 生成两个候选模型：
  1. equivalent_dipoles_full40.txt：只用原有 40 个位置做全平面最小二乘；
  2. equivalent_dipoles_v2.txt：用原有 40 个位置 + 30 个新候选位置
     做全平面最小二乘，再合并/裁剪弱偶极子并重拟合。
- 两个模型随后都要通过 FEKO 就地仿真验证。

用法：
    python nn_refinement/reconstruct_full_tenv.py
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


FULL_TENV_PATH = os.path.join(esr.OUTPUT_DIR, "T_env_full.npz")


def columns_for_positions(src, positions):
    """返回位置列表对应的 T_env 列索引。"""
    cols = []
    for pos in positions:
        match = np.where(np.all(np.isclose(src, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"位置不在全观测点 T_env 中: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    return np.asarray(cols, dtype=int)


def main():
    if not os.path.exists(FULL_TENV_PATH):
        raise FileNotFoundError("请先运行 generate_full_tenv.py 生成全观测点 T_env")

    data = np.load(FULL_TENV_PATH, allow_pickle=True)
    if str(data["source"]) != "feko_unit_dipole_full":
        raise RuntimeError("T_env_full 来源不正确")
    src = np.asarray(data["src_coords"])
    obs = np.asarray(data["obs_coords"])
    T = np.asarray(data["T"])
    print(f"全观测点 T_env: {T.shape}, 位置 {len(src)} 个")

    # 全扫描观测磁场
    planes = esr.load_all_planes()
    b_parts = []
    obs_parts = []
    for plane in planes:
        b_parts.append(plane["H"].ravel())
        obs_parts.append(plane["coords"])
    b = np.concatenate(b_parts)
    if not np.allclose(np.vstack(obs_parts), obs):
        raise RuntimeError("全观测点 T_env 观测坐标与扫描数据不一致")

    def save_solution(txt_name, positions, moments, method):
        out_txt = os.path.join(esr.OUTPUT_DIR, txt_name)
        esr.save_equivalent_dipoles_txt(out_txt, positions, moments)
        err = np.linalg.norm(b - T[:, columns_for_positions(src, positions)]
                             @ moments.ravel()) / np.linalg.norm(b)
        print(f"{method}: {len(positions)} 个偶极子, "
              f"全平面拟合相对误差 {err * 100:.4f}%")
        return out_txt, round(float(err * 100.0), 6)

    results = {}

    # 方案 2：只用原有 40 个位置，全平面最小二乘
    pos40, _ = nn_common.load_initial_dipoles()
    cols40 = columns_for_positions(src, pos40)
    m40, _, _, _ = np.linalg.lstsq(T[:, cols40], b, rcond=None)
    out40, err40 = save_solution(
        "equivalent_dipoles_full40.txt", pos40, m40.reshape(-1, 3),
        "方案2（40位置全平面LS）"
    )
    results["full40"] = {
        "file": out40,
        "n_dipoles": len(pos40),
        "fit_error_percent": err40,
    }

    # 方案 1 + 2：原有 40 + 新候选 30，全平面 LS 后裁剪/合并/重拟合
    new_candidates = np.load(
        os.path.join(esr.OUTPUT_DIR, "new_candidate_positions.npy")
    )
    combined = np.vstack([pos40, new_candidates])
    unique = []
    for p in combined:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in unique):
            unique.append(p)
    combined = np.asarray(unique)
    cols_all = columns_for_positions(src, combined)
    m_all, _, _, _ = np.linalg.lstsq(T[:, cols_all], b, rcond=None)
    pos_prune, mom_prune = esr.merge_prune_dipoles(
        combined, m_all.reshape(-1, 3),
        merge_dist=esr.MERGE_DIST,
        rel_threshold=esr.PRUNE_REL_THRESHOLD,
    )
    if len(pos_prune) > 0:
        cols_prune = columns_for_positions(src, pos_prune)
        m_refit, _, _, _ = np.linalg.lstsq(T[:, cols_prune], b, rcond=None)
        mom_prune = m_refit.reshape(-1, 3)
    out_v2, err_v2 = save_solution(
        "equivalent_dipoles_v2.txt", pos_prune, mom_prune,
        "方案1+2（70位置全平面LS+裁剪）"
    )
    results["v2"] = {
        "file": out_v2,
        "n_dipoles": len(pos_prune),
        "fit_error_percent": err_v2,
    }

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "full_tenv": FULL_TENV_PATH,
        "method": "full_plane_lstsq + prune/merge",
        "results": results,
    }
    with open(os.path.join(esr.OUTPUT_DIR, "full_tenv_reconstruction.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("重构结果元数据已保存")


if __name__ == "__main__":
    main()
