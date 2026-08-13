# -*- coding: utf-8 -*-
"""新增候选偶极子位置筛选。"""

import datetime
import json
import os

import numpy as np

from . import config, tenv


def select_new_candidates(obs_coords, residual, existing_src,
                          n_candidates=config.NEW_CANDIDATES,
                          spacing=config.NEW_CANDIDATE_SPACING,
                          min_dist=config.NEW_CANDIDATE_MIN_DIST):
    """
    用自由空间格林函数残差相关性筛选新位置。

    返回彼此距离不小于 min_dist 的前 n_candidates 个位置。
    """
    dense = tenv.build_candidate_grid(spacing=spacing)
    keep = np.ones(len(dense), dtype=bool)
    for pos in existing_src:
        keep &= np.linalg.norm(dense - pos, axis=1) > 0.02
    dense = dense[keep]
    print(f"自由空间粗筛候选位置: {len(dense)} 个")

    T_free = tenv.build_free_space_T(obs_coords, dense)
    corr = T_free.conj().T @ residual
    group_corr = np.linalg.norm(corr.reshape(-1, 3), axis=1)
    order = np.argsort(-group_corr)

    selected = []
    for idx in order:
        pos = dense[idx]
        if selected:
            dist = np.linalg.norm(np.asarray(selected) - pos, axis=1)
            if dist.min() < min_dist:
                continue
        selected.append(pos.tolist())
        if len(selected) >= n_candidates:
            break
    return np.asarray(selected)


def save_new_candidates(results_dir, selected, meta_extra=None):
    """保存新增候选位置与元数据。"""
    os.makedirs(results_dir, exist_ok=True)
    npy_path = os.path.join(results_dir, "new_candidate_positions.npy")
    np.save(npy_path, selected)
    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_candidates": len(selected),
        "method": "free_space_residual_correlation",
        "file": npy_path,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(results_dir, "new_candidates_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npy_path
