# -*- coding: utf-8 -*-
"""偶极子求解：Group OMP、交叉验证选数、合并与裁剪。"""

import numpy as np

from . import config


def group_omp_run(T, b, n_groups, T_val=None, b_val=None, gsize=3):
    """
    组正交匹配追踪。

    组定义为同一个位置的 3 个偶极矩分量。每次迭代选择与残差相关性最强
    的一组，再对已选组做最小二乘并更新残差。
    """
    n_sites = T.shape[1] // gsize
    n_groups = min(n_groups, n_sites)
    residual = b.copy()
    selected = []
    train_errors = []
    val_errors = []
    x = None

    for _ in range(n_groups):
        corr = np.linalg.norm(
            (T.conj().T @ residual).reshape(-1, gsize), axis=1
        )
        best_group = int(np.argmax(corr))
        if best_group not in selected:
            selected.append(best_group)

        cols = _group_columns(selected, gsize)
        T_S = T[:, cols]
        x, _, _, _ = np.linalg.lstsq(T_S, b, rcond=None)
        residual = b - T_S @ x
        train_errors.append(
            np.linalg.norm(residual) / np.linalg.norm(b)
        )

        if T_val is not None and b_val is not None:
            val_error = (
                np.linalg.norm(b_val - T_val[:, cols] @ x)
                / np.linalg.norm(b_val)
            )
            val_errors.append(val_error)

    return selected, x, np.asarray(train_errors), np.asarray(val_errors)


def _group_columns(groups, gsize=3):
    cols = []
    for g in groups:
        cols.extend(range(g * gsize, (g + 1) * gsize))
    return np.asarray(cols, dtype=int)


def choose_support_size(val_errors, min_support=config.MIN_SUPPORT,
                        tolerance=0.02):
    """
    在验证误差最小值的容差范围内选择更小的支持集。
    """
    if len(val_errors) == 0:
        return min_support
    best_idx = int(np.argmin(val_errors))
    best_val = val_errors[best_idx]
    candidates = [
        i for i, e in enumerate(val_errors)
        if e <= best_val * (1.0 + tolerance)
    ]
    return max(min_support, candidates[0] + 1)


def merge_prune_dipoles(positions, moments,
                        merge_dist=config.MERGE_DIST,
                        rel_threshold=config.PRUNE_REL_THRESHOLD):
    """删除弱偶极子，并按距离合并近距离偶极子。"""
    if len(positions) == 0:
        return positions, moments

    mag = np.abs(moments).max(axis=1)
    keep = mag >= rel_threshold * mag.max()
    positions = positions[keep]
    moments = moments[keep]

    order = np.argsort(np.abs(moments).max(axis=1))[::-1]
    positions = positions[order]
    moments = moments[order]

    kept_pos = []
    kept_mom = []
    used = np.zeros(len(positions), dtype=bool)
    for i in range(len(positions)):
        if used[i]:
            continue
        used[i] = True
        p0 = positions[i]
        m0 = moments[i].copy()
        for j in range(i + 1, len(positions)):
            if used[j]:
                continue
            if np.linalg.norm(positions[j] - p0) <= merge_dist:
                m0 = m0 + moments[j]
                used[j] = True
        kept_pos.append(p0)
        kept_mom.append(m0)

    return np.asarray(kept_pos), np.asarray(kept_mom)


def refit_moments(T, cols, b):
    """用指定列的最小二乘解作为偶极矩，返回 (N, 3) 复矩阵。"""
    m, _, _, _ = np.linalg.lstsq(T[:, cols], b, rcond=None)
    return m.reshape(-1, 3)
