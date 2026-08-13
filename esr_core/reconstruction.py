# -*- coding: utf-8 -*-
"""重构主流程：T_env 生成、稀疏求解、新增候选与全平面最小二乘。"""

import datetime
import json
import os

import numpy as np

from . import candidates, config, fields, io, solver, tenv, validation


def _unique_positions(positions):
    unique = []
    for p in positions:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in unique):
            unique.append(p)
    return np.asarray(unique)


def _cv_split(sub_plane_offsets, sub_plane_sizes, b_len):
    rng = np.random.default_rng(config.CV_SEED)
    rows_train = []
    rows_test = []
    for pi, n_plane in enumerate(sub_plane_sizes):
        start = sub_plane_offsets[pi]
        idx = np.arange(n_plane)
        rng.shuffle(idx)
        n_test = max(1, int(round(config.CV_TEST_FRACTION * n_plane)))
        obs_test = start + idx[:n_test]
        obs_train = start + idx[n_test:]
        rows_test.append(np.concatenate([
            3 * obs_test, 3 * obs_test + 1, 3 * obs_test + 2
        ]))
        rows_train.append(np.concatenate([
            3 * obs_train, 3 * obs_train + 1, 3 * obs_train + 2
        ]))
    return np.concatenate(rows_train), np.concatenate(rows_test)


def _solve_initial_model(sub_tenv, b, src_positions, max_support):
    T = np.asarray(sub_tenv["T"])
    offsets = np.asarray(sub_tenv["plane_offsets"], dtype=int)
    plane_sizes = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
    plane_sizes.append(len(b) // 3 - offsets[-1])
    rows_train, rows_test = _cv_split(offsets, plane_sizes, len(b))

    _, _, _, val_errors = solver.group_omp_run(
        T[rows_train], b[rows_train], max_support,
        T[rows_test], b[rows_test],
    )
    n_dipoles = solver.choose_support_size(val_errors)
    print(f"交叉验证选定偶极子数量: {n_dipoles}")

    selected, x_all, train_errors, _ = solver.group_omp_run(
        T, b, n_dipoles
    )
    positions = src_positions[selected].copy()
    moments = x_all.reshape(-1, 3)
    print(f"初始支持集偶极子数: {len(positions)}, "
          f"训练相对误差: {train_errors[-1] * 100:.4f}%")

    positions, moments = solver.merge_prune_dipoles(positions, moments)
    if len(positions) > 0:
        cols = tenv.columns_for_positions(src_positions, positions)
        moments = solver.refit_moments(T, cols, b)
    print(f"合并/裁剪后初始偶极子数量: {len(positions)}")
    return positions, moments


def _run_expansion(planes, full_tenv, positions, moments,
                   src_positions, sub_tenv, b_sub, results_dir,
                   n_candidates, feko_bin, mpi_procs,
                   concurrent_jobs, max_runs, force_tenv):
    """误差超阈值时，筛选新增候选并做全平面最小二乘。"""
    sub_T = np.asarray(sub_tenv["T"])
    sub_obs = np.asarray(sub_tenv["obs_coords"])
    cols = tenv.columns_for_positions(src_positions, positions)
    residual = b_sub - sub_T[:, cols] @ moments.ravel()
    residual_ratio = np.linalg.norm(residual) / np.linalg.norm(b_sub)
    print(f"当前重构残差范数占比: {residual_ratio * 100:.4f}%")

    new_positions = candidates.select_new_candidates(
        sub_obs, residual, src_positions,
        n_candidates=n_candidates,
    )
    print(f"已筛选 {len(new_positions)} 个新增候选位置")
    candidates.save_new_candidates(results_dir, new_positions)

    combined = _unique_positions(np.vstack([positions, new_positions]))
    print(f"扩展后位置总数: {len(combined)}")

    # 只对缺失位置补做 FEKO 仿真，已有位置直接复用全平面 T_env。
    full_tenv = tenv.generate_full_tenv(
        combined, planes, results_dir,
        feko_bin=feko_bin,
        mpi_procs=mpi_procs,
        concurrent_jobs=concurrent_jobs,
        max_runs=max_runs,
        force=force_tenv,
    )
    T_full = np.asarray(full_tenv["T"])
    src_full = np.asarray(full_tenv["src_coords"])
    b_full = fields.concatenate_fields(planes)
    cols_all = tenv.columns_for_positions(src_full, combined)
    m_all, _, _, _ = np.linalg.lstsq(T_full[:, cols_all], b_full, rcond=None)

    positions, moments = solver.merge_prune_dipoles(
        combined, m_all.reshape(-1, 3)
    )
    if len(positions) > 0:
        cols_prune = tenv.columns_for_positions(src_full, positions)
        moments = solver.refit_moments(T_full, cols_prune, b_full)
    print(f"全平面重构后偶极子数量: {len(positions)}")
    return positions, moments, full_tenv


def run_reconstruction(results_dir=config.RESULTS_DIR,
                       feko_bin=config.DEFAULT_FEKO_BIN,
                       mpi_procs=config.MPI_PROCS,
                       concurrent_jobs=config.CONCURRENT_JOBS,
                       max_support=config.MAX_SUPPORT,
                       max_runs=None,
                       force_tenv=False,
                       candidate_spacing=config.CANDIDATE_SPACING,
                       n_candidates=config.NEW_CANDIDATES,
                       error_threshold=config.NEW_CANDIDATE_ERROR_THRESHOLD_PERCENT):
    """执行完整等效源重构，并把结果写入 Results 目录。"""
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("步骤 1/4：读取真实扫描磁场")
    planes = fields.load_real_planes(config.RECONSTRUCTION_PLANES)

    print("=" * 60)
    print("步骤 2/4：生成/复用全平面 FEKO T_env")
    src_positions = tenv.build_candidate_grid(spacing=candidate_spacing)
    full_tenv = tenv.generate_full_tenv(
        src_positions, planes, results_dir,
        feko_bin=feko_bin,
        mpi_procs=mpi_procs,
        concurrent_jobs=concurrent_jobs,
        max_runs=max_runs,
        force=force_tenv,
    )

    print("=" * 60)
    print("步骤 3/4：Group OMP 稀疏重构")
    sub_coords, b_sub, sub_offsets, subsample_indices = fields.subsample_planes(
        planes
    )
    sub_tenv = tenv.derive_subsampled_tenv(full_tenv, subsample_indices)
    if not np.allclose(sub_coords, sub_tenv["obs_coords"]):
        raise RuntimeError("抽稀观测点与 T_env 不一致，请检查抽稀参数")

    positions, moments = _solve_initial_model(
        sub_tenv, b_sub, src_positions, max_support
    )
    initial_n_dipoles = len(positions)
    initial_error = validation.tenv_model_error(
        planes, full_tenv, positions, moments
    )["overall"]["rel_l2_error_percent"]
    print(f"初始模型 T_env 整体相对 L2 误差: {initial_error:.4f}%")

    print("=" * 60)
    print("步骤 4/4：误差评估与新增候选决策")
    expanded = False
    if initial_error > error_threshold:
        print(f"误差 {initial_error:.4f}% 大于阈值 {error_threshold:.4f}%，"
              f"执行新增候选偶极子")
        positions, moments, full_tenv = _run_expansion(
            planes, full_tenv, positions, moments,
            src_positions, sub_tenv, b_sub, results_dir,
            n_candidates=n_candidates,
            feko_bin=feko_bin,
            mpi_procs=mpi_procs,
            concurrent_jobs=concurrent_jobs,
            max_runs=max_runs,
            force_tenv=force_tenv,
        )
        expanded = True
    else:
        print(f"误差 {initial_error:.4f}% 小于等于阈值 "
              f"{error_threshold:.4f}%，跳过新增候选步骤")

    final_error = validation.tenv_model_error(
        planes, full_tenv, positions, moments
    )["overall"]["rel_l2_error_percent"]
    print(f"最终模型 T_env 整体相对 L2 误差: {final_error:.4f}%")

    dipole_path = os.path.join(results_dir, "equivalent_dipoles.txt")
    io.save_dipoles_txt(dipole_path, positions, moments)
    _save_metadata(results_dir, {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": "group_omp + cv + optional_new_candidates + full_plane_lstsq",
        "initial_n_dipoles": initial_n_dipoles,
        "expanded": expanded,
        "final_n_dipoles": len(positions),
        "initial_tenv_error_percent": round(initial_error, 6),
        "final_tenv_error_percent": round(final_error, 6),
        "error_threshold_percent": error_threshold,
        "n_new_candidates": n_candidates,
        "model": config.MODEL_TAG,
    })
    _save_report(results_dir, {
        "initial_error": initial_error,
        "final_error": final_error,
        "expanded": expanded,
        "n_dipoles": len(positions),
    })
    print(f"等效偶极子阵列已保存: {dipole_path}")
    return dipole_path


def _save_metadata(results_dir, meta):
    with open(os.path.join(results_dir, "dipole_metadata.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _save_report(results_dir, data):
    lines = [
        "等效源重构报告",
        "=" * 50,
        f"生成时间: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"初始模型 T_env 整体相对 L2 误差: {data['initial_error']:.4f}%",
        f"最终模型 T_env 整体相对 L2 误差: {data['final_error']:.4f}%",
        f"执行新增候选偶极子: {'是' if data['expanded'] else '否'}",
        f"最终偶极子数量: {data['n_dipoles']}",
        "",
        "说明：这里的误差使用全平面 T_env 线性计算，不调用 FEKO；",
        "最终 FEKO 验证请运行 validata.py。",
    ]
    with open(os.path.join(results_dir, "reconstruction_report.txt"),
              "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
