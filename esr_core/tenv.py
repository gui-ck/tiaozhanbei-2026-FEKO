# -*- coding: utf-8 -*-
"""候选网格、自由空间格林函数与 FEKO 全平面 T_env 生成。"""

import concurrent.futures
import datetime
import json
import os
import time

import numpy as np

from . import config, feko, fields


def get_candidate_bounds(box=config.BOX_BOUNDS,
                         shrink=config.CANDIDATE_SHRINK_RATIO):
    """候选偶极子范围：机箱各方向居中 shrink 比例。"""
    bounds = {}
    for axis, (lo, hi) in box.items():
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * shrink
        bounds[axis] = (center - half, center + half)
    return bounds


def build_candidate_grid(spacing=config.CANDIDATE_SPACING):
    """生成三维规则候选网格。"""
    bounds = get_candidate_bounds()
    xs = np.arange(bounds["x"][0], bounds["x"][1] + spacing / 2, spacing)
    ys = np.arange(bounds["y"][0], bounds["y"][1] + spacing / 2, spacing)
    zs = np.arange(bounds["z"][0], bounds["z"][1] + spacing / 2, spacing)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    print(f"候选网格: x {len(xs)} 点, y {len(ys)} 点, z {len(zs)} 点, "
          f"共 {len(points)} 个位置")
    return points


def _build_magnetic_T(obs_points, src_points, comps, freq=config.FREQ_HZ):
    """自由空间磁偶极子格林函数矩阵。"""
    k = 2.0 * np.pi * freq / config.C0
    n_obs = len(obs_points)
    n_src = len(src_points)

    R_vec = obs_points[:, None, :] - src_points[None, :, :]
    R = np.linalg.norm(R_vec, axis=2)
    R[R == 0.0] = 1e-12
    R_hat = R_vec / R[:, :, None]
    R_hat_tensor = R_hat[:, :, :, None] * R_hat[:, :, None, :]
    I3 = np.eye(3)

    term1 = (k**2 / R[:, :, None, None]) * (
        I3[None, None, :, :] - R_hat_tensor
    )
    term2 = (1.0 / R[:, :, None, None] ** 3
             + 1j * k / R[:, :, None, None] ** 2) * (
        3.0 * R_hat_tensor - I3[None, None, :, :]
    )
    G = (np.exp(-1j * k * R[:, :, None, None]) / (4.0 * np.pi)) * (
        term1 + term2
    )
    G = G[:, :, :, list(comps)]
    return G.transpose(0, 2, 1, 3).reshape(n_obs * 3, n_src * len(comps))


def _build_electric_T(obs_points, src_points, comps, freq=config.FREQ_HZ):
    """自由空间电偶极子格林函数矩阵。"""
    k = 2.0 * np.pi * freq / config.C0
    n_obs = len(obs_points)
    n_src = len(src_points)

    R_vec = obs_points[:, None, :] - src_points[None, :, :]
    R = np.linalg.norm(R_vec, axis=2)
    R[R == 0.0] = 1e-12
    R_hat = R_vec / R[:, :, None]
    s = (np.exp(-1j * k * R) / (4.0 * np.pi)
         * (-1j * k / R - 1.0 / R**2))

    C = np.zeros((n_obs, n_src, 3, 3), dtype=complex)
    C[:, :, 0, 1] = R_hat[:, :, 2]
    C[:, :, 0, 2] = -R_hat[:, :, 1]
    C[:, :, 1, 0] = -R_hat[:, :, 2]
    C[:, :, 1, 2] = R_hat[:, :, 0]
    C[:, :, 2, 0] = R_hat[:, :, 1]
    C[:, :, 2, 1] = -R_hat[:, :, 0]

    G = np.zeros((n_obs, n_src, 3, 3), dtype=complex)
    for c_idx in comps:
        G[:, :, :, c_idx] = s[:, :, None] * C[:, :, :, c_idx]
    G = G[:, :, :, list(comps)]
    return G.transpose(0, 2, 1, 3).reshape(n_obs * 3, n_src * len(comps))


def build_free_space_T(obs_points, src_points, freq=config.FREQ_HZ):
    """
    构建混合偶极子自由空间传递矩阵。

    每个位置固定 3 列：Mx、My（磁偶极子）、Pz（电偶极子）。
    """
    magnetic = _build_magnetic_T(obs_points, src_points, (0, 1), freq)
    electric = _build_electric_T(obs_points, src_points, (2,), freq)
    return np.hstack([magnetic, electric])


def columns_for_positions(src_coords, positions):
    """返回位置列表在 T_env 中对应的列索引。"""
    cols = []
    for pos in positions:
        match = np.where(np.all(np.isclose(src_coords, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"偶极子位置不在 T_env 中: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    return np.asarray(cols, dtype=int)


def load_tenv(path):
    """读取 T_env npz 并校验模型标识。"""
    if not os.path.exists(path):
        return None
    data = dict(np.load(path, allow_pickle=True))
    if "model" in data and str(data["model"]) != config.MODEL_TAG:
        raise ValueError(f"T_env 模型标识不匹配: {path}")
    return data


def _obs_coords_from_planes(planes):
    return np.vstack([p["coords"] for p in planes])


def _tenv_matches(data, src_coords, obs_coords):
    if data is None:
        return False
    if str(data.get("source", "")) != "feko_unit_dipole_full":
        return False
    if not np.allclose(np.asarray(data["src_coords"]), src_coords):
        return False
    return np.allclose(np.asarray(data["obs_coords"]), obs_coords)


def _read_unit_column(base, work_root, planes):
    """读取单个单元偶极子工程在全部扫描面上的磁场列。"""
    columns = []
    for plane in planes:
        request_name = plane["request_name"]
        hfe_path = os.path.join(work_root, f"{base}_{request_name}.hfe")
        if not os.path.exists(hfe_path):
            fallback = os.path.join(work_root, f"{base}.hfe")
            if os.path.exists(fallback):
                hfe_path = fallback
            else:
                raise FileNotFoundError(hfe_path)
        if os.path.getsize(hfe_path) == 0:
            raise RuntimeError(f"FEKO 磁场输出为空: {hfe_path}")
        _, H, _ = fields.read_field_file(hfe_path)
        columns.append(H.ravel())
    return np.concatenate(columns)


def generate_full_tenv(src_positions, planes, results_dir,
                       feko_bin=config.DEFAULT_FEKO_BIN,
                       mpi_procs=config.MPI_PROCS,
                       concurrent_jobs=config.CONCURRENT_JOBS,
                       max_runs=None, force=False):
    """
    对每个候选位置 x/y/z 分量运行一次 FEKO 单元偶极子仿真，
    保存全观测点 T_env，并支持逐列断点续跑。

    如果 Results 下已有可复用的 T_env_full，并且源位置一致，则直接复用；
    如果已有矩阵只覆盖部分位置，则只补做缺失位置，避免重复仿真。
    """
    os.makedirs(results_dir, exist_ok=True)
    obs_coords = _obs_coords_from_planes(planes)
    tenv_path = os.path.join(results_dir, "T_env_full.npz")

    existing = load_tenv(tenv_path)
    if existing is not None and not force and _tenv_matches(
        existing, src_positions, obs_coords
    ):
        print(f"复用已有全平面 T_env: {tenv_path}, shape={existing['T'].shape}")
        return existing

    if existing is not None and not force:
        if str(existing.get("source", "")) != "feko_unit_dipole_full":
            raise ValueError("已有 T_env_full 来源不正确，请使用 --force-tenv 重新生成")
        existing_src = np.asarray(existing["src_coords"])
        have_mask = np.array([
            np.any(np.all(np.isclose(existing_src, pos), axis=1))
            for pos in src_positions
        ])
        if have_mask.all():
            if not np.allclose(existing["obs_coords"], obs_coords):
                raise RuntimeError("已有 T_env 观测点与当前扫描面不一致")
            print(f"复用已有全平面 T_env: {tenv_path}, shape={existing['T'].shape}")
            return existing
        print(f"已有 T_env 覆盖 {have_mask.sum()}/{len(src_positions)} 个位置，"
              f"只补做缺失位置")
        missing_src = src_positions[~have_mask]
        new_tenv = _extend_tenv(
            existing, missing_src, planes, results_dir,
            feko_bin, mpi_procs, concurrent_jobs, max_runs,
        )
        new_tenv = _slice_tenv_to_positions(new_tenv, src_positions)
        _save_tenv(tenv_path, new_tenv)
        return new_tenv

    return _generate_from_scratch(
        src_positions, planes, results_dir, tenv_path,
        feko_bin, mpi_procs, concurrent_jobs, max_runs,
    )


def _generate_from_scratch(src_positions, planes, results_dir, tenv_path,
                           feko_bin, mpi_procs, concurrent_jobs, max_runs):
    """从零生成全平面 T_env。"""
    work_root = config.MODEL_EQ_DIR
    columns_dir = os.path.join(results_dir, "feko_tenv_columns")
    os.makedirs(columns_dir, exist_ok=True)
    n_obs = sum(len(p["coords"]) for p in planes)
    jobs = _build_jobs(len(src_positions), max_runs)
    progress_path = os.path.join(results_dir, "feko_tenv_progress.json")

    t_start = time.time()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(concurrent_jobs))
    ) as executor:
        futures = {
            executor.submit(
                _run_unit_job, s_idx, comp, src_positions, planes,
                work_root, columns_dir, feko_bin, mpi_procs,
            ): (s_idx, comp)
            for s_idx, comp in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            base, status = future.result()
            completed += 1
            elapsed = time.time() - t_start
            eta_h = elapsed / completed * (len(jobs) - completed) / 3600.0
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({
                    "completed": completed,
                    "total": len(jobs),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_hours": round(eta_h, 2),
                }, f, ensure_ascii=False, indent=2)
            print(f"[T_env] {completed}/{len(jobs)}: {base} ({status}) "
                  f"耗时 {elapsed:.0f}s")

    T = np.zeros((n_obs * 3, len(jobs)), dtype=complex)
    for col_idx, (s_idx, comp) in enumerate(jobs):
        base = _unit_case_base(s_idx, comp)
        T[:, col_idx] = np.load(
            os.path.join(columns_dir, base + ".npy")
        )

    plane_offsets = _plane_offsets(planes)
    tenv = {
        "T": T,
        "src_coords": src_positions,
        "obs_coords": _obs_coords_from_planes(planes),
        "plane_offsets": plane_offsets,
        "model": config.MODEL_TAG,
        "source": "feko_unit_dipole_full",
        "frequency_hz": config.FREQ_HZ,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _save_tenv(tenv_path, tenv)
    print(f"全平面 T_env 已保存: {tenv_path}, shape={T.shape}")
    return tenv


def _slice_tenv_to_positions(tenv_data, requested_src):
    """把 T_env 的列重排为 requested_src 的固定顺序。"""
    old_src = np.asarray(tenv_data["src_coords"])
    T = np.asarray(tenv_data["T"])
    if old_src.shape == requested_src.shape and np.allclose(old_src, requested_src):
        return tenv_data
    cols = []
    for pos in requested_src:
        match = np.where(np.all(np.isclose(old_src, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"T_env 中缺少位置: {pos}")
        idx = int(match[0])
        cols.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    tenv_data = dict(tenv_data)
    tenv_data["T"] = T[:, cols]
    tenv_data["src_coords"] = np.asarray(requested_src)
    return tenv_data


def _extend_tenv(existing, missing_src, planes, results_dir,
                 feko_bin, mpi_procs, concurrent_jobs, max_runs):
    """只对缺失位置补做单元偶极子仿真并拼接列。"""
    if len(missing_src) == 0:
        return existing
    old_T = np.asarray(existing["T"])
    old_src = np.asarray(existing["src_coords"])
    old_obs = np.asarray(existing["obs_coords"])
    if not np.allclose(old_obs, _obs_coords_from_planes(planes)):
        raise RuntimeError("已有 T_env 观测点与当前扫描面不一致，无法扩展")

    work_root = config.MODEL_EQ_DIR
    columns_dir = os.path.join(results_dir, "feko_tenv_columns")
    os.makedirs(columns_dir, exist_ok=True)
    jobs = _build_jobs(len(missing_src), max_runs)
    n_obs = len(old_obs)
    new_T = np.zeros((n_obs * 3, len(jobs)), dtype=complex)

    for col_idx, (s_idx, comp) in enumerate(jobs):
        base = _unit_case_base(len(old_src) + s_idx, comp)
        col_path = os.path.join(columns_dir, base + ".npy")
        if os.path.exists(col_path):
            new_T[:, col_idx] = np.load(col_path)
            continue
        col = _run_unit_job(
            len(old_src) + s_idx, comp, missing_src,
            planes, work_root, columns_dir, feko_bin, mpi_procs,
        )
        new_T[:, col_idx] = np.load(col_path)
        print(f"[Extend] {base} done: {col[1]}")

    new_src = np.vstack([old_src, missing_src])
    return {
        "T": np.hstack([old_T, new_T]),
        "src_coords": new_src,
        "obs_coords": old_obs,
        "plane_offsets": np.asarray(existing["plane_offsets"]),
        "model": config.MODEL_TAG,
        "source": "feko_unit_dipole_full",
        "frequency_hz": config.FREQ_HZ,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _build_jobs(n_sites, max_runs=None):
    jobs = []
    for s_idx in range(n_sites):
        for comp in ("x", "y", "z"):
            if max_runs is not None and len(jobs) >= max_runs:
                break
            jobs.append((s_idx, comp))
        if max_runs is not None and len(jobs) >= max_runs:
            break
    return jobs


def _unit_case_base(site_idx, comp):
    return f"tenv_full_s{site_idx:04d}_{comp}"


def _run_unit_job(site_idx, comp, src_positions, planes, work_root,
                  columns_dir, feko_bin, mpi_procs):
    """运行一个单元偶极子仿真并保存该列。"""
    base = _unit_case_base(site_idx, comp)
    col_path = os.path.join(columns_dir, base + ".npy")
    if os.path.exists(col_path):
        return base, "resume"

    feko.cleanup_case_files(work_root, base)
    pre_path = os.path.join(work_root, base + ".pre")
    src_line = (
        feko.make_a6_line(src_positions[site_idx], comp, 1.0, 0.0)
        if comp in ("x", "y")
        else feko.make_a5_line(src_positions[site_idx], comp, 1.0, 0.0)
    )
    feko.insert_sources_into_pre(config.TEMPLATE_PRE, [src_line], pre_path)
    log_path = pre_path + ".log"

    ok = feko.run_feko_case_parallel(
        pre_path, feko_bin, mpi_procs=mpi_procs, log_path=log_path
    )
    if not ok:
        feko.cleanup_case_files(work_root, base)
        feko.insert_sources_into_pre(config.TEMPLATE_PRE, [src_line], pre_path)
        ok = feko.run_feko_case_parallel(
            pre_path, feko_bin, mpi_procs=mpi_procs, log_path=log_path
        )
    if not ok:
        raise RuntimeError(f"单元偶极子仿真失败: {base}")

    col = _read_unit_column(base, work_root, planes)
    np.save(col_path, col)
    feko.cleanup_case_files(work_root, base)
    return base, "done"


def _plane_offsets(planes):
    offsets = []
    offset = 0
    for plane in planes:
        offsets.append(offset)
        offset += len(plane["coords"])
    return np.asarray(offsets, dtype=int)


def _save_tenv(path, tenv):
    np.savez_compressed(
        path,
        T=tenv["T"],
        src_coords=tenv["src_coords"],
        obs_coords=tenv["obs_coords"],
        plane_offsets=tenv["plane_offsets"],
        model=tenv["model"],
        source=tenv["source"],
        frequency_hz=tenv["frequency_hz"],
        created_at=tenv["created_at"],
    )


def derive_subsampled_tenv(full_tenv, subsample_indices):
    """从全平面 T_env 按抽稀索引取出重构用子矩阵。"""
    T = np.asarray(full_tenv["T"])
    obs_coords = np.asarray(full_tenv["obs_coords"])
    plane_offsets = np.asarray(full_tenv["plane_offsets"], dtype=int)
    if len(plane_offsets) != len(subsample_indices):
        raise ValueError("全平面 T_env 平面数与抽稀索引不匹配")

    rows = []
    keep_obs = []
    sub_offsets = []
    offset = 0
    for pi, keep in enumerate(subsample_indices):
        global_obs = plane_offsets[pi] + np.asarray(keep, dtype=int)
        keep_obs.append(global_obs)
        sub_offsets.append(offset)
        offset += len(global_obs)
        rows.append(np.concatenate([
            3 * global_obs, 3 * global_obs + 1, 3 * global_obs + 2
        ]))
    rows = np.sort(np.concatenate(rows))
    keep_obs = np.sort(np.concatenate(keep_obs))
    return {
        "T": T[rows],
        "obs_coords": obs_coords[keep_obs],
        "plane_offsets": np.asarray(sub_offsets, dtype=int),
    }
