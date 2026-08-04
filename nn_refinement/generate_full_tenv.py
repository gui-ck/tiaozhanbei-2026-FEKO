# -*- coding: utf-8 -*-
"""
全观测点 FEKO T_env 生成脚本（方案 1 + 方案 2）
================================================

目的：
- 当前 T_env.npz 只保存了抽稀后的 19663 个观测点；
- 方案 2 要求使用全部 81903 个近场扫描点做最小二乘；
- 方案 1 要求在原有 40 个偶极子位置之外补充新的候选位置；
- 本脚本为“原有位置 + 新候选位置”逐个单元偶极子运行 FEKO，
  并把三个扫描面的全部观测点磁场保存为全观测点 T_env。

流程：
1. 读取当前 40 个偶极子位置和 new_candidate_positions.npy；
2. 合并、去重后得到待仿真位置表；
3. 每个位置 x/y/z 分量生成一个单源 .pre（x/y 为 A6 磁偶极子，
   z 为 A5 电偶极子），在 Model_EQ_Surce 中运行 FEKO；
4. 读取三个完整 .hfe，按 xoy/xoz/yoz 顺序拼接成全观测点列；
5. 每列单独保存为 npy，支持断点续跑；
6. 全部完成后汇总为 Output_ESR/T_env_full.npz。

用法：
    python nn_refinement/generate_full_tenv.py \
        --positions Output_ESR/equivalent_dipoles.txt \
        --new-candidates Output_ESR/new_candidate_positions.npy \
        --mpi-procs 6 --jobs 2
"""

import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402
import feko_esr_tools as fek  # noqa: E402
import nn_common  # noqa: E402


def case_base(site_idx, comp):
    """全观测点 T_env 的工程名，与旧 T_env 工程名区分开。"""
    return f"tenvf_s{site_idx:04d}_{comp}"


def cleanup_case(work_root, base):
    """清理单个单元偶极子工程的临时文件。"""
    for pattern in (base + ".*", base + "_NearField_*.hfe"):
        for f in glob.glob(os.path.join(work_root, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def read_full_column(base, work_root, plane_files):
    """读取一个单元工程生成的三个完整 .hfe，拼接成一列。"""
    columns = []
    for pf in plane_files:
        real_base = os.path.splitext(os.path.basename(pf))[0]
        request_name = real_base.split("NearField_")[-1]
        hfe_path = os.path.join(
            work_root, f"{base}_NearField_{request_name}.hfe"
        )
        if not os.path.exists(hfe_path):
            raise FileNotFoundError(hfe_path)
        if os.path.getsize(hfe_path) == 0:
            raise RuntimeError(f"hfe 文件为空: {hfe_path}")
        _, H_full, _ = fek.read_hfe_subsampled(hfe_path, step=1)
        columns.append(H_full.ravel())
    return np.concatenate(columns)


def main():
    parser = argparse.ArgumentParser(description="生成全观测点 FEKO T_env")
    parser.add_argument("--positions",
                        default=os.path.join(esr.OUTPUT_DIR,
                                             "equivalent_dipoles.txt"))
    parser.add_argument("--new-candidates",
                        default=os.path.join(esr.OUTPUT_DIR,
                                             "new_candidate_positions.npy"))
    parser.add_argument("--feko-bin", default=r"D:\FEKO2025\feko\bin")
    parser.add_argument("--mpi-procs", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    # 1. 合并位置：当前 40 个 + 新候选 30 个，去掉重复
    positions, _ = nn_common.load_initial_dipoles(args.positions)
    new_candidates = np.load(args.new_candidates)
    combined = np.vstack([positions, new_candidates])
    unique_pos = []
    for p in combined:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in unique_pos):
            unique_pos.append(p)
    src_positions = np.asarray(unique_pos)
    print(f"待仿真位置: {len(src_positions)} 个 "
          f"({len(positions)} 原有 + {len(new_candidates)} 新增)")

    plane_files = [
        os.path.join(esr.DATA_DIR, p["file"]) for p in esr.PLANE_DEFS
    ]
    obs_coords_list = []
    plane_offsets = []
    offset = 0
    for pf in plane_files:
        coords_full, _, _ = esr.read_hfe(pf)
        obs_coords_list.append(coords_full)
        plane_offsets.append(offset)
        offset += len(coords_full)
    obs_coords = np.vstack(obs_coords_list)
    n_obs = len(obs_coords)
    print(f"全观测点总数: {n_obs}, plane_offsets={plane_offsets}")

    work_root = os.path.join(PROJECT_ROOT, "Model_EQ_Surce")
    columns_dir = os.path.join(esr.OUTPUT_DIR, "feko_full_tenv_columns")
    os.makedirs(columns_dir, exist_ok=True)

    comp_names = ["x", "y", "z"]
    jobs = []
    for s_idx, pos in enumerate(src_positions):
        for comp in comp_names:
            if args.max_runs is not None and len(jobs) >= args.max_runs:
                break
            jobs.append((s_idx, comp))
        if args.max_runs is not None and len(jobs) >= args.max_runs:
            break
    n_cols = len(jobs)
    print(f"仿真任务总数: {n_cols}")

    def run_one(site_idx, comp):
        base = case_base(site_idx, comp)
        col_path = os.path.join(columns_dir, base + ".npy")
        if os.path.exists(col_path):
            return base, "resume"

        cleanup_case(work_root, base)
        pre_path = os.path.join(work_root, base + ".pre")
        src_line = (
            fek.make_a6_line(src_positions[site_idx], comp, 1.0, 0.0)
            if comp in ("x", "y")
            else fek.make_a5_line(src_positions[site_idx], comp, 1.0, 0.0)
        )
        # 使用干净的 Model_pred.pre.clean 模板，避免残留源
        template = os.path.join(
            PROJECT_ROOT, "Model_EQ_Surce", "Model_pred.pre.clean"
        )
        fek.insert_sources_into_pre(template, [src_line], pre_path)

        log_path = pre_path + ".log"
        ok = fek.run_feko_case_parallel(
            pre_path, args.feko_bin,
            mpi_procs=args.mpi_procs, log_path=log_path,
        )
        if not ok:
            cleanup_case(work_root, base)
            fek.insert_sources_into_pre(template, [src_line], pre_path)
            ok = fek.run_feko_case_parallel(
                pre_path, args.feko_bin,
                mpi_procs=args.mpi_procs, log_path=log_path,
            )
        if not ok:
            raise RuntimeError(f"仿真失败: {base}")

        col = read_full_column(base, work_root, plane_files)
        np.save(col_path, col)
        cleanup_case(work_root, base)
        return base, "done"

    t_start = time.time()
    completed = 0
    progress_path = os.path.join(esr.OUTPUT_DIR,
                                 "feko_full_tenv_progress.json")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.jobs)
    ) as executor:
        futures = {
            executor.submit(run_one, s_idx, comp): (s_idx, comp)
            for s_idx, comp in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            base, status = future.result()
            completed += 1
            elapsed = time.time() - t_start
            eta_h = elapsed / completed * (n_cols - completed) / 3600.0
            progress = {
                "completed": completed,
                "total": n_cols,
                "elapsed_seconds": round(elapsed, 1),
                "eta_hours": round(eta_h, 2),
                "updated_at": datetime.datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(
                f"[FullTenv] {completed}/{n_cols}: {base} ({status}) "
                f"耗时 {elapsed:.0f}s, 预计剩余 {eta_h:.1f}h"
            )

    T = np.zeros((n_obs * 3, n_cols), dtype=complex)
    for col_idx, (s_idx, comp) in enumerate(jobs):
        base = case_base(s_idx, comp)
        col = np.load(os.path.join(columns_dir, base + ".npy"))
        T[:, col_idx] = col

    out_name = (
        "T_env_full.npz"
        if args.max_runs is None
        else f"T_env_full_partial_{args.max_runs}.npz"
    )
    out_path = os.path.join(esr.OUTPUT_DIR, out_name)
    np.savez_compressed(
        out_path,
        T=T,
        src_coords=src_positions,
        obs_coords=obs_coords,
        model=esr.MODEL_TAG,
        moment_component_indices=np.tile([0, 1, 2], len(src_positions)),
        component_types=np.tile(["M", "M", "E"], len(src_positions)),
        frequency_hz=esr.FREQ_HZ,
        source="feko_unit_dipole_full",
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        plane_offsets=np.array(plane_offsets, dtype=int),
        mpi_procs=args.mpi_procs,
        concurrent_jobs=args.jobs,
    )
    print(f"全观测点 T_env 已保存: {out_path}, shape={T.shape}")


if __name__ == "__main__":
    main()
