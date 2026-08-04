# -*- coding: utf-8 -*-
"""
脚本 1/4：生成环境感知传递矩阵 T_env
=====================================

功能：
- 若 Output_ESR/T_env.npz 已存在且模型标识匹配，则直接复用；
- 否则按 fallback（自由空间混合偶极子）或 feko（FEKO 单元偶极子仿真）
  生成 T_env 并保存。

用法：
    python nn_refinement/generate_T_env.py --mode fallback
    python nn_refinement/generate_T_env.py --mode feko --feko-bin D:/FEKO2025/feko/bin
"""

import argparse
import datetime
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402
import nn_common  # noqa: E402


def build_fallback_tenv():
    """
    构建自由空间回退 T_env，并保存为 Output_ESR/T_env.npz。
    """
    planes = esr.load_all_planes()
    src_positions = esr.build_candidate_grid()

    sub_coords_list = []
    sub_H_list = []
    subsample_indices_list = []
    plane_offsets = []
    offset = 0
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep_idx = esr.compute_subsample_indices(
            len(mag), mag, esr.OBS_SUBSAMPLE_STEP, esr.PEAK_KEEP_FRACTION
        )
        subsample_indices_list.append(keep_idx)
        sub_coords_list.append(plane["coords"][keep_idx])
        sub_H_list.append(plane["H"][keep_idx])
        plane_offsets.append(offset)
        offset += len(keep_idx)

    sub_coords = np.vstack(sub_coords_list)
    T = esr.build_free_space_T(sub_coords, src_positions)

    np.savez_compressed(
        esr.T_ENV_PATH,
        T=T,
        obs_coords=sub_coords,
        src_coords=src_positions,
        model=esr.MODEL_TAG,
        moment_component_indices=np.tile([0, 1, 2], len(src_positions)),
        component_types=np.tile(["M", "M", "E"], len(src_positions)),
        frequency_hz=esr.FREQ_HZ,
        source="free_space_fallback",
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        plane_offsets=np.array(plane_offsets),
        subsample_step=esr.OBS_SUBSAMPLE_STEP,
    )
    print(f"T_env 已保存: {esr.T_ENV_PATH} (shape={T.shape})")


def main():
    parser = argparse.ArgumentParser(description="生成环境感知传递矩阵 T_env")
    parser.add_argument("--mode", choices=["fallback", "feko"],
                        default="fallback")
    parser.add_argument("--feko-bin",
                        default=r"D:\FEKO2025\feko\bin")
    parser.add_argument(
        "--positions-txt",
        default=nn_common.INITIAL_DIPOLES_TXT,
        help="FEKO 模式使用的偶极子位置文件（默认取初步偶极子阵列）",
    )
    parser.add_argument("--force", action="store_true",
                        help="忽略已有 T_env，强制重新生成")
    parser.add_argument("--mpi-procs", type=int, default=6,
                        help="单个 FEKO 工程的 MPI 进程数（默认 6）")
    parser.add_argument("--jobs", type=int, default=2,
                        help="并发运行的 FEKO 工程数（默认 2）")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="只生成前若干列，用于冒烟测试")
    args = parser.parse_args()

    os.makedirs(esr.OUTPUT_DIR, exist_ok=True)

    # FEKO 模式固定使用初步偶极子阵列的位置（数量和位置不随训练变化）；
    # fallback 模式仍使用全候选网格，用于自由空间粗选支持集。
    if args.mode == "feko":
        positions, _ = nn_common.load_initial_dipoles(args.positions_txt)
    else:
        positions = esr.build_candidate_grid()
    print(f"偶极子位置数量: {len(positions)}")

    # 已存在且模型标识匹配时直接复用；
    # FEKO 模式只复用 FEKO 生成的环境感知矩阵，且源位置必须一致。
    if os.path.exists(esr.T_ENV_PATH) and not args.force:
        try:
            data = np.load(esr.T_ENV_PATH, allow_pickle=True)
            reusable = "model" in data and str(data["model"]) == esr.MODEL_TAG
            if reusable and args.mode == "feko":
                reusable = (
                    "source" in data
                    and str(data["source"]) == "feko_unit_dipole"
                    and "src_coords" in data
                    and data["src_coords"].shape == positions.shape
                    and np.allclose(data["src_coords"], positions)
                )
            if reusable:
                print(f"复用已有 T_env: shape={data['T'].shape}, "
                      f"source={data['source']}")
                return
        except Exception as exc:
            print("已有 T_env 无法读取，将重新生成:", exc)

    if args.mode == "feko":
        import feko_esr_tools

        planes = esr.load_all_planes()
        subsample_indices = []
        for plane in planes:
            mag = np.abs(plane["H"]).max(axis=1)
            subsample_indices.append(esr.compute_subsample_indices(
                len(mag), mag, esr.OBS_SUBSAMPLE_STEP, esr.PEAK_KEEP_FRACTION
            ))
        # 模板优先使用干净备份；若没有备份则用 Model_pred.pre 自身。
        template_pre = os.path.join(
            PROJECT_ROOT, "Model_EQ_Surce", "Model_pred.pre"
        )
        if os.path.exists(template_pre + ".backup"):
            template_pre = template_pre + ".backup"
        feko_esr_tools.generate_T_env_with_feko(
            src_positions=positions,
            plane_files=[os.path.join(esr.DATA_DIR, p["file"])
                         for p in esr.PLANE_DEFS],
            template_pre_path=template_pre,
            output_dir=esr.OUTPUT_DIR,
            feko_bin=args.feko_bin,
            subsample_step=esr.OBS_SUBSAMPLE_STEP,
            subsample_indices=subsample_indices,
            max_runs=args.max_runs,
            mpi_procs=args.mpi_procs,
            concurrent_jobs=args.jobs,
        )
    else:
        build_fallback_tenv()


if __name__ == "__main__":
    main()
