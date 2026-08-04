# -*- coding: utf-8 -*-
"""
在 Model_EQ_Surce 中就地运行 FEKO 偶极子阵列验证
=================================================

流程：
1. 读取指定的偶极子阵列 TXT（默认 nn_refinement/equivalent_dipoles_refined.txt）；
2. 把 Mx/My 转成 A6 磁偶极子卡片、Pz 转成 A5 电偶极子卡片；
3. 备份并直接修改 Model_EQ_Surce/Model_pred.pre；
4. 在 Model_EQ_Surce 目录运行 prefeko + feko；
5. 读取仿真产生的 3 个 .hfe（NearField_xoy/xoz/yoz）；
6. 与真实扫描磁场逐点比较，输出误差 JSON、中文报告和场图。

用法：
    python nn_refinement/run_feko_validation_inplace.py
    python nn_refinement/run_feko_validation_inplace.py --dipoles Output_ESR/equivalent_dipoles.txt
"""

import argparse
import json
import os
import shutil
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


def main():
    parser = argparse.ArgumentParser(description="FEKO 就地验证偶极子阵列")
    parser.add_argument(
        "--dipoles",
        default=os.path.join(nn_common.OUTPUT_ROOT,
                             "equivalent_dipoles_refined.txt"),
        help="偶极子阵列 TXT 路径",
    )
    parser.add_argument(
        "--feko-bin",
        default=r"D:\FEKO2025\feko\bin",
        help="FEKO bin 目录",
    )
    parser.add_argument(
        "--mpi-procs",
        type=int,
        default=6,
        help="FEKO 并行求解进程数（默认 6）",
    )
    parser.add_argument(
        "--out-suffix",
        default="validation",
        help="输出子目录后缀，例如 initial/refined，避免两次验证互相覆盖",
    )
    args = parser.parse_args()

    model_dir = os.path.join(PROJECT_ROOT, "Model_EQ_Surce")
    pre_path = os.path.join(model_dir, "Model_pred.pre")
    if not os.path.exists(pre_path):
        raise FileNotFoundError(pre_path)

    # 1. 读取偶极子阵列并生成 A6/A5 源卡片
    positions, moments = nn_common.load_initial_dipoles(args.dipoles)
    source_lines = fek.moments_to_source_lines(positions, moments)
    print(f"偶极子数量: {len(positions)}，源卡片数量: {len(source_lines)}")

    # 2. 备份并直接修改 Model_pred.pre
    backup = pre_path + ".backup"
    shutil.copy2(pre_path, backup)
    print(f"已备份原工程到: {backup}")
    fek.insert_sources_into_pre(pre_path, source_lines, pre_path)
    print(f"已将偶极子源写入: {pre_path}")

    # 3. 在 Model_EQ_Surce 目录运行 FEKO
    print("开始运行 FEKO 仿真（约数分钟）...")
    t0 = time.time()
    ok = fek.run_feko_case_parallel(
        pre_path,
        args.feko_bin,
        mpi_procs=args.mpi_procs,
        log_path=pre_path + ".runfeko.log",
    )
    print(f"FEKO 运行结束: ok={ok}, 耗时 {time.time()-t0:.1f} s")
    if not ok:
        raise RuntimeError("FEKO 仿真失败，请查看 Model_pred.out")

    # 4. 读取 3 个 .hfe 文件
    out_dir = nn_common.ensure_output_dir()
    feko_out_dir = os.path.join(out_dir, f"feko_validation_{args.out_suffix}")
    os.makedirs(feko_out_dir, exist_ok=True)

    planes_real = esr.load_all_planes()
    errors = {"overall": {}, "planes": {}}
    b_all = []
    b_rec_all = []

    for plane in planes_real:
        hfe_name = f"Model_pred_NearField_{plane['name']}.hfe"
        hfe_path = os.path.join(model_dir, hfe_name)
        if not os.path.exists(hfe_path):
            raise FileNotFoundError(f"FEKO 未生成: {hfe_path}")
        coords_rec, H_rec, counts_rec = esr.read_hfe(hfe_path)

        # 保存一份到 nn_refinement/feko_validation 目录
        shutil.copy2(hfe_path, os.path.join(feko_out_dir, hfe_name))
        # 同时按真实 .hfe 的列格式输出一份 TXT，便于直接查看和比较
        txt_path = os.path.join(
            feko_out_dir, f"predicted_field_{plane['name']}.txt"
        )
        nn_common.write_predicted_hfe_txt(
            txt_path,
            coords_rec,
            H_rec,
            counts_rec,
            request_name=f"NearField_{plane['name']}",
            source_label="ESR_FEKO",
        )

        # 逐面相对误差
        rel_l2 = np.linalg.norm(plane["H"] - H_rec) / np.linalg.norm(plane["H"])
        comp_errors = {}
        for c_idx, c_name in enumerate(["Hx", "Hy", "Hz"]):
            comp_errors[c_name] = round(float(
                np.linalg.norm(plane["H"][:, c_idx] - H_rec[:, c_idx])
                / np.linalg.norm(plane["H"][:, c_idx]) * 100.0
            ), 4)
        errors["planes"][plane["name"]] = {
            "rel_l2_error_percent": round(float(rel_l2 * 100.0), 4),
            "component_errors_percent": comp_errors,
            "hfe_file": hfe_path,
        }
        b_all.append(plane["H"].ravel())
        b_rec_all.append(H_rec.ravel())

        # 场图对比
        png = os.path.join(feko_out_dir, f"feko_comparison_{plane['name']}.png")
        esr.plot_field_comparison(
            png, coords_rec, plane["H"], H_rec, counts_rec, plane["name"]
        )

    b_all = np.concatenate(b_all)
    b_rec_all = np.concatenate(b_rec_all)
    errors["overall"] = {
        "rel_l2_error_percent": round(float(
            np.linalg.norm(b_all - b_rec_all) / np.linalg.norm(b_all) * 100.0
        ), 4),
        "n_obs": len(b_all) // 3,
    }

    # 5. 保存误差 JSON 与报告
    with open(os.path.join(feko_out_dir, "feko_validation_errors.json"),
              "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    lines = [
        "FEKO 就地验证报告",
        "=" * 50,
        f"偶极子文件: {args.dipoles}",
        f"偶极子数量: {len(positions)}",
        f"整体相对 L2 误差: {errors['overall']['rel_l2_error_percent']:.4f}%",
    ]
    for name, per in errors["planes"].items():
        lines.append(
            f"  {name}: {per['rel_l2_error_percent']:.4f}%"
            f"  (Hx {per['component_errors_percent']['Hx']:.4f}%, "
            f"Hy {per['component_errors_percent']['Hy']:.4f}%, "
            f"Hz {per['component_errors_percent']['Hz']:.4f}%)"
        )
    report_path = os.path.join(feko_out_dir, "feko_validation_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nFEKO 验证结果已保存到: {feko_out_dir}")


if __name__ == "__main__":
    main()
