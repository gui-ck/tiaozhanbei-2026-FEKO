# -*- coding: utf-8 -*-
"""误差计算、场图与验证报告。"""

import json
import os
import shutil
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, feko, fields, io, tenv


def compute_relative_errors(real_planes, pred_fields):
    """
    计算整体与逐面的相对 L2 误差。

    pred_fields 为与 real_planes 一一对应的 (N, 3) 复磁场数组。
    """
    b_all = []
    b_rec_all = []
    result = {"overall": {}, "planes": {}}
    for plane, H_pred in zip(real_planes, pred_fields):
        H_orig = plane["H"]
        rel_l2 = np.linalg.norm(H_orig - H_pred) / np.linalg.norm(H_orig)
        comp_errors = {}
        for c_idx, c_name in enumerate(["Hx", "Hy", "Hz"]):
            comp_errors[c_name] = round(float(
                np.linalg.norm(H_orig[:, c_idx] - H_pred[:, c_idx])
                / np.linalg.norm(H_orig[:, c_idx]) * 100.0
            ), 4)
        result["planes"][plane["name"]] = {
            "rel_l2_error_percent": round(float(rel_l2 * 100.0), 4),
            "component_errors_percent": comp_errors,
            "n_obs": len(H_orig),
        }
        b_all.append(H_orig.ravel())
        b_rec_all.append(H_pred.ravel())

    b_all = np.concatenate(b_all)
    b_rec_all = np.concatenate(b_rec_all)
    result["overall"] = {
        "rel_l2_error_percent": round(float(
            np.linalg.norm(b_all - b_rec_all) / np.linalg.norm(b_all) * 100.0
        ), 4),
        "n_obs": len(b_all) // 3,
    }
    return result


def tenv_model_error(real_planes, full_tenv, positions, moments):
    """用全平面 T_env 计算某组偶极子的辐射磁场误差，不调用 FEKO。"""
    T = np.asarray(full_tenv["T"])
    src = np.asarray(full_tenv["src_coords"])
    cols = tenv.columns_for_positions(src, positions)
    H_all = (T[:, cols] @ moments.ravel()).reshape(-1, 3)
    offsets = np.asarray(full_tenv["plane_offsets"], dtype=int)
    pred_fields = []
    for pi, plane in enumerate(real_planes):
        start = offsets[pi]
        end = offsets[pi + 1] if pi + 1 < len(offsets) else len(plane["coords"]) + start
        pred_fields.append(H_all[start:end])
    return compute_relative_errors(real_planes, pred_fields)


def plot_field_comparison(filepath, coords, H_orig, H_rec, counts, name):
    """绘制单个扫描面 Hx/Hy/Hz 的原始与重构对比图。"""
    axis_labels = {
        "xoy": ("X (m)", "Y (m)"),
        "xoz": ("X (m)", "Z (m)"),
        "yoz": ("Y (m)", "Z (m)"),
        "pred": ("Y (m)", "Z (m)"),
    }
    label_key = name.split("_")[0] if "_" in name else name
    x_label, y_label = axis_labels.get(
        label_key, ("axis1 (m)", "axis2 (m)")
    )
    comp_names = ["Hx", "Hy", "Hz"]

    varying_axes = [i for i, n in enumerate(counts) if n > 1]
    horiz_axis, vert_axis = varying_axes[0], varying_axes[1]
    diff_first = coords[1] - coords[0]
    fast_axis = int(np.argmax(np.abs(diff_first)))
    slow_axis = [a for a in varying_axes if a != fast_axis][0]
    n_slow = int(counts[slow_axis])
    n_fast = int(counts[fast_axis])

    x0, x1 = float(coords[:, horiz_axis].min()), float(coords[:, horiz_axis].max())
    y0, y1 = float(coords[:, vert_axis].min()), float(coords[:, vert_axis].max())
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)

    for row, c_name in enumerate(comp_names):
        Z_orig = np.abs(H_orig[:, row]).reshape(n_slow, n_fast)
        Z_rec = np.abs(H_rec[:, row]).reshape(n_slow, n_fast)
        if fast_axis != horiz_axis:
            Z_orig = Z_orig.T
            Z_rec = Z_rec.T
        vmin = min(Z_orig.min(), Z_rec.min())
        vmax = max(Z_orig.max(), Z_rec.max())

        for col, (Z, title) in enumerate([
            (Z_orig, f"{name} Original |{c_name}|"),
            (Z_rec, f"{name} Reconstructed |{c_name}|"),
        ]):
            im = axes[row, col].imshow(
                Z, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto", cmap="viridis",
                vmin=vmin, vmax=vmax, interpolation="bicubic",
            )
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel(x_label)
            axes[row, col].set_ylabel(y_label)
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.03)

    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def write_validation_outputs(out_dir, real_planes, pred_fields,
                             title="等效源验证报告",
                             source_label="ESR_FEKO"):
    """
    保存预测磁场 .txt、场图、误差 JSON 和中文报告。
    """
    os.makedirs(out_dir, exist_ok=True)
    errors = compute_relative_errors(real_planes, pred_fields)

    with open(os.path.join(out_dir, "validation_errors.json"),
              "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    for plane, H_pred in zip(real_planes, pred_fields):
        name = plane["name"]
        fields.write_field_txt(
            os.path.join(out_dir, f"predicted_field_{name}.txt"),
            plane["coords"], H_pred, plane["counts"],
            request_name=f"NearField_{name}",
            source_label=source_label,
        )
        plot_field_comparison(
            os.path.join(out_dir, f"field_comparison_{name}.png"),
            plane["coords"], plane["H"], H_pred,
            plane["counts"], name,
        )

    lines = [title, "=" * 50,
             f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
             f"整体相对 L2 误差: "
             f"{errors['overall']['rel_l2_error_percent']:.4f}%"]
    for name, per in errors["planes"].items():
        lines.append(
            f"  {name}: {per['rel_l2_error_percent']:.4f}%"
            f"  (Hx {per['component_errors_percent']['Hx']:.4f}%, "
            f"Hy {per['component_errors_percent']['Hy']:.4f}%, "
            f"Hz {per['component_errors_percent']['Hz']:.4f}%)"
        )
    report_path = os.path.join(out_dir, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    return errors


def run_feko_validation(dipole_path, results_dir=config.RESULTS_DIR,
                        feko_bin=config.DEFAULT_FEKO_BIN,
                        mpi_procs=config.MPI_PROCS,
                        skip_feko=False):
    """
    读取等效偶极子阵列，导入 FEKO 求解放射场，并与真实磁场对比。
    """
    positions, moments = io.load_dipoles_txt(dipole_path)
    print(f"读取等效偶极子: {len(positions)} 个, 源卡片: "
          f"{len(feko.moments_to_source_lines(positions, moments))} 个")

    real_planes = fields.load_real_planes(config.VALIDATION_PLANES)
    pre_path = os.path.join(config.MODEL_EQ_DIR, "Model_pred_validation.pre")
    feko.build_validation_pre(
        config.TEMPLATE_PRE,
        feko.moments_to_source_lines(positions, moments),
        pre_path,
        extra_plane_defs=config.EXTRA_VALIDATION_PLANES,
    )
    print(f"验证工程已生成: {pre_path}")

    if not skip_feko:
        print("开始运行 FEKO 验证仿真 ...")
        ok = feko.run_feko_case_parallel(
            pre_path, feko_bin, mpi_procs=mpi_procs,
            log_path=pre_path + ".runfeko.log",
        )
        if not ok:
            raise RuntimeError("FEKO 验证仿真失败，请检查 .out 和 .runfeko.log")

    out_dir = os.path.join(results_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)
    pred_fields = []
    for plane in real_planes:
        hfe_path = os.path.join(
            config.MODEL_EQ_DIR,
            f"Model_pred_validation_NearField_{plane['name']}.hfe",
        )
        if not os.path.exists(hfe_path):
            raise FileNotFoundError(f"FEKO 未生成验证磁场: {hfe_path}")
        coords_pred, H_pred, counts = fields.read_field_file(hfe_path)
        if not np.allclose(coords_pred, plane["coords"]):
            raise RuntimeError(f"{plane['name']}: FEKO 输出坐标与真实场不一致")
        pred_fields.append(H_pred)
        shutil.copy2(hfe_path, os.path.join(out_dir, f"predicted_field_{plane['name']}.hfe"))

    write_validation_outputs(
        out_dir, real_planes, pred_fields,
        title="FEKO 等效源验证报告",
        source_label="ESR_FEKO",
    )
    print(f"FEKO 验证结果已保存到: {out_dir}")
    return out_dir
