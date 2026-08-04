# -*- coding: utf-8 -*-
"""
脚本 4/4：验证神经网络修正后的偶极子阵列
========================================

功能：
- 用训练好的网络对真实扫描磁场做预测，得到修正偶极子阵列；
- 保存修正偶极子阵列 TXT；
- 保存初步偶极子阵列与修正偶极子阵列的完整预测磁场 TXT 和场图；
- 计算两者相对 L2 误差并输出验证报告。

用法：
    python nn_refinement/validate_refined_dipoles.py
"""

import json
import os
import shutil
import sys

import joblib
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402
import nn_common  # noqa: E402


def save_fields_and_errors(planes, positions, moments, label, out_dir):
    """
    计算某一组偶极子阵列的完整预测磁场，保存 TXT、场图与误差 JSON。
    """
    errors = esr.compute_validation_errors(planes, positions, moments)

    # 保存误差 JSON
    with open(os.path.join(out_dir, f"validation_errors_{label}.json"),
              "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    # 保存每个面的完整预测磁场 TXT（与真实 .hfe 同格式）和场图
    for plane in planes:
        H_pred = nn_common.predict_fields_full(
            positions, moments, plane["coords"]
        )
        txt_path = os.path.join(
            out_dir, f"predicted_field_{label}_{plane['name']}.txt"
        )
        nn_common.write_predicted_hfe_txt(
            txt_path,
            plane["coords"],
            H_pred,
            plane["counts"],
            request_name=f"NearField_{plane['name']}",
            source_label=f"ESR_{label}",
        )
        png_path = os.path.join(
            out_dir, f"field_comparison_{label}_{plane['name']}.png"
        )
        esr.plot_field_comparison(
            png_path, plane["coords"], plane["H"], H_pred,
            plane["counts"], f"{plane['name']}_{label}"
        )
    return errors


def main():
    out_dir = nn_common.ensure_output_dir()
    tenv = nn_common.load_tenv()
    positions, initial_moments = nn_common.load_initial_dipoles()
    planes = esr.load_all_planes()

    # 保存初步偶极子阵列快照到 nn_refinement 目录
    initial_txt = os.path.join(out_dir, "initial_equivalent_dipoles.txt")
    shutil.copyfile(nn_common.INITIAL_DIPOLES_TXT, initial_txt)

    # 读取训练时使用的观测点索引，保证预测输入与训练特征一致
    samples = np.load(os.path.join(out_dir, "training_samples.npz"),
                      allow_pickle=True)
    obs_indices = np.asarray(samples["nn_obs_indices"])
    T_sub, src_cols, rows = nn_common.select_tenv_columns(
        tenv, positions, obs_indices
    )

    # 构造真实扫描磁场的神经网络输入特征
    sub_H_list = []
    offset = 0
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep_idx = esr.compute_subsample_indices(
            len(mag), mag, esr.OBS_SUBSAMPLE_STEP, esr.PEAK_KEEP_FRACTION
        )
        sub_H_list.append(plane["H"][keep_idx])
        offset += len(keep_idx)
    b_sub = np.concatenate([h.ravel() for h in sub_H_list])
    b_sel = b_sub[rows]
    # 真实扫描场中可能存在 T_env 无法表示的成分（机箱外设备/噪声等）。
    # 若直接把原始场送入网络，这些不可表示成分会被错误映射到偶极矩上，
    # 导致 FEKO 验证误差反而变大。因此先把真实场投影到 T_env 列空间：
    # b_proj = T_env @ m_ls，只用可表示部分作为网络输入；
    # 预测矩也按投影场的范数缩放，保证与训练样本的定义一致。
    m_ls, _, _, _ = np.linalg.lstsq(T_sub, b_sel, rcond=1e-8)
    b_proj = T_sub @ m_ls
    norm_proj = float(np.linalg.norm(b_proj))
    if norm_proj < 1e-30:
        norm_proj = 1.0
    X_real = nn_common.fields_to_features(b_proj / norm_proj).reshape(1, -1)

    scaler_x = joblib.load(os.path.join(out_dir, "scaler_x.joblib"))
    scaler_y = joblib.load(os.path.join(out_dir, "scaler_y.joblib"))
    X_scaled = scaler_x.transform(X_real)
    mlp_path = os.path.join(out_dir, "nn_model.joblib")
    linear_path = os.path.join(out_dir, "linear_nn_weights.npz")
    if os.path.exists(mlp_path):
        mlp = joblib.load(mlp_path)
        y_pred = scaler_y.inverse_transform(mlp.predict(X_scaled))
    elif os.path.exists(linear_path):
        W = np.load(linear_path)["W"]
        A = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])
        y_pred = scaler_y.inverse_transform(A @ W)
    else:
        raise FileNotFoundError("未找到训练好的神经网络模型，请先运行 train_nn.py")
    refined_moments = nn_common.targets_to_moments(y_pred[0]) * norm_proj

    # 诊断：真实场在当前 T_env 列空间中的可表示性。
    # 若残差较大，说明当前 T_env（自由空间回退）与真实环境不匹配，
    # 神经网络无法给出有意义的修正，必须使用 FEKO 环境感知 T_env。
    resid_ls = np.linalg.norm(b_sel - T_sub @ m_ls) / np.linalg.norm(b_sel)
    diag = {
        "real_field_norm": float(np.linalg.norm(b_sel)),
        "projected_field_norm": float(norm_proj),
        "training_field_norm_mean": float(
            np.linalg.norm(np.asarray(samples["X"]), axis=1).mean()
        ),
        "tenv_subspace_fit_relative_error": round(float(resid_ls * 100.0), 4),
        "predicted_moment_max_abs": float(np.abs(refined_moments).max()),
    }
    with open(os.path.join(out_dir, "nn_diagnostics.json"), "w",
              encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    if resid_ls > 0.2:
        print("警告: 真实场在当前 T_env 列空间中拟合残差较大，"
              "神经网络修正结果不可信；建议使用 FEKO 环境感知 T_env。")

    refined_txt = os.path.join(out_dir, "equivalent_dipoles_refined.txt")
    nn_common.save_dipoles_txt(refined_txt, positions, refined_moments)
    print(f"修正偶极子阵列已保存: {refined_txt}")

    # 验证初步与修正偶极子阵列
    print("验证初步偶极子阵列 ...")
    err_initial = save_fields_and_errors(
        planes, positions, initial_moments, "initial", out_dir
    )
    print("验证修正偶极子阵列 ...")
    err_refined = save_fields_and_errors(
        planes, positions, refined_moments, "final", out_dir
    )

    # 汇总验证报告
    report = []
    report.append("神经网络修正偶极子阵列验证报告")
    report.append("=" * 60)
    report.append(f"初步偶极子数量: {len(positions)}")
    report.append(f"初步整体相对 L2 误差: "
                  f"{err_initial['overall']['rel_l2_error_percent']:.4f}%")
    report.append(f"修正后整体相对 L2 误差: "
                  f"{err_refined['overall']['rel_l2_error_percent']:.4f}%")
    report.append("")
    report.append("逐面误差（初步 / 修正后）:")
    for name in err_initial["planes"]:
        e0 = err_initial["planes"][name]["rel_l2_error_percent"]
        e1 = err_refined["planes"][name]["rel_l2_error_percent"]
        report.append(f"  {name}: {e0:.4f}% -> {e1:.4f}%")
    report.append("")
    report.append("说明：预测磁场 TXT 与真实 .hfe 格式一致；")
    report.append("注意：本报告的误差和场图用自由空间格林函数计算，仅作为中间检查；")
    report.append("最终误差以 FEKO 就地仿真结果（run_feko_validation_inplace.py）为准。")
    report.append(f"真实场在 T_env 子空间中的拟合相对误差: "
                  f"{resid_ls * 100:.4f}%")
    report.append("若该误差较大，说明自由空间回退 T_env 与真实环境不匹配，")
    report.append("神经网络修正不可信，应改用 FEKO 环境感知 T_env 重新生成样本。")
    report_path = os.path.join(out_dir, "nn_validation_report.txt")
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\n所有结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
