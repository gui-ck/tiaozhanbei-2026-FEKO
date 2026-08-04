# -*- coding: utf-8 -*-
"""
后台流水线：等待 FEKO T_env 完成后自动重新执行整个重构任务
==========================================================

流程：
1. 轮询 ``Output_ESR/T_env.npz``，直到其来源变为 ``feko_unit_dipole``
   且列数等于“偶极子位置数 x 3”（说明 297 列单元偶极子仿真全部完成）；
2. 删除之前基于自由空间回退矩阵生成的旧结果（保留 T_env.npz 和列缓存）；
3. 用 FEKO T_env 重新做传统稀疏重构，得到初步偶极子阵列；
4. 在初步偶极子阵列附近随机生成 5000 组训练样本；
5. 训练“磁场 -> 偶极矩”神经网络；
6. 用训练好的网络修正偶极子阵列并做中间验证；
7. 分别对初步阵列和修正阵列做 FEKO 就地仿真验证，输出最终误差和场图。

注意：所有子进程均使用包含 “PINN” 字样的 conda 环境
（默认 ``D:\\Anaconda3\\envs\\PINN_py311\\python.exe``）。

用法：
    python nn_refinement/run_pipeline_after_tenv.py
"""

import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_ESR")
T_ENV_PATH = os.path.join(OUTPUT_DIR, "T_env.npz")
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "feko_tenv_progress.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "pipeline_status.json")
RUN_LOG = os.path.join(OUTPUT_DIR, "pipeline_run.log")

# PINN conda 环境中的 Python 解释器
PYTHON = r"D:\Anaconda3\envs\PINN_py311\python.exe"


def log(msg):
    """带时间戳写日志，同时打印到控制台。"""
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(status):
    """把流水线当前状态写入 JSON，方便后台查看。"""
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def tenv_ready():
    """
    判断 FEKO T_env 是否已经生成完毕。

    同时检查来源标识和列数，避免读到旧版自由空间矩阵或未写完的临时文件。
    """
    if not os.path.exists(T_ENV_PATH):
        return False
    try:
        data = np.load(T_ENV_PATH, allow_pickle=True)
        if "source" not in data or "src_coords" not in data or "T" not in data:
            return False
        if str(data["source"]) != "feko_unit_dipole":
            return False
        src = np.asarray(data["src_coords"])
        T = np.asarray(data["T"])
        return T.shape[1] == 3 * len(src)
    except Exception:
        # npz 正在被写入时可能短暂无法读取，返回 False 等待下一轮
        return False


def feko_idle():
    """检查是否还有 FEKO 并行求解进程在运行。"""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq feko.csv.impi.exe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return "feko.csv.impi.exe" not in r.stdout
    except Exception:
        return False


def wait_for_tenv():
    """轮询等待 FEKO T_env 完成，每 60 秒检查一次。"""
    log("等待 FEKO T_env 批处理完成 ...")
    while True:
        progress_txt = ""
        if os.path.exists(PROGRESS_PATH):
            try:
                with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    progress_txt = (
                        f"completed={progress.get('completed')}/"
                        f"{progress.get('total')}, "
                        f"eta={progress.get('eta_hours')}h"
                    )
            except Exception:
                progress_txt = "progress unreadable"
        if tenv_ready() and feko_idle():
            log("FEKO T_env 已完成")
            return
        log(f"T_env 未完成（{progress_txt}），60 秒后重试 ...")
        time.sleep(60)


def cleanup_old_outputs():
    """
    删除之前基于自由空间回退矩阵产生的冗余结果。

    保留：
    - Output_ESR/T_env.npz（FEKO 环境感知传递矩阵）；
    - Output_ESR/feko_tenv_columns/（列缓存，方便断点续跑）。
    删除：
    - 旧的偶极子 TXT、验证报告、场图、元数据；
    - 旧的 nn_refinement 输出目录；
    - 早期自由空间冒烟测试目录。
    """
    log("开始清理之前的多余结果 ...")
    delete_files = [
        os.path.join(OUTPUT_DIR, "dipole_metadata.json"),
        os.path.join(OUTPUT_DIR, "equivalent_dipoles.txt"),
        os.path.join(OUTPUT_DIR, "validation_errors.json"),
        os.path.join(OUTPUT_DIR, "validation_report.txt"),
        os.path.join(OUTPUT_DIR, "tenv_initial_meta.json"),
    ]
    for pattern in [
        os.path.join(OUTPUT_DIR, "field_comparison_*.png"),
        os.path.join(OUTPUT_DIR, "field_comparison_*.json"),
    ]:
        delete_files.extend(glob.glob(pattern))

    delete_dirs = [
        os.path.join(OUTPUT_DIR, "nn_refinement"),
        os.path.join(OUTPUT_DIR, "feko_smoke"),
    ]
    for f in delete_files:
        if os.path.exists(f):
            os.remove(f)
            log(f"删除文件: {f}")
    for d in delete_dirs:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
            log(f"删除目录: {d}")
    log("旧结果清理完成")


def run_step(name, script, extra_args=None):
    """用 PINN 环境运行一个流水线脚本并记录输出。"""
    cmd = [PYTHON, "-X", "utf8", script]
    if extra_args:
        cmd.extend(extra_args)
    log(f"开始步骤: {name} -> {' '.join(cmd)}")
    r = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = (r.stdout or "")[-4000:] + (r.stderr or "")[-2000:]
    log(f"步骤完成: {name}, returncode={r.returncode}")
    log("输出摘要:\n" + tail)
    write_status({
        "current_step": name,
        "returncode": r.returncode,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "output_tail": tail,
    })
    if r.returncode != 0:
        raise RuntimeError(f"步骤失败: {name}\n{tail}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_status({
        "state": "waiting_for_tenv",
        "message": "等待 FEKO T_env 批处理完成",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    wait_for_tenv()
    cleanup_old_outputs()

    # 1. 传统稀疏重构：基于 FEKO T_env 得到初步偶极子阵列
    run_step(
        "传统重构（FEKO T_env + Group OMP）",
        os.path.join("nn_refinement", "reconstruct_initial_with_tenv.py"),
    )

    # 2. 生成神经网络训练样本：固定位置，随机扰动偶极矩
    run_step(
        "生成训练样本",
        os.path.join("nn_refinement", "generate_training_samples.py"),
        ["--n-samples", "5000", "--n-obs-points", "500", "--sigma", "0.5"],
    )

    # 3. 训练神经网络
    run_step(
        "训练神经网络",
        os.path.join("nn_refinement", "train_nn.py"),
        ["--solver", "linear", "--alpha", "1e-6"],
    )

    # 4. 预测修正偶极子阵列并做中间验证
    run_step(
        "验证修正偶极子阵列",
        os.path.join("nn_refinement", "validate_refined_dipoles.py"),
    )

    # 5. FEKO 就地验证：初步偶极子阵列
    initial_txt = os.path.join(OUTPUT_DIR, "equivalent_dipoles.txt")
    run_step(
        "FEKO 验证初步偶极子阵列",
        os.path.join("nn_refinement", "run_feko_validation_inplace.py"),
        ["--dipoles", initial_txt, "--mpi-procs", "6", "--out-suffix", "initial"],
    )

    # 6. FEKO 就地验证：神经网络修正后的偶极子阵列
    refined_txt = os.path.join(
        OUTPUT_DIR, "nn_refinement", "equivalent_dipoles_refined.txt"
    )
    run_step(
        "FEKO 验证修正偶极子阵列",
        os.path.join("nn_refinement", "run_feko_validation_inplace.py"),
        ["--dipoles", refined_txt, "--mpi-procs", "6", "--out-suffix", "refined"],
    )

    write_status({
        "state": "completed",
        "message": "整个重构流程执行完毕",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    log("整个重构流程执行完毕")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status({
            "state": "failed",
            "message": str(exc),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        log(f"流水线失败: {exc}")
        raise
