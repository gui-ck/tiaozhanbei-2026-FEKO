# -*- coding: utf-8 -*-
"""
方案 1 + 方案 2 后台流水线
=========================

等待全观测点 T_env 生成完成后自动执行：
1. 全平面最小二乘重构（仅原有 40 个位置）；
2. 全平面最小二乘重构（原有 40 + 新候选 30 个位置，并裁剪/合并）；
3. 分别对两个候选模型运行 FEKO 就地仿真验证；
4. 保存最终验证误差和场图。

用法：
    python nn_refinement/run_full_tenv_pipeline.py
"""

import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_ESR")
FULL_TENV_PATH = os.path.join(OUTPUT_DIR, "T_env_full.npz")
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "feko_full_tenv_progress.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "full_tenv_pipeline_status.json")
RUN_LOG = os.path.join(OUTPUT_DIR, "full_tenv_pipeline.log")

PYTHON = r"D:\Anaconda3\envs\PINN_py311\python.exe"


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(status):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def full_tenv_ready():
    if not os.path.exists(FULL_TENV_PATH):
        return False
    try:
        data = np.load(FULL_TENV_PATH, allow_pickle=True)
        if "source" not in data or str(data["source"]) != "feko_unit_dipole_full":
            return False
        src = np.asarray(data["src_coords"])
        T = np.asarray(data["T"])
        return T.shape[1] == 3 * len(src)
    except Exception:
        return False


def feko_idle():
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


def wait_for_full_tenv():
    log("等待全观测点 T_env 生成完成 ...")
    while True:
        progress_txt = ""
        if os.path.exists(PROGRESS_PATH):
            try:
                with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                    p = json.load(f)
                progress_txt = (f"completed={p.get('completed')}/"
                                f"{p.get('total')}, "
                                f"eta={p.get('eta_hours')}h")
            except Exception:
                progress_txt = "progress unreadable"
        if full_tenv_ready() and feko_idle():
            log("全观测点 T_env 已完成")
            return
        log(f"未完成（{progress_txt}），60 秒后重试 ...")
        time.sleep(60)


def run_step(name, script, extra_args=None):
    cmd = [PYTHON, "-X", "utf8", script]
    if extra_args:
        cmd.extend(extra_args)
    log(f"开始步骤: {name}")
    r = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = (r.stdout or "")[-4000:] + (r.stderr or "")[-2000:]
    log(f"步骤完成: {name}, returncode={r.returncode}\n{tail}")
    write_status({
        "current_step": name,
        "returncode": r.returncode,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    if r.returncode != 0:
        raise RuntimeError(f"步骤失败: {name}\n{tail}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_status({"state": "waiting_for_full_tenv"})
    wait_for_full_tenv()

    run_step("全观测点重构",
             os.path.join("nn_refinement", "reconstruct_full_tenv.py"))

    run_step(
        "FEKO 验证 full40",
        os.path.join("nn_refinement", "run_feko_validation_inplace.py"),
        ["--dipoles", os.path.join(OUTPUT_DIR, "equivalent_dipoles_full40.txt"),
         "--mpi-procs", "6", "--out-suffix", "full40"],
    )
    run_step(
        "FEKO 验证 v2",
        os.path.join("nn_refinement", "run_feko_validation_inplace.py"),
        ["--dipoles", os.path.join(OUTPUT_DIR, "equivalent_dipoles_v2.txt"),
         "--mpi-procs", "6", "--out-suffix", "v2"],
    )

    write_status({
        "state": "completed",
        "message": "方案1+方案2执行完毕",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    log("方案1+方案2执行完毕")


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
