# -*- coding: utf-8 -*-
"""
后台进度监视脚本
================

功能：
- 每 5 分钟检查一次 FEKO T_env 批处理进度；
- 当 T_env.npz 变为 feko_unit_dipole 后，转去监视后续重构流水线；
- 流水线完成或失败后退出并打印结果。

用法：
    python nn_refinement/monitor_tenv_pipeline.py
"""

import datetime
import json
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_ESR")
T_ENV_PATH = os.path.join(OUTPUT_DIR, "T_env.npz")
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "feko_tenv_progress.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "pipeline_status.json")

# 检查间隔：5 分钟
POLL_INTERVAL_SEC = 300


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tenv_ready():
    """T_env 是否已经是完整的 FEKO 环境感知矩阵。"""
    if not os.path.exists(T_ENV_PATH):
        return False
    try:
        data = np.load(T_ENV_PATH, allow_pickle=True)
        if "source" not in data or str(data["source"]) != "feko_unit_dipole":
            return False
        T = np.asarray(data["T"])
        src = np.asarray(data["src_coords"])
        return T.shape[1] == 3 * len(src)
    except Exception:
        return False


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    phase = "waiting_tenv"
    last_progress = -1
    last_pipeline_state = None

    print(f"[{now()}] 开始监视：等待 FEKO T_env 完成", flush=True)
    while True:
        if phase == "waiting_tenv":
            if tenv_ready():
                print(f"[{now()}] T_ENV_READY：FEKO 环境感知 T_env 已生成", flush=True)
                phase = "pipeline"
                continue
            progress = read_json(PROGRESS_PATH)
            if progress is not None:
                done = int(progress.get("completed", 0))
                total = int(progress.get("total", 0))
                eta = progress.get("eta_hours", 0)
                if done != last_progress:
                    last_progress = done
                    print(
                        f"[{now()}] T_env 进度 {done}/{total}，"
                        f"预计剩余 {eta} 小时",
                        flush=True,
                    )
        else:
            status = read_json(STATUS_PATH)
            if status is not None:
                state = status.get("state")
                if state != last_pipeline_state:
                    last_pipeline_state = state
                    print(
                        f"[{now()}] 流水线状态: {state} | "
                        f"{status.get('message', '')}",
                        flush=True,
                    )
                if state == "completed":
                    print(f"[{now()}] PIPELINE_DONE：整个重构流程执行完毕", flush=True)
                    return
                if state == "failed":
                    print(f"[{now()}] PIPELINE_FAILED：{status.get('message')}", flush=True)
                    return

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
