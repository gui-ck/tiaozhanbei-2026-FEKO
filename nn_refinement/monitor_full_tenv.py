# -*- coding: utf-8 -*-
"""
全观测点 T_env 后台状态监视脚本
===============================

功能：
- 每隔 10 分钟读取全观测点 T_env 生成进度和后续流水线状态；
- 把每次检查结果追加写入 Output_ESR/background_monitor.log；
- 当后续流水线完成或失败时退出。

用法：
    python nn_refinement/monitor_full_tenv.py
"""

import datetime
import json
import os
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_ESR")
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "feko_full_tenv_progress.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "full_tenv_pipeline_status.json")
MONITOR_LOG = os.path.join(OUTPUT_DIR, "background_monitor.log")

# 检查间隔：10 分钟
POLL_INTERVAL_SEC = 600


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    print(f"[{now()}] 开始后台状态监视，每 10 分钟检查一次", flush=True)
    while True:
        progress = read_json(PROGRESS_PATH)
        status = read_json(STATUS_PATH)

        lines = [f"[{now()}] 状态检查"]
        if progress is not None:
            lines.append(
                f"  全观测点 T_env: {progress.get('completed')}/"
                f"{progress.get('total')}, "
                f"预计剩余 {progress.get('eta_hours')}h"
            )
        else:
            lines.append("  全观测点 T_env 进度文件不可读")

        state = status.get("state") if status else None
        message = status.get("message") if status else ""
        lines.append(f"  流水线状态: {state} | {message}")

        text = "\n".join(lines)
        print(text, flush=True)
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")

        if state in ("completed", "failed"):
            print(f"[{now()}] 监视结束，流水线状态: {state}", flush=True)
            return
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
