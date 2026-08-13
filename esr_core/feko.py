# -*- coding: utf-8 -*-
"""FEKO 源卡片生成、.pre 工程构建与并行求解。"""

import glob
import os
import subprocess

import numpy as np

from . import config


def make_a6_line(position, component, magnitude=1.0, phase_deg=0.0):
    """生成 A6 磁偶极子源卡片。"""
    direction = {
        "x": (90.0, 0.0),
        "y": (90.0, 90.0),
        "z": (0.0, 0.0),
    }
    theta, phi = direction[component]
    return (
        f"A6: 1 :  :  :  : 0 : "
        f"{magnitude:.8E} : {phase_deg:.8E} : "
        f"{position[0]:.8E} : {position[1]:.8E} : {position[2]:.8E} : "
        f"{theta:.6f} : {phi:.6f}"
    )


def make_a5_line(position, component, magnitude=1.0, phase_deg=0.0):
    """生成 A5 电偶极子源卡片。"""
    direction = {
        "x": (90.0, 0.0),
        "y": (90.0, 90.0),
        "z": (0.0, 0.0),
    }
    theta, phi = direction[component]
    return (
        f"A5: 1 :  :  :  :  : "
        f"{magnitude:.8E} : {phase_deg:.8E} : "
        f"{position[0]:.8E} : {position[1]:.8E} : {position[2]:.8E} : "
        f"{theta:.6f} : {phi:.6f}"
    )


def moments_to_source_lines(positions, moments):
    """
    把复偶极矩拆成 FEKO 源卡片：
    Mx/My 用 A6 磁偶极子，Pz 用 A5 电偶极子。
    """
    lines = []
    for pos, m in zip(positions, moments):
        for c_idx, comp in enumerate(["x", "y"]):
            mag = abs(m[c_idx])
            if mag < 1e-14:
                continue
            lines.append(make_a6_line(
                pos, comp, magnitude=mag,
                phase_deg=np.degrees(np.angle(m[c_idx])),
            ))
        mag_z = abs(m[2])
        if mag_z >= 1e-14:
            lines.append(make_a5_line(
                pos, "z", magnitude=mag_z,
                phase_deg=np.degrees(np.angle(m[2])),
            ))
    return lines


def _template_lines(template_pre_path):
    """优先使用干净模板，避免残留上一轮验证源。"""
    for suffix in (".clean", ".backup"):
        candidate = template_pre_path + suffix
        if os.path.exists(candidate):
            template_pre_path = candidate
            break
    with open(template_pre_path, "r", encoding="utf-8") as f:
        return f.readlines()


def insert_sources_into_pre(template_pre_path, source_lines, output_pre_path):
    """在 ``** Sources`` 段后插入偶极子源卡片。"""
    lines = _template_lines(template_pre_path)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "** Sources":
            insert_idx = i
            break
    if insert_idx is None:
        raise ValueError("模板 .pre 中找不到 ** Sources 段")

    new_lines = lines[: insert_idx + 1]
    new_lines.append("** Added by ESR pipeline\n")
    new_lines.extend(line + "\n" for line in source_lines)
    new_lines.append("\n")
    new_lines.extend(lines[insert_idx + 1:])
    _write_pre(output_pre_path, new_lines)


def add_extra_plane_requests(pre_lines, plane_defs):
    """在 EN 卡片前插入额外近场请求。"""
    blocks = []
    for p in plane_defs:
        blocks.append(f"** Near field: NearField_{p['name']}")
        blocks.append("DA: 3 : 3 : 0 : 0 : 0 : 0 : 0 : 0 : 0 : 0 :  : 0")
        blocks.append("OF: 1 : 0 :  :  :  : 0 : 0 : 0 : 0 : 0 : 0")
        blocks.append(
            f"FE: 2 : 1 : {p['n_y']} : {p['n_z']} : 0 : "
            f"{p['x']} : {p['y_start']} : {p['z_start']} : "
            f"0 : {p['y_step']} : {p['z_step']}   ** NearField_{p['name']}"
        )
        blocks.append("")

    insert_idx = None
    for i, line in enumerate(pre_lines):
        if line.strip().startswith("EN"):
            insert_idx = i
            break
    if insert_idx is None:
        raise ValueError("模板 .pre 中找不到 EN 卡片")
    block_lines = [line + "\n" for line in blocks]
    return pre_lines[:insert_idx] + block_lines + pre_lines[insert_idx:]


def build_validation_pre(template_pre_path, source_lines, output_pre_path,
                         extra_plane_defs=None):
    """生成带全部验证近场请求的 FEKO 工程。"""
    lines = _template_lines(template_pre_path)
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "** Sources":
            insert_idx = i
            break
    if insert_idx is None:
        raise ValueError("模板 .pre 中找不到 ** Sources 段")

    new_lines = lines[: insert_idx + 1]
    new_lines.append("** Added by ESR validation\n")
    new_lines.extend(line + "\n" for line in source_lines)
    new_lines.append("\n")
    new_lines.extend(lines[insert_idx + 1:])
    if extra_plane_defs:
        new_lines = add_extra_plane_requests(new_lines, extra_plane_defs)
    _write_pre(output_pre_path, new_lines)


def _write_pre(filepath, lines):
    with open(filepath, "w", encoding="ascii", newline="\n") as f:
        f.writelines(lines)


def run_feko_case_parallel(pre_path, feko_bin=config.DEFAULT_FEKO_BIN,
                           mpi_procs=config.MPI_PROCS, log_path=None):
    """
    用 runfeko 并行求解单个 FEKO 工程。

    返回 True 表示 .out 存在且包含 Finished。
    """
    workdir = os.path.dirname(os.path.abspath(pre_path))
    base = os.path.splitext(os.path.basename(pre_path))[0]
    runfeko = os.path.join(feko_bin, "runfeko.exe")
    cmd = [runfeko, base, "-np", str(mpi_procs)]

    log_handle = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        r = subprocess.run(
            cmd, cwd=workdir, stdout=log_handle, stderr=subprocess.STDOUT
        )
    finally:
        if log_handle is not None:
            log_handle.close()

    out_path = os.path.join(workdir, f"{base}.out")
    if r.returncode != 0 or not os.path.exists(out_path):
        return False
    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        return "Finished" in f.read()


def cleanup_case_files(work_root, base):
    """清理某个 FEKO 工程产生的临时文件，避免目录被大量模型占满。"""
    for pattern in (base + ".*", base + "_NearField_*.hfe"):
        for f in glob.glob(os.path.join(work_root, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass
