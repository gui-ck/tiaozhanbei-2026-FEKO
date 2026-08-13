# -*- coding: utf-8 -*-
"""近场磁场文件的读取、抽稀与写入。"""

import os
from datetime import datetime

import numpy as np

from . import config


def read_field_file(filepath):
    """
    读取 FEKO 近场磁场文件（.hfe 或 .txt）。

    两种格式的数据行均为：
    ``X Y Z Re(Hx) Im(Hx) Re(Hy) Im(Hy) Re(Hz) Im(Hz)``

    返回 (coords, H, counts)。counts 为 (nx, ny, nz)。
    """
    coords = []
    H = []
    counts = None
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#No. of X Samples"):
                counts = [int(line.split(":")[1])]
                continue
            if line.startswith("#No. of Y Samples"):
                counts.append(int(line.split(":")[1]))
                continue
            if line.startswith("#No. of Z Samples"):
                counts.append(int(line.split(":")[1]))
                continue
            parts = line.replace('"', "").split()
            if len(parts) < 9:
                continue
            try:
                xyz = [float(v) for v in parts[:3]]
                h = [
                    complex(float(parts[3]), float(parts[4])),
                    complex(float(parts[5]), float(parts[6])),
                    complex(float(parts[7]), float(parts[8])),
                ]
            except ValueError:
                continue
            coords.append(xyz)
            H.append(h)

    if not coords:
        raise ValueError(f"文件中没有可解析的磁场数据: {filepath}")
    coords = np.asarray(coords)
    H = np.asarray(H, dtype=complex)
    if counts is None:
        counts = infer_counts(coords)
    return coords, H, tuple(counts)


def infer_counts(coords):
    """从坐标的独立采样数推断 (nx, ny, nz)。"""
    return tuple(int(np.unique(coords[:, axis]).size) for axis in range(3))


def write_field_txt(filepath, coords, H, counts, request_name,
                    source_label="EquivalentSR_Prediction"):
    """把预测磁场写成与 Data_Scanning_Field 相同格式的 .txt。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        '      "X"                "Y"                "Z"              '
        '"Re(Hx)"           "Im(Hx)"           "Re(Hy)"           "Im(Hy)"           '
        '"Re(Hz)"           "Im(Hz)"'
    )
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"##Request: {request_name}\n")
        f.write(f"##Source: {source_label}\n")
        f.write(f"##Date: {now}\n")
        f.write(f"##Counts: {' '.join(str(c) for c in counts)}\n")
        f.write(header + "\n")
        for c, h in zip(coords, H):
            f.write(
                f"{c[0]: .8E} {c[1]: .8E} {c[2]: .8E} "
                f"{h[0].real: .8E} {h[0].imag: .8E} "
                f"{h[1].real: .8E} {h[1].imag: .8E} "
                f"{h[2].real: .8E} {h[2].imag: .8E}\n"
            )


def real_field_path(name):
    """返回 Data_Scanning_Field 下的真实磁场 .txt 路径。"""
    return os.path.join(config.DATA_DIR, f"Model_real_NearField_{name}.txt")


def load_real_planes(plane_defs=None):
    """按平面定义批量读取真实磁场。"""
    if plane_defs is None:
        plane_defs = config.RECONSTRUCTION_PLANES
    planes = []
    for pdef in plane_defs:
        path = real_field_path(pdef["name"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"真实磁场文件不存在: {path}")
        coords, H, counts = read_field_file(path)
        planes.append({
            "name": pdef["name"],
            "desc": pdef.get("desc", ""),
            "request_name": f"NearField_{pdef['name']}",
            "coords": coords,
            "H": H,
            "counts": counts,
        })
        print(f"读取 {pdef['name']}: {len(coords)} 个观测点, counts={counts}")
    return planes


def compute_subsample_indices(n_points, mag, step=config.OBS_SUBSAMPLE_STEP,
                              keep_fraction=config.PEAK_KEEP_FRACTION):
    """均匀抽稀，同时保留幅值最高的峰值点。"""
    uniform_idx = np.arange(0, n_points, step)
    n_peak = max(1, int(round(keep_fraction * n_points)))
    peak_idx = np.argsort(mag)[-n_peak:]
    return np.unique(np.concatenate([uniform_idx, peak_idx]))


def subsample_planes(planes):
    """
    对全部扫描面抽稀。

    返回 sub_coords、b（复磁场列向量）、plane_offsets、subsample_indices。
    """
    sub_coords_list = []
    sub_H_list = []
    plane_offsets = []
    subsample_indices = []
    offset = 0
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep = compute_subsample_indices(len(mag), mag)
        subsample_indices.append(keep)
        sub_coords_list.append(plane["coords"][keep])
        sub_H_list.append(plane["H"][keep])
        plane_offsets.append(offset)
        offset += len(keep)
    sub_coords = np.vstack(sub_coords_list)
    b = np.concatenate([H.ravel() for H in sub_H_list])
    return sub_coords, b, np.asarray(plane_offsets, dtype=int), subsample_indices


def concatenate_fields(planes):
    """把所有平面的复磁场拼成一个列向量。"""
    return np.concatenate([p["H"].ravel() for p in planes])
