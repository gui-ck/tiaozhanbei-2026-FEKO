# -*- coding: utf-8 -*-
"""等效偶极子 TXT 的读取与保存。"""

import numpy as np


def save_dipoles_txt(filepath, positions, moments):
    """保存偶极子 TXT，列格式与既有结果保持一致。"""
    header = "x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for pos, m in zip(positions, moments):
            f.write(
                f"{pos[0]:.8E} {pos[1]:.8E} {pos[2]:.8E} "
                f"{m[0].real:.8E} {m[0].imag:.8E} "
                f"{m[1].real:.8E} {m[1].imag:.8E} "
                f"{m[2].real:.8E} {m[2].imag:.8E}\n"
            )


def load_dipoles_txt(filepath):
    """读取偶极子 TXT，返回 positions 与 moments。"""
    positions = []
    moments = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            if len(vals) < 9 or vals[0] == "x":
                continue
            positions.append([float(v) for v in vals[:3]])
            moments.append([
                complex(float(vals[3]), float(vals[4])),
                complex(float(vals[5]), float(vals[6])),
                complex(float(vals[7]), float(vals[8])),
            ])
    return np.asarray(positions), np.asarray(moments, dtype=complex)
