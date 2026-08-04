# -*- coding: utf-8 -*-
"""
神经网络修正等效偶极子阵列的共享工具模块
==========================================

本模块只提供各独立脚本公用的函数和路径配置，不包含具体流程。
四个独立脚本分别为：

1. ``generate_T_env.py``           生成/复用环境感知传递矩阵 T_env；
2. ``generate_training_samples.py`` 在初始偶极子阵列附近随机扰动偶极矩，
                                    通过 T_env 计算磁场，生成训练样本；
3. ``train_nn.py``                 训练“磁场 -> 偶极矩”神经网络；
4. ``validate_refined_dipoles.py`` 预测修正偶极子阵列，并做完整验证。

说明：当前 T_env 若为自由空间回退矩阵，则训练与验证也基于自由空间模型；
若之后用 FEKO 生成环境感知 T_env，脚本无需改动，只需替换 T_env.npz。
"""

import datetime
import os
import sys

import numpy as np


# 把项目根目录加入搜索路径，便于导入 esr_reconstruction.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import esr_reconstruction as esr  # noqa: E402


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Output_ESR", "nn_refinement")
T_ENV_PATH = os.path.join(PROJECT_ROOT, "Output_ESR", "T_env.npz")
INITIAL_DIPOLES_TXT = os.path.join(
    PROJECT_ROOT, "Output_ESR", "equivalent_dipoles.txt"
)
FREQ_HZ = 150e6

# 偶极子模型标识，必须与 esr_reconstruction.py 中的 MODEL_TAG 一致
MODEL_TAG = "Mx_My_magnetic_Pz_electric"


def ensure_output_dir():
    """创建 nn_refinement 输出目录。"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    return OUTPUT_ROOT


# ---------------------------------------------------------------------------
# 偶极子阵列读写
# ---------------------------------------------------------------------------
def load_initial_dipoles(txt_path=INITIAL_DIPOLES_TXT):
    """
    读取初步偶极子阵列 TXT 文件。

    列格式：x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)
    返回：
        positions : (N, 3) float64
        moments   : (N, 3) complex128
    """
    positions = []
    moments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        vals = line.split()
        if len(vals) < 9:
            continue
        positions.append([float(vals[0]), float(vals[1]), float(vals[2])])
        moments.append([
            complex(float(vals[3]), float(vals[4])),
            complex(float(vals[5]), float(vals[6])),
            complex(float(vals[7]), float(vals[8])),
        ])
    return np.array(positions), np.array(moments, dtype=complex)


def save_dipoles_txt(txt_path, positions, moments):
    """
    保存偶极子阵列 TXT，格式与初步偶极子阵列一致。
    """
    header = "x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for pos, m in zip(positions, moments):
            row = (
                f"{pos[0]:.8E} {pos[1]:.8E} {pos[2]:.8E} "
                f"{m[0].real:.8E} {m[0].imag:.8E} "
                f"{m[1].real:.8E} {m[1].imag:.8E} "
                f"{m[2].real:.8E} {m[2].imag:.8E}"
            )
            f.write(row + "\n")


# ---------------------------------------------------------------------------
# T_env 读取与观测点选择
# ---------------------------------------------------------------------------
def load_tenv(path=T_ENV_PATH):
    """
    读取 T_env.npz，并校验模型标识。
    返回字典视图（NpzFile），包含 T / obs_coords / src_coords / plane_offsets 等。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"T_env 文件不存在: {path}")
    data = np.load(path, allow_pickle=True)
    if "model" not in data or str(data["model"]) != MODEL_TAG:
        raise ValueError("T_env 模型标识不匹配，请用 generate_T_env.py 重新生成")
    return data


def compute_nn_obs_indices(plane_sizes, plane_offsets, n_target):
    """
    在 T_env 对应的抽稀观测点中，按每个扫描面等比例选择神经网络输入点。

    参数：
        plane_sizes    : 每个扫描面抽稀后的观测点数
        plane_offsets  : 每个扫描面在总观测点向量中的起始索引
        n_target       : 希望选取的观测点总数
    返回：
        obs_indices    : 选中的观测点索引（升序）
    """
    total = sum(plane_sizes)
    indices = []
    for pi, size in enumerate(plane_sizes):
        if size <= 0:
            continue
        n_plane = max(1, int(round(n_target * size / total)))
        n_plane = min(n_plane, size)
        # 用等间隔方式均匀覆盖该面的抽稀观测点
        chosen = np.unique(np.linspace(0, size - 1, n_plane).astype(int))
        indices.append(plane_offsets[pi] + chosen)
    return np.sort(np.concatenate(indices))


def obs_indices_to_tenv_rows(obs_indices):
    """
    把观测点索引展开成 T_env 的行索引。
    T_env 每 3 行对应一个观测点的 Hx/Hy/Hz。
    """
    rows = np.concatenate([
        np.asarray([3 * i, 3 * i + 1, 3 * i + 2]) for i in obs_indices
    ])
    return np.sort(rows)


def select_tenv_columns(tenv, positions, obs_indices):
    """
    按神经网络观测点和初步偶极子位置，取出 T_env 的子矩阵。

    返回：
        T_sub    : (3*n_obs_points, 3*n_dipoles) 复数矩阵
        src_cols : 初步偶极子对应的 T_env 列索引
        rows     : 观测点对应的 T_env 行索引
    """
    rows = obs_indices_to_tenv_rows(obs_indices)
    src_coords = np.asarray(tenv["src_coords"])
    src_cols = []
    for pos in positions:
        match = np.where(np.all(np.isclose(src_coords, pos), axis=1))[0]
        if len(match) != 1:
            raise RuntimeError(f"偶极子位置不在 T_env 候选网格中: {pos}")
        src_idx = int(match[0])
        src_cols.extend([3 * src_idx, 3 * src_idx + 1, 3 * src_idx + 2])
    src_cols = np.asarray(src_cols, dtype=int)
    T_sub = np.asarray(tenv["T"])[rows][:, src_cols]
    return T_sub, src_cols, rows


# ---------------------------------------------------------------------------
# 特征与目标转换
# ---------------------------------------------------------------------------
def fields_to_features(fields):
    """
    复磁场向量 -> 神经网络输入特征。
    特征顺序：先全部实部，再全部虚部。
    """
    return np.concatenate([fields.real, fields.imag]).astype(np.float32)


def moments_to_targets(moments_batch):
    """
    复偶极矩批量 (n_samples, N, 3) -> 神经网络目标。
    目标顺序：先全部实部，再全部虚部，长度 6*N。
    """
    n = moments_batch.shape[0]
    return np.concatenate([
        moments_batch.real.reshape(n, -1),
        moments_batch.imag.reshape(n, -1),
    ], axis=1).astype(np.float32)


def targets_to_moments(y):
    """
    神经网络输出 -> (N, 3) 复偶极矩。
    """
    y = np.asarray(y).ravel()
    n3 = len(y) // 2
    real = y[:n3].reshape(-1, 3)
    imag = y[n3:].reshape(-1, 3)
    return real + 1j * imag


# ---------------------------------------------------------------------------
# 磁场预测与 .hfe 格式 TXT 输出
# ---------------------------------------------------------------------------
def predict_fields_full(positions, moments, coords, freq=FREQ_HZ):
    """
    用自由空间混合偶极子模型计算完整观测面上的磁场。
    返回 (N, 3) complex128。
    """
    T_full = esr.build_free_space_T(coords, positions, freq)
    return (T_full @ moments.reshape(-1)).reshape(-1, 3)


def write_predicted_hfe_txt(txt_path, coords, H, counts, request_name,
                            source_label="EquivalentSR_Prediction"):
    """
    把预测磁场写成与 FEKO .hfe 相同格式的 TXT 文件。

    文件包含：
    - 以 # 或 ** 开头的表头；
    - 采样点数、频率、坐标系等元信息；
    - 数据行：X Y Z Re(Hx) Im(Hx) Re(Hy) Im(Hy) Re(Hz) Im(Hz)
    """
    nx, ny, nz = counts
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_lines = [
        "##File Type: Magnetic Near Field",
        "##File Format: 8",
        f"##Source: {source_label}",
        f"##Date: {now}",
        "** Generated by Equivalent Source Reconstruction pipeline",
        "",
        "#Configuration Name: EquivalentSourceModel",
        f"#Request Name: {request_name}",
        "#Frequency:   1.50000000E+008",
        "#Coordinate System: Cartesian",
        f"#No. of X Samples: {nx}",
        f"#No. of Y Samples: {ny}",
        f"#No. of Z Samples: {nz}",
        "#Result Type: Magnetic Field Values",
        "#No. of Header Lines: 1",
        '#         "X"                "Y"                "Z"              '
        '"Re(Hx)"           "Im(Hx)"           "Re(Hy)"           "Im(Hy)"           '
        '"Re(Hz)"           "Im(Hz)"',
    ]
    with open(txt_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(header_lines) + "\n")
        for c, h in zip(coords, H):
            f.write(
                f"{c[0]: .8E} {c[1]: .8E} {c[2]: .8E} "
                f"{h[0].real: .8E} {h[0].imag: .8E} "
                f"{h[1].real: .8E} {h[1].imag: .8E} "
                f"{h[2].real: .8E} {h[2].imag: .8E}\n"
            )
