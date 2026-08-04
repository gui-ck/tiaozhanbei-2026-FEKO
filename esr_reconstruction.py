# -*- coding: utf-8 -*-
"""
复杂电磁环境下的等效磁偶极子阵列重构主程序
=============================================

功能说明
--------
1. 读取 ``Data_Scanning_Field`` 目录下三个近场扫描面（xoy: 上方面、xoz: 前方面、
   yoz: 右方面）的 FEKO ``.hfe`` 复磁场数据（150 MHz）。
2. 在机箱内部生成候选偶极子位置。默认采用机箱居中 80% 范围、5 cm 三维网格；
   如果后续能从 CAD 几何提取机箱壁厚，则替换为按内表面内缩后的范围。
3. 建立观测点到候选偶极子的环境感知传递矩阵 ``T_env``：

   - 回退模式（fallback）：使用自由空间偶极子格林函数构建 ``T_env``，
     该矩阵只作为环境感知矩阵不可用时的近似；
   - FEKO 模式（feko）：调用 ``feko_esr_tools.py`` 对每个单位偶极子做 FEKO
     仿真，机箱屏蔽、通风孔泄漏和周围设备反射都会被包含在 ``T_env`` 中。
4. 使用组稀疏回归（Group OMP）求解偶极子位置、数量和复偶极矩，
    并用每个扫描面 20% 留出点交叉验证确定偶极子数量。
   偶极子模型为：x/y 方向使用磁偶极矩 (Mx, My)，z 方向使用电偶极矩 (Pz)；
   FEKO 中分别对应 A6 磁偶极子卡片和 A5 电偶极子卡片。
5. 对入选位置做局部网格精化、矩重拟合、合并与裁剪。
6. 在完整三个扫描面上计算重构磁场，与真实源磁场逐点对比，
   输出相对 L2 误差、分量误差和 Hx/Hy/Hz 分量场图。

输出目录
--------
所有结果统一保存在项目根目录下的 ``Output_ESR`` 子文件夹：

- ``equivalent_dipoles.txt`` : 偶极子主结果，列格式为
  ``x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)``，详见
  :func:`save_equivalent_dipoles_txt`；
- ``T_env.npz``              : 环境感知传递矩阵（自由空间回退或 FEKO 生成）；
- ``dipole_metadata.json``   : 求解参数和元数据；
- ``validation_errors.json`` : 数值误差结果；
- ``validation_report.txt``  : 中文验证报告；
- ``field_comparison_*.png`` : 三个扫描面 Hx/Hy/Hz 分量对比图。

用法
----
.. code-block:: text

    python esr_reconstruction.py --mode fallback
    python esr_reconstruction.py --mode feko --feko-bin D:/FEKO2025/feko/bin

说明：FEKO 模式需要逐个单元偶极子做仿真，仿真次数等于候选点数乘 3，
运行时间较长，建议在集群或后台批量执行。
"""

import argparse
import datetime
import json
import os
import sys
import time

import matplotlib

# 使用无界面后端，保证在没有显示器的环境下也能保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 保证 Windows 控制台能正确显示中文输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Scanning_Field")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Output_ESR")
T_ENV_PATH = os.path.join(OUTPUT_DIR, "T_env.npz")

# 工作频率与光速（单位：Hz 和 m/s）
FREQ_HZ = 150e6
C0 = 299792458.0

# 偶极子模型标识：x/y 为磁偶极矩，z 为电偶极矩。
# 该标识写入 T_env.npz，防止旧版（z 方向为磁偶极矩）矩阵被误复用。
MODEL_TAG = "Mx_My_magnetic_Pz_electric"

# 三个扫描面的定义。文件来自 FEKO 真实源模型 Model_real。
# xoy: 常 z 面，z = -0.08 m，法向为 z，对应“上方面”；
# xoz: 常 y 面，y = -0.02 m，法向为 y，对应“前方面”；
# yoz: 常 x 面，x = -6.2 m，法向为 x，对应“右方面”。
PLANE_DEFS = [
    {
        "name": "xoy",
        "file": "Model_real_NearField_xoy.hfe",
        "desc": "上方面（法向 z，z=-0.08 m）",
    },
    {
        "name": "xoz",
        "file": "Model_real_NearField_xoz.hfe",
        "desc": "前方面（法向 y，y=-0.02 m）",
    },
    {
        "name": "yoz",
        "file": "Model_real_NearField_yoz.hfe",
        "desc": "右方面（法向 x，x=-6.2 m）",
    },
]

# 机箱外尺寸（单位 m）。用户给定尺寸为毫米，这里统一换算成米：
# x=[-6555,-6280] mm -> [-6.555,-6.280] m
# y=[-260,-41]   mm -> [-0.260,-0.041] m
# z=[-500,-115]  mm -> [-0.500,-0.115] m
BOX_BOUNDS = {
    "x": (-6.555, -6.280),
    "y": (-0.260, -0.041),
    "z": (-0.500, -0.115),
}

# 无法从 CAD 几何读取壁厚时，偶极子候选范围取机箱居中 80%
CANDIDATE_SHRINK_RATIO = 0.8

# 初始候选网格间距（m）
CANDIDATE_SPACING = 0.05

# 观测点抽稀参数：均匀每隔 5 个点取 1 个；同时保留幅值最高的前 5% 峰值点
OBS_SUBSAMPLE_STEP = 5
PEAK_KEEP_FRACTION = 0.05

# 交叉验证参数：每个面随机留出 20% 的点用于验证
CV_TEST_FRACTION = 0.2
CV_SEED = 12345

# 稀疏求解的偶极子数量范围（按候选点计，每个点 3 个磁偶极矩分量）
MIN_SUPPORT = 10
MAX_SUPPORT = 175

# 位置精化参数：局部搜索步长 1 cm，每个方向搜索 ±1 cm 和 ±2 cm
REFINE_STEP = 0.01
REFINE_PASSES = 2

# 场图显示平均：原始数据在正确的重排顺序下本身是平滑的，默认不再做块平均。
DISPLAY_AVERAGE = 1

# 最终偶极子处理参数：合并距离 5 mm，删除矩幅小于最大矩幅 0.1% 的偶极子。
# 实验表明按 1% 裁剪会把 129 个偶极子减到 54 个，导致整体误差从 13% 恶化到 25%；
# 这里取 0.1%，既保留主要物理偶极子，又不明显牺牲重构精度。
MERGE_DIST = 0.005
PRUNE_REL_THRESHOLD = 0.001


# ---------------------------------------------------------------------------
# 数据读取与预处理
# ---------------------------------------------------------------------------
def read_hfe(filepath):
    """
    读取 FEKO ``.hfe`` 近场磁场文件。

    文件格式说明（每一行数据列）：
    ``X Y Z Re(Hx) Im(Hx) Re(Hy) Im(Hy) Re(Hz) Im(Hz)``

    返回：
        coords : (N, 3) float64，观测点坐标
        H      : (N, 3) complex128，复磁场
        counts : (nx, ny, nz) 三个采样轴的点数，用于二维场图重排
    """
    coords = []
    H = []
    counts = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # 解析文件头中的采样点数，例如 "#No. of X Samples: 201"
            if line.startswith("#No. of X Samples"):
                counts = [int(line.split(":")[1])]
                continue
            if line.startswith("#No. of Y Samples"):
                counts.append(int(line.split(":")[1]))
                continue
            if line.startswith("#No. of Z Samples"):
                counts.append(int(line.split(":")[1]))
                continue

            # 跳过注释行和空行
            if not line or line.startswith("#") or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
                hx = complex(float(parts[3]), float(parts[4]))
                hy = complex(float(parts[5]), float(parts[6]))
                hz = complex(float(parts[7]), float(parts[8]))
                coords.append([x, y, z])
                H.append([hx, hy, hz])
            except ValueError:
                # 个别非数据行直接跳过，不影响整体读取
                continue

    if counts is None:
        raise ValueError("未能在文件头找到采样点数，请检查 .hfe 文件格式")
    return np.array(coords), np.array(H, dtype=complex), tuple(counts)


def compute_subsample_indices(n_points, mag, step=OBS_SUBSAMPLE_STEP,
                              keep_fraction=PEAK_KEEP_FRACTION):
    """
    计算单个扫描面的抽稀索引，控制传递矩阵规模。

    策略：
    - 均匀每隔 ``step`` 个点取 1 个点；
    - 同时保留 |H| 幅值最高的前 ``keep_fraction`` 比例的峰值点，
      峰值通常对应通风孔泄漏最强的区域，不能被抽稀丢掉。
    """
    uniform_idx = np.arange(0, n_points, step)
    n_peak = max(1, int(round(keep_fraction * n_points)))
    peak_idx = np.argsort(mag)[-n_peak:]
    return np.unique(np.concatenate([uniform_idx, peak_idx]))


def subsample_plane(coords, H, step=OBS_SUBSAMPLE_STEP,
                    keep_fraction=PEAK_KEEP_FRACTION):
    """
    对单个扫描面做抽稀，返回抽稀后的坐标和磁场。
    """
    mag = np.abs(H).max(axis=1)
    keep_idx = compute_subsample_indices(
        len(coords), mag, step, keep_fraction
    )
    return coords[keep_idx], H[keep_idx]


def load_all_planes():
    """
    读取全部三个扫描面的数据，返回每个面的字典：
    name / coords / H / counts / desc
    """
    planes = []
    for pdef in PLANE_DEFS:
        path = os.path.join(DATA_DIR, pdef["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"扫描数据文件不存在: {path}")
        coords, H, counts = read_hfe(path)
        planes.append(
            {
                "name": pdef["name"],
                "desc": pdef["desc"],
                "coords": coords,
                "H": H,
                "counts": counts,
            }
        )
        print(f"读取 {pdef['name']}: {len(coords)} 个观测点, counts={counts}")
    return planes


# ---------------------------------------------------------------------------
# 候选偶极子网格
# ---------------------------------------------------------------------------
def get_candidate_bounds(box=BOX_BOUNDS, shrink=CANDIDATE_SHRINK_RATIO):
    """
    计算偶极子候选范围。

    当无法从 CAD 几何提取机箱壁厚时，取机箱三个方向各自的居中 80% 范围，
    相当于自动向内收缩壁厚。
    """
    bounds = {}
    for axis, (lo, hi) in box.items():
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * shrink
        bounds[axis] = (center - half, center + half)
    return bounds


def build_candidate_grid(spacing=CANDIDATE_SPACING):
    """
    在候选范围内生成三维规则网格，返回 (N_src, 3) 的偶极子位置。
    每个网格点后续对应 3 个正交磁偶极矩分量 (mx, my, mz)。
    """
    bounds = get_candidate_bounds()
    xs = np.arange(bounds["x"][0], bounds["x"][1] + spacing / 2, spacing)
    ys = np.arange(bounds["y"][0], bounds["y"][1] + spacing / 2, spacing)
    zs = np.arange(bounds["z"][0], bounds["z"][1] + spacing / 2, spacing)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    print(
        f"候选网格: x {len(xs)} 点, y {len(ys)} 点, z {len(zs)} 点, "
        f"共 {len(points)} 个偶极子候选点"
    )
    return points


# ---------------------------------------------------------------------------
# 偶极子格林函数（自由空间回退方案）
# ---------------------------------------------------------------------------
def build_magnetic_T(obs_points, src_points, comps, freq=FREQ_HZ):
    """
    构建自由空间磁偶极子格林函数传递矩阵的指定分量。

    物理公式（磁偶极子 m 在自由空间产生的磁场 H，时谐因子 e^{-jkr}）：

    .. math::

        H(r) = \\frac{e^{-jkr}}{4\\pi}
               \\left[
                   \\frac{k^2}{R}(I - \\hat{R}\\hat{R})
                   + \\left(\\frac{1}{R^3} + \\frac{jk}{R^2}\\right)
                   (3\\hat{R}\\hat{R} - I)
               \\right] \\cdot m

    其中 R = |r - r'|，\\hat{R} = (r - r')/R，k = 2\\pi f/c。
    """
    k = 2.0 * np.pi * freq / C0
    n_obs = len(obs_points)
    n_src = len(src_points)

    # 观测点到源点的向量 R_vec: (n_obs, n_src, 3)
    R_vec = obs_points[:, None, :] - src_points[None, :, :]
    R = np.linalg.norm(R_vec, axis=2)
    # 防止源点与观测点重合时除零
    R[R == 0.0] = 1e-12
    R_hat = R_vec / R[:, :, None]

    # 张量积 R_hat * R_hat^T
    R_hat_tensor = R_hat[:, :, :, None] * R_hat[:, :, None, :]
    I3 = np.eye(3)

    # 公式中的两项（都是 (n_obs, n_src, 3, 3) 张量）
    term1 = (k**2 / R[:, :, None, None]) * (I3[None, None, :, :] - R_hat_tensor)
    term2 = (1.0 / R[:, :, None, None] ** 3
             + 1j * k / R[:, :, None, None] ** 2) * (3.0 * R_hat_tensor - I3[None, None, :, :])

    G = (np.exp(-1j * k * R[:, :, None, None]) / (4.0 * np.pi)) * (term1 + term2)

    # 只保留需要的磁偶极矩分量
    G = G[:, :, :, list(comps)]
    # 重排为 (n_obs*3, n_src*len(comps))：行顺序为 Hx,Hy,Hz 交替
    return G.transpose(0, 2, 1, 3).reshape(n_obs * 3, n_src * len(comps))


def build_electric_T(obs_points, src_points, comps, freq=FREQ_HZ):
    """
    构建自由空间电偶极子格林函数传递矩阵的指定分量。

    物理公式（电偶极子 p 在自由空间产生的磁场 H，时谐因子 e^{-jkr}）：

    .. math::

        H(r) = \\frac{e^{-jkr}}{4\\pi}
               \\left(-\\frac{jk}{R} - \\frac{1}{R^2}\\right)
               (\\hat{R} \\times p)

    其中 R = |r - r'|，\\hat{R} = (r - r')/R，k = 2\\pi f/c。
    电偶极矩 p 的单位为 A*m，与磁偶极矩的单位 A*m^2 不同。
    """
    k = 2.0 * np.pi * freq / C0
    n_obs = len(obs_points)
    n_src = len(src_points)

    R_vec = obs_points[:, None, :] - src_points[None, :, :]
    R = np.linalg.norm(R_vec, axis=2)
    R[R == 0.0] = 1e-12
    R_hat = R_vec / R[:, :, None]

    # 标量因子 s = e^{-jkr}/(4π) * (-jk/R - 1/R^2)
    s = (np.exp(-1j * k * R) / (4.0 * np.pi)
         * (-1j * k / R - 1.0 / R**2))

    # 叉乘矩阵 C[r, i] = (R_hat x p)_r / p_i
    C = np.zeros((n_obs, n_src, 3, 3), dtype=complex)
    C[:, :, 0, 1] = R_hat[:, :, 2]
    C[:, :, 0, 2] = -R_hat[:, :, 1]
    C[:, :, 1, 0] = -R_hat[:, :, 2]
    C[:, :, 1, 2] = R_hat[:, :, 0]
    C[:, :, 2, 0] = R_hat[:, :, 1]
    C[:, :, 2, 1] = -R_hat[:, :, 0]

    G = np.zeros((n_obs, n_src, 3, 3), dtype=complex)
    for c_idx in comps:
        G[:, :, :, c_idx] = s[:, :, None] * C[:, :, :, c_idx]

    G = G[:, :, :, list(comps)]
    return G.transpose(0, 2, 1, 3).reshape(n_obs * 3, n_src * len(comps))


def build_free_space_T(obs_points, src_points, freq=FREQ_HZ,
                       magnetic_comps=(0, 1), electric_comps=(2,)):
    """
    构建混合偶极子自由空间传递矩阵 T。

    每个候选点的列顺序固定为：Mx(磁)、My(磁)、Pz(电)。

    - 磁偶极矩分量使用 :func:`build_magnetic_T`；
    - 电偶极矩分量使用 :func:`build_electric_T`。

    该自由空间矩阵仅作为回退和初始粗选支持集使用；机箱周围存在反射设备时，
    最终应使用 FEKO 生成的环境感知矩阵 ``T_env``。
    """
    parts = []
    if magnetic_comps:
        parts.append(build_magnetic_T(
            obs_points, src_points, list(magnetic_comps), freq
        ))
    if electric_comps:
        parts.append(build_electric_T(
            obs_points, src_points, list(electric_comps), freq
        ))
    if len(parts) == 1:
        return parts[0]
    return np.hstack(parts)


def single_site_columns(obs_points, position, freq=FREQ_HZ):
    """
    返回单个候选位置的 3 列自由空间传递矩阵（对应 Mx, My, Pz）。
    位置精化时反复调用该函数生成局部扰动后的新列。
    """
    return build_free_space_T(obs_points, np.asarray([position]), freq)


# ---------------------------------------------------------------------------
# 组稀疏求解
# ---------------------------------------------------------------------------
def group_omp_run(T, b, n_groups, T_val=None, b_val=None, gsize=3):
    """
    组正交匹配追踪（Group OMP）。

    - 组定义为同一个偶极子位置的 3 个偶极矩分量（Mx, My, Pz）；
    - 每次迭代用 ``||T_g^H r||`` 衡量候选组与残差的相关性，选择相关性最大的一组；
    - 对所有已选组做一次最小二乘重拟合，更新残差；
    - 若传入验证矩阵，则同步记录每一步的验证集相对误差，用于选择偶极子数量。
    """
    n_sites = T.shape[1] // gsize
    n_groups = min(n_groups, n_sites)

    residual = b.copy()
    selected = []
    train_errors = []
    val_errors = []
    x = None

    for step in range(n_groups):
        # 一次性计算所有组的相关性，避免 Python 循环内逐组做矩阵乘法
        th_r = T.conj().T @ residual
        corr = np.linalg.norm(th_r.reshape(-1, gsize), axis=1)
        best_group = int(np.argmax(corr))

        if best_group not in selected:
            selected.append(best_group)

        # 组装已选组的全部列
        cols = []
        for g in selected:
            cols.extend(range(g * gsize, (g + 1) * gsize))
        cols = np.asarray(cols, dtype=int)

        # 对已选列做最小二乘，更新系数和残差
        T_S = T[:, cols]
        x, _, _, _ = np.linalg.lstsq(T_S, b, rcond=None)
        residual = b - T_S @ x

        train_errors.append(np.linalg.norm(residual) / np.linalg.norm(b))

        if T_val is not None and b_val is not None:
            val_error = (
                np.linalg.norm(b_val - T_val[:, cols] @ x)
                / np.linalg.norm(b_val)
            )
            val_errors.append(val_error)

    return selected, x, np.asarray(train_errors), np.asarray(val_errors)


def choose_support_size(val_errors, min_support=MIN_SUPPORT,
                        tolerance=0.02):
    """
    根据验证集误差曲线选择偶极子数量。

    选择规则：
    - 找到验证误差最小的位置；
    - 在最小误差的 2% 容差内优先选择更小的支持集（偶极子更少、更可解释）。
    """
    if len(val_errors) == 0:
        # 没有验证误差曲线时退回最小支持数
        return min_support

    best_idx = int(np.argmin(val_errors))
    best_val = val_errors[best_idx]
    candidates = [
        i for i, e in enumerate(val_errors)
        if e <= best_val * (1.0 + tolerance)
    ]
    # 返回偶极子个数 = 最优迭代步数
    chosen = max(min_support, candidates[0] + 1)
    return chosen


# ---------------------------------------------------------------------------
# 位置精化
# ---------------------------------------------------------------------------
def refine_positions(obs_points, b, positions, T_S, bounds,
                     get_columns=single_site_columns,
                     step=REFINE_STEP, passes=REFINE_PASSES):
    """
    对入选偶极子位置做局部贪婪精化。

    思路：
    - 保持当前支持集不变，逐个偶极子尝试沿 x/y/z 移动 ±1 cm、±2 cm；
    - 用“残差 - (新列-旧列)*当前矩”近似估计移动后的残差，选择残差最小的位置；
    - 完成一轮扫描后做一次整体最小二乘重拟合，重复若干轮；
    - 该方法只生成少量新传递矩阵列，适合后续扩展成 FEKO 环境感知列。

    注意：当前实现使用自由空间格林函数生成新列；使用 FEKO 时，把
    ``get_columns`` 替换为调用 FEKO 单元偶极子仿真获取环境感知列即可。
    """
    n_dipoles = len(positions)

    # 初始最小二乘解和残差
    x, _, _, _ = np.linalg.lstsq(T_S, b, rcond=None)
    residual = b - T_S @ x

    for _ in range(passes):
        moved = False
        for g in range(n_dipoles):
            col_slice = slice(3 * g, 3 * (g + 1))
            A_old = T_S[:, col_slice].copy()
            x_g = x[col_slice]

            best_residual = residual
            best_A = A_old
            best_pos = positions[g].copy()

            # 沿三个坐标轴搜索 ±1 cm、±2 cm 的候选位置
            for axis in range(3):
                for delta in (-2 * step, -step, step, 2 * step):
                    p_new = positions[g].copy()
                    p_new[axis] += delta

                    # 新位置必须仍位于候选范围内
                    if not (bounds["x"][0] <= p_new[0] <= bounds["x"][1]
                            and bounds["y"][0] <= p_new[1] <= bounds["y"][1]
                            and bounds["z"][0] <= p_new[2] <= bounds["z"][1]):
                        continue

                    A_new = get_columns(obs_points, p_new)
                    # 近似更新残差：只移动当前偶极子，其它偶极子暂不变
                    cand_residual = residual - (A_new - A_old) @ x_g
                    if np.linalg.norm(cand_residual) < np.linalg.norm(best_residual):
                        best_residual = cand_residual
                        best_A = A_new
                        best_pos = p_new

            # 若找到了更好的位置，则替换该偶极子的列与位置
            if not np.allclose(best_pos, positions[g]) or best_A is not A_old:
                T_S[:, col_slice] = best_A
                positions[g] = best_pos
                residual = best_residual
                moved = True

        if not moved:
            break

    # 精化结束后对最终支持集做整体最小二乘，重估全部偶极矩
    x, _, _, _ = np.linalg.lstsq(T_S, b, rcond=None)
    return positions, x, T_S


def merge_prune_dipoles(positions, moments,
                        merge_dist=MERGE_DIST,
                        rel_threshold=PRUNE_REL_THRESHOLD):
    """
    合并与裁剪偶极子：

    - 删除矩幅小于最大矩幅 1% 的偶极子；
    - 将距离小于 ``merge_dist`` 的偶极子按复矩向量相加合并，降低偶极子数量。
    """
    if len(positions) == 0:
        return positions, moments

    mag = np.abs(moments).max(axis=1)
    keep = mag >= rel_threshold * mag.max()
    positions = positions[keep]
    moments = moments[keep]

    # 按矩幅从大到小排序，优先保留强偶极子
    order = np.argsort(np.abs(moments).max(axis=1))[::-1]
    positions = positions[order]
    moments = moments[order]

    kept_pos = []
    kept_mom = []
    used = np.zeros(len(positions), dtype=bool)
    for i in range(len(positions)):
        if used[i]:
            continue
        used[i] = True
        p0 = positions[i]
        m0 = moments[i].copy()
        for j in range(i + 1, len(positions)):
            if used[j]:
                continue
            if np.linalg.norm(positions[j] - p0) <= merge_dist:
                # 复矩向量直接相加，近似合并近距离偶极子的贡献
                m0 = m0 + moments[j]
                used[j] = True
        kept_pos.append(p0)
        kept_mom.append(m0)

    return np.array(kept_pos), np.array(kept_mom)


# ---------------------------------------------------------------------------
# 误差计算、场图与文件输出
# ---------------------------------------------------------------------------
def compute_validation_errors(planes, positions, moments, freq=FREQ_HZ):
    """
    在完整三个扫描面上计算重构磁场，并与真实源磁场比较。

    返回的误差字典包含：
    - overall: 整体相对 L2 误差；
    - planes:  逐面相对 L2 误差和 Hx/Hy/Hz 分量误差。
    """
    result = {"overall": {}, "planes": {}}

    b_all = []
    b_rec_all = []
    for plane in planes:
        coords = plane["coords"]
        H_orig = plane["H"]
        T_full = build_free_space_T(coords, positions, freq)
        H_rec = (T_full @ moments.reshape(-1)).reshape(-1, 3)

        b_all.append(H_orig.ravel())
        b_rec_all.append(H_rec.ravel())

        rel_l2 = np.linalg.norm(H_orig - H_rec) / np.linalg.norm(H_orig)
        comp_errors = {}
        for c_idx, c_name in enumerate(["Hx", "Hy", "Hz"]):
            e = (
                np.linalg.norm(H_orig[:, c_idx] - H_rec[:, c_idx])
                / np.linalg.norm(H_orig[:, c_idx])
            )
            comp_errors[c_name] = round(float(e * 100.0), 4)

        result["planes"][plane["name"]] = {
            "rel_l2_error_percent": round(float(rel_l2 * 100.0), 4),
            "component_errors_percent": comp_errors,
            "n_obs": len(coords),
        }

    b_all = np.concatenate(b_all)
    b_rec_all = np.concatenate(b_rec_all)
    overall = np.linalg.norm(b_all - b_rec_all) / np.linalg.norm(b_all)
    result["overall"] = {
        "rel_l2_error_percent": round(float(overall * 100.0), 4),
        "n_obs": len(b_all) // 3,
    }
    return result


def plot_field_comparison(filepath, coords, H_orig, H_rec, counts, name):
    """
    绘制单个扫描面 Hx/Hy/Hz 三个分量的原始与重构对比图。

    counts 是 (nx, ny, nz)，其中只有一个轴为 1，squeeze 后得到二维场图。
    坐标轴标签根据扫描面名称给出，避免横纵坐标混淆。
    """
    axis_labels = {
        "xoy": ("X (m)", "Y (m)"),
        "xoz": ("X (m)", "Z (m)"),
        "yoz": ("Y (m)", "Z (m)"),
    }
    label_key = name.split("_")[0] if "_" in name else name
    x_label, y_label = axis_labels.get(
        label_key, ("axis1 (m)", "axis2 (m)")
    )
    comp_names = ["Hx", "Hy", "Hz"]

    varying_axes = [i for i, n in enumerate(counts) if n > 1]
    horiz_axis, vert_axis = varying_axes[0], varying_axes[1]

    # 关键修正：.hfe 文件的行顺序是“慢轴在外、快轴在内”。
    # 例如 xoy 面实际是 y 慢、x 快，因此二维数组应为 (ny, nx)，
    # 而不是按头部采样数顺序 (nx, ny) 直接 reshape，否则会产生假的细条纹。
    diff_first = coords[1] - coords[0]
    fast_axis = int(np.argmax(np.abs(diff_first)))
    slow_axis = [a for a in varying_axes if a != fast_axis][0]
    n_slow = int(counts[slow_axis])
    n_fast = int(counts[fast_axis])

    x0, x1 = float(coords[:, horiz_axis].min()), float(coords[:, horiz_axis].max())
    y0, y1 = float(coords[:, vert_axis].min()), float(coords[:, vert_axis].max())

    # 每个分量一行，左列为原始场，右列为重构场；
    # 使用 imshow + bicubic 平滑插值，显示效果更清晰。
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)

    for row, c_name in enumerate(comp_names):
        Z_orig = np.abs(H_orig[:, row]).reshape(n_slow, n_fast)
        Z_rec = np.abs(H_rec[:, row]).reshape(n_slow, n_fast)
        # 若最快变化轴不是横轴，则转置，保证行=纵轴、列=横轴
        if fast_axis != horiz_axis:
            Z_orig = Z_orig.T
            Z_rec = Z_rec.T
        # 显示用块平均：先截断到偶数尺寸，再按 2x2 取平均
        if DISPLAY_AVERAGE > 1:
            Z_orig = _block_average(Z_orig, DISPLAY_AVERAGE)
            Z_rec = _block_average(Z_rec, DISPLAY_AVERAGE)
        vmin = min(Z_orig.min(), Z_rec.min())
        vmax = max(Z_orig.max(), Z_rec.max())

        # imshow 的第一个轴是图像行（对应 y），第二个轴是列（对应 x），
        # 因此这里必须转置：Z.T 的形状为 (y点数, x点数)。
        im0 = axes[row, 0].imshow(
            Z_orig,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        axes[row, 0].set_title(f"{name} Original |{c_name}|")
        axes[row, 0].set_xlabel(x_label)
        axes[row, 0].set_ylabel(y_label)

        im1 = axes[row, 1].imshow(
            Z_rec,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
        )
        axes[row, 1].set_title(f"{name} Reconstructed |{c_name}|")
        axes[row, 1].set_xlabel(x_label)
        axes[row, 1].set_ylabel(y_label)
        # 每行（同一个分量）只保留右侧一个色标，减少色标过多造成的视觉混乱
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.03)

    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _block_average(Z, k):
    """
    把二维数组按 k x k 的窗口做块平均，用于显示降采样。
    只取能被 k 整除的尺寸，避免边界错位。
    """
    h = Z.shape[0] - Z.shape[0] % k
    w = Z.shape[1] - Z.shape[1] % k
    return Z[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def save_equivalent_dipoles_txt(filepath, positions, moments):
    """
    保存偶极子阵列 TXT 文件。

    列格式（每行一个偶极子）：
    ``x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)``

    坐标单位：m；Mx/My 为磁偶极矩，单位 A*m^2；Pz 为电偶极矩，单位 A*m。
    文件为空格分隔纯文本，首行为表头。
    """
    header = (
        "x y z Re(Mx) Im(Mx) Re(My) Im(My) Re(Pz) Im(Pz)"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for pos, m in zip(positions, moments):
            row = (
                f"{pos[0]:.8E} {pos[1]:.8E} {pos[2]:.8E} "
                f"{m[0].real:.8E} {m[0].imag:.8E} "
                f"{m[1].real:.8E} {m[1].imag:.8E} "
                f"{m[2].real:.8E} {m[2].imag:.8E}"
            )
            f.write(row + "\n")


def save_validation_outputs(planes, positions, moments, meta):
    """
    计算误差、画场图，并写出 validation_errors.json、validation_report.txt
    和 field_comparison_*.png。
    """
    errors = compute_validation_errors(planes, positions, moments)

    # 写数值误差 JSON
    errors_path = os.path.join(OUTPUT_DIR, "validation_errors.json")
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    # 写三面场图
    for plane in planes:
        coords = plane["coords"]
        T_full = build_free_space_T(coords, positions)
        H_rec = (T_full @ moments.reshape(-1)).reshape(-1, 3)
        png_path = os.path.join(
            OUTPUT_DIR, f"field_comparison_{plane['name']}.png"
        )
        plot_field_comparison(
            png_path, coords, plane["H"], H_rec,
            plane["counts"], plane["name"]
        )

    # 写中文验证报告
    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    lines = []
    lines.append("等效磁偶极子阵列重构验证报告")
    lines.append("=" * 50)
    lines.append("偶极子模型: Mx/My 磁偶极子 + Pz 电偶极子")
    lines.append(f"生成时间: {meta['created_at']}")
    lines.append(f"工作频率: {meta['frequency_hz'] / 1e6:.1f} MHz")
    lines.append(f"T_env 来源: {meta['tenv_source']}")
    lines.append(f"最终偶极子数量: {len(positions)}")
    lines.append("")
    lines.append(f"整体相对 L2 误差: "
                 f"{errors['overall']['rel_l2_error_percent']:.4f}%")
    for name, per_plane in errors["planes"].items():
        lines.append(
            f"  {name} 面: {per_plane['rel_l2_error_percent']:.4f}%"
            f"  (Hx {per_plane['component_errors_percent']['Hx']:.4f}%, "
            f"Hy {per_plane['component_errors_percent']['Hy']:.4f}%, "
            f"Hz {per_plane['component_errors_percent']['Hz']:.4f}%)"
        )
    lines.append("")
    lines.append("说明：若 T_env 为自由空间回退矩阵，该误差是自由空间模型的上限；")
    lines.append("改用 FEKO 生成环境感知 T_env 后，机箱反射与通风孔泄漏会被纳入模型，")
    lines.append("误差预计可进一步下降。")

    # 使用 utf-8-sig 写入 BOM，避免 Windows 记事本/旧工具按 ANSI 解码中文乱码
    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return errors


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def save_metadata(meta):
    """把求解元数据写入 Output_ESR/dipole_metadata.json。"""
    path = os.path.join(OUTPUT_DIR, "dipole_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_existing_T_env(expected_src_positions, expected_obs_n):
    """
    尝试加载已存在的 T_env.npz。

    只有当候选点坐标和观测点数与当前配置完全一致时才复用，
    避免用旧矩阵配上新网格造成错误结果。
    """
    if not os.path.exists(T_ENV_PATH):
        return None
    data = np.load(T_ENV_PATH, allow_pickle=True)
    if "T" not in data or "src_coords" not in data or "obs_coords" not in data:
        return None
    # 模型版本检查：z 方向必须为电偶极矩 Pz，不能复用旧版 Mz 磁偶极矩矩阵
    if "model" not in data or str(data["model"]) != MODEL_TAG:
        return None
    if data["src_coords"].shape != expected_src_positions.shape:
        return None
    if data["obs_coords"].shape[0] != expected_obs_n:
        return None
    if not np.allclose(data["src_coords"], expected_src_positions):
        return None
    return data


def main():
    parser = argparse.ArgumentParser(
        description="等效磁偶极子阵列重构主程序（含 FEKO 环境感知模式）"
    )
    parser.add_argument(
        "--mode",
        choices=["fallback", "feko"],
        default="fallback",
        help="fallback=自由空间格林函数回退；feko=FEKO 环境感知 T_env",
    )
    parser.add_argument(
        "--feko-bin",
        default=r"D:\FEKO2025\feko\bin",
        help="FEKO 安装目录下的 bin 路径",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取三个扫描面
    print("=" * 60)
    print("步骤 1/6：读取扫描面数据")
    planes = load_all_planes()

    # 2. 生成候选偶极子网格
    print("=" * 60)
    print("步骤 2/6：生成候选偶极子网格")
    src_positions = build_candidate_grid()
    n_sites = len(src_positions)
    max_support = min(MAX_SUPPORT, n_sites)

    # 3. 对每个扫描面抽稀，并记录各面在总向量中的行范围
    print("=" * 60)
    print("步骤 3/6：观测点抽稀")
    sub_coords_list = []
    sub_H_list = []
    subsample_indices_list = []
    plane_offsets = []
    offset = 0
    for plane in planes:
        mag = np.abs(plane["H"]).max(axis=1)
        keep_idx = compute_subsample_indices(
            len(mag), mag, OBS_SUBSAMPLE_STEP, PEAK_KEEP_FRACTION
        )
        subsample_indices_list.append(keep_idx)
        coords_sub = plane["coords"][keep_idx]
        H_sub = plane["H"][keep_idx]
        sub_coords_list.append(coords_sub)
        sub_H_list.append(H_sub)
        plane_offsets.append(offset)
        offset += len(coords_sub)
        print(
            f"  {plane['name']}: 抽稀后 {len(coords_sub)} 个点 "
            f"(原始 {len(plane['coords'])} 个点)"
        )

    sub_coords = np.vstack(sub_coords_list)
    b = np.concatenate([H.ravel() for H in sub_H_list])
    n_obs_sub = len(sub_coords)

    # 4. 构建或加载 T_env
    print("=" * 60)
    print("步骤 4/6：构建 T_env")
    # FEKO 模式只复用 FEKO 生成的 T_env；自由空间回退矩阵不能冒充环境感知矩阵
    tenv_data = None
    if args.mode == "feko":
        cached = load_existing_T_env(src_positions, n_obs_sub)
        if cached is not None and str(cached["source"]) == "feko_unit_dipole":
            tenv_data = cached
    else:
        tenv_data = load_existing_T_env(src_positions, n_obs_sub)
    tenv_source = ""

    if tenv_data is not None:
        T = tenv_data["T"]
        tenv_source = str(tenv_data["source"])
        print(f"复用已有 T_env: {T.shape}, 来源={tenv_source}")
    elif args.mode == "feko":
        # FEKO 环境感知模式：逐个单元偶极子仿真生成 T_env
        import feko_esr_tools

        print("开始 FEKO 单元偶极子仿真（运行时间长，请耐心等待）...")
        T, _ = feko_esr_tools.generate_T_env_with_feko(
            src_positions=src_positions,
            plane_files=[os.path.join(DATA_DIR, p["file"]) for p in PLANE_DEFS],
            template_pre_path=os.path.join(
                PROJECT_ROOT, "Model_EQ_Surce", "Model_pred.pre"
            ),
            output_dir=OUTPUT_DIR,
            feko_bin=args.feko_bin,
            subsample_step=OBS_SUBSAMPLE_STEP,
            subsample_indices=subsample_indices_list,
        )
        tenv_source = "feko_unit_dipole"
    else:
        print("使用自由空间偶极子格林函数构建回退 T_env ...")
        T = build_free_space_T(sub_coords, src_positions)
        tenv_source = "free_space_fallback"

    # 保存 T_env（自由空间回退或 FEKO 生成的矩阵都单独保存，便于复用）
    np.savez_compressed(
        T_ENV_PATH,
        T=T,
        obs_coords=sub_coords,
        src_coords=src_positions,
        model=MODEL_TAG,
        moment_component_indices=np.tile([0, 1, 2], n_sites),
        component_types=np.tile(["M", "M", "E"], n_sites),
        frequency_hz=FREQ_HZ,
        source=tenv_source,
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        plane_offsets=np.array(plane_offsets),
        subsample_step=OBS_SUBSAMPLE_STEP,
    )
    print(f"T_env 已保存: {T_ENV_PATH} (shape={T.shape}, source={tenv_source})")

    # 5. 交叉验证 + 稀疏求解 + 位置精化
    print("=" * 60)
    print("步骤 5/6：稀疏求解与位置精化")

    # 按面做 80/20 留出切分，并转换为总行索引
    rng = np.random.default_rng(CV_SEED)
    rows_train = []
    rows_test = []
    for plane_idx, plane in enumerate(planes):
        n_plane = len(sub_coords_list[plane_idx])
        start = plane_offsets[plane_idx]
        idx = np.arange(n_plane)
        rng.shuffle(idx)
        n_test = max(1, int(round(CV_TEST_FRACTION * n_plane)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        rows_test.append(start + test_idx)
        rows_train.append(start + train_idx)

    rows_train = np.concatenate(rows_train)
    rows_test = np.concatenate(rows_test)
    T_train = T[rows_train]
    T_test = T[rows_test]
    b_train = b[rows_train]
    b_test = b[rows_test]

    print(f"训练点: {len(rows_train)}，验证点: {len(rows_test)}")
    _, _, _, val_errors = group_omp_run(
        T_train, b_train, max_support, T_test, b_test
    )
    n_dipoles = choose_support_size(val_errors)
    print(f"交叉验证选定偶极子数量: {n_dipoles}")

    # 用全部观测点做最终稀疏求解
    selected_groups, _, train_errors, _ = group_omp_run(
        T, b, n_dipoles
    )
    final_train_error = train_errors[-1]
    print(f"最终支持集偶极子数: {len(selected_groups)}, "
          f"训练相对误差: {final_train_error * 100:.3f}%")

    # 组装最终支持集的列，用于位置精化
    cols = []
    for g in selected_groups:
        cols.extend(range(g * 3, (g + 1) * 3))
    cols = np.asarray(cols, dtype=int)
    positions = src_positions[selected_groups].copy()
    T_S = T[:, cols]

    # 先记录精化前的支持集与误差，若精化反而变差则回退
    positions_before = positions.copy()
    T_S_before = T_S.copy()
    x_before, _, _, _ = np.linalg.lstsq(T_S_before, b, rcond=None)
    err_before = np.linalg.norm(b - T_S_before @ x_before) / np.linalg.norm(b)

    print("开始偶极子位置精化 ...")
    bounds = get_candidate_bounds()
    positions, moments, T_S = refine_positions(
        sub_coords, b, positions, T_S, bounds
    )
    # lstsq 返回一维矩向量，这里恢复成 (偶极子数, 3) 的矩阵形状
    moments = moments.reshape(-1, 3)
    err_refined = np.linalg.norm(b - T_S @ moments.reshape(-1)) / np.linalg.norm(b)

    if err_refined > err_before:
        # 精化未改善时使用精化前的位置与矩，保证稀疏求解结果不退化
        print(f"位置精化未改善（{err_refined*100:.3f}% > {err_before*100:.3f}%），回退到精化前位置")
        positions = positions_before
        moments = x_before.reshape(-1, 3)
        T_S = T_S_before

    # 合并距离过近的偶极子，并删除弱偶极子
    positions, moments = merge_prune_dipoles(positions, moments)
    # 合并/裁剪后用支持集位置重新做一次最小二乘，恢复最优矩
    if len(positions) > 0:
        T_final = build_free_space_T(sub_coords, positions)
        moments, _, _, _ = np.linalg.lstsq(T_final, b, rcond=None)
        moments = moments.reshape(-1, 3)
    print(f"精化并裁剪后偶极子数量: {len(positions)}")

    # 6. 输出与验证
    print("=" * 60)
    print("步骤 6/6：输出结果与验证")
    txt_path = os.path.join(OUTPUT_DIR, "equivalent_dipoles.txt")
    save_equivalent_dipoles_txt(txt_path, positions, moments)
    print(f"偶极子阵列已保存: {txt_path}")

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "frequency_hz": FREQ_HZ,
        "box_bounds": BOX_BOUNDS,
        "candidate_bounds": bounds,
        "candidate_spacing": CANDIDATE_SPACING,
        "candidate_shrink_ratio": CANDIDATE_SHRINK_RATIO,
        "obs_subsample_step": OBS_SUBSAMPLE_STEP,
        "n_obs_sub": n_obs_sub,
        "n_candidate_sites": n_sites,
        "selected_n_dipoles": len(positions),
        "final_train_error_percent": round(float(final_train_error * 100.0), 4),
        "tenv_source": tenv_source,
        "solver": "group_omp + lstsq + local refinement",
        "model": MODEL_TAG,
    }
    save_metadata(meta)

    errors = save_validation_outputs(planes, positions, moments, meta)
    print(
        f"整体验证相对 L2 误差: "
        f"{errors['overall']['rel_l2_error_percent']:.4f}%"
    )
    for name, per_plane in errors["planes"].items():
        print(
            f"  {name} 面: {per_plane['rel_l2_error_percent']:.4f}%"
        )

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
