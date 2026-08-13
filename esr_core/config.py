# -*- coding: utf-8 -*-
"""项目路径、物理参数与扫描面配置。"""

import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Scanning_Field")
MODEL_EQ_DIR = os.path.join(PROJECT_ROOT, "Model_EQ_Surce")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")

TEMPLATE_PRE = os.path.join(MODEL_EQ_DIR, "Model_pred.pre")
DEFAULT_FEKO_BIN = r"D:\FEKO2025\feko\bin"

FREQ_HZ = 150e6
C0 = 299792458.0
MODEL_TAG = "Mx_My_magnetic_Pz_electric"

# 机箱外尺寸（m），用户给定毫米尺寸，这里统一换算成米。
BOX_BOUNDS = {
    "x": (-6.555, -6.280),
    "y": (-0.260, -0.041),
    "z": (-0.500, -0.115),
}

# 无法从 CAD 读取壁厚时，候选范围取机箱居中 80%。
CANDIDATE_SHRINK_RATIO = 0.8
CANDIDATE_SPACING = 0.05

# 观测点抽稀：均匀每隔 5 个点取 1 个，并保留幅值最高的前 5% 峰值点。
OBS_SUBSAMPLE_STEP = 5
PEAK_KEEP_FRACTION = 0.05

# Group OMP 支持集范围与交叉验证参数。
MIN_SUPPORT = 10
MAX_SUPPORT = 60
CV_TEST_FRACTION = 0.2
CV_SEED = 12345

# 新增候选位置参数。
NEW_CANDIDATES = 30
NEW_CANDIDATE_SPACING = 0.04
NEW_CANDIDATE_MIN_DIST = 0.04
# 初始模型误差超过该阈值（%）时才执行新增候选偶极子步骤。
NEW_CANDIDATE_ERROR_THRESHOLD_PERCENT = 15.0

# 最终偶极子后处理参数。
MERGE_DIST = 0.005
PRUNE_REL_THRESHOLD = 0.001

# FEKO 并行参数。
MPI_PROCS = 6
CONCURRENT_JOBS = 2


# 重构使用的三个扫描面，真实磁场数据以 Data_Scanning_Field 下的 .txt 为准。
RECONSTRUCTION_PLANES = [
    {
        "name": "xoy",
        "file": "Model_real_NearField_xoy.txt",
        "desc": "上方面（法向 z，z=-0.08 m）",
    },
    {
        "name": "xoz",
        "file": "Model_real_NearField_xoz.txt",
        "desc": "前方面（法向 y，y=-0.02 m）",
    },
    {
        "name": "yoz",
        "file": "Model_real_NearField_yoz.txt",
        "desc": "右方面（法向 x，x=-6.2 m）",
    },
]

# 验证使用的全部五个扫描面，包含两个新增外推平面。
VALIDATION_PLANES = RECONSTRUCTION_PLANES + [
    {
        "name": "pred_1",
        "file": "Model_real_NearField_pred_1.txt",
        "desc": "新增验证面 1（x=-6.15 m）",
    },
    {
        "name": "pred_2",
        "file": "Model_real_NearField_pred_2.txt",
        "desc": "新增验证面 2（x=-6.10 m）",
    },
]

# 验证工程需要额外写入 .pre 的两个近场请求。
EXTRA_VALIDATION_PLANES = [
    {
        "name": "pred_1",
        "x": -6.15,
        "n_y": 111,
        "n_z": 191,
        "y_start": -0.26,
        "z_start": -0.5,
        "y_step": 0.002,
        "z_step": 0.002,
    },
    {
        "name": "pred_2",
        "x": -6.10,
        "n_y": 111,
        "n_z": 191,
        "y_start": -0.26,
        "z_start": -0.5,
        "y_step": 0.002,
        "z_step": 0.002,
    },
]
