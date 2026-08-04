# -*- coding: utf-8 -*-
"""
FEKO 批处理工具：环境感知 T_env 生成与最终偶极子阵列验证
==========================================================

本模块解决“偶极子周围不再是自由空间”的建模问题：

1. 环境感知 T_env 生成
   - 对候选网格中的每个位置放置单位偶极子：x/y 方向为磁偶极子（A6 卡片），
     z 方向为电偶极子（A5 卡片）；
   - 通过 EditFEKO/PREFEKO 生成含该源卡的 ``.pre`` 文件并运行 FEKO；
   - 从三个扫描面的 ``.hfe`` 中读取磁场，形成 T_env 的一列；
   - 这样机箱屏蔽、通风孔泄漏和周围设备的反射都被包含在传递矩阵中。

2. 最终偶极子阵列验证
   - 读取 ``equivalent_dipoles.txt`` 中的偶极子；
   - 把每个偶极子的复矩向量按分量拆成源：Mx/My 用 A6 磁偶极子，Pz 用 A5 电偶极子；
   - 批量导入 ``Model_EQ_Surce/Model_pred.pre`` 并运行 FEKO；
   - 用三个扫描面的重构磁场与真实源磁场计算相对误差。

FEKO A6 卡片字段（参考 FEKO 2025 官方帮助）：
``A6: 1 :  :  :  : 0 : 幅值 : 相位(度) : x : y : z : theta(度) : phi(度)``
FEKO A5 卡片字段（电偶极子）：
``A5: 1 :  :  :  :  : 幅值 : 相位(度) : x : y : z : theta(度) : phi(度)``

- 第 1 个字段 1 表示“添加到已有源”；
- 第 5 个字段 0 表示使用 Electric ring current 模型；
- theta 为偶极子与 z 轴的夹角，phi 为偶极子在 xy 平面投影与 x 轴的夹角。
"""

import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# 保证 Windows 控制台能正确显示中文输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_FEKO_BIN = r"D:\FEKO2025\feko\bin"

# 与主程序保持一致：x/y 为磁偶极矩，z 为电偶极矩
MODEL_TAG = "Mx_My_magnetic_Pz_electric"


# ---------------------------------------------------------------------------
# A6 磁偶极子源卡片
# ---------------------------------------------------------------------------
def make_a6_line(position, component, magnitude=1.0, phase_deg=0.0):
    """
    生成单个 A6 磁偶极子源卡片行。

    参数：
        position  : (x, y, z)，单位 m
        component : 'x' / 'y' / 'z'，偶极矩方向
        magnitude : 偶极矩幅值，单位 A*m^2
        phase_deg : 复相位，单位度
    """
    # 不同方向的偶极矩对应的 theta/phi（A6 卡片用球坐标描述方向）
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
    """
    生成单个 A5 电偶极子源卡片行。

    A5 卡片用于 FEKO 中的电赫兹偶极子（Electric Hertzian dipole），
    在本模型中只用于 z 方向分量 Pz。字段结构与 A6 类似，但少一个模型选择字段。
    """
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
    把 (位置, 复矩向量) 列表转换成 FEKO 源卡片列表。

    一个复矩向量的各个分量相位可能不同，不能用一个标量源表示，因此按分量拆分：
    - Mx、My：A6 磁偶极子源；
    - Pz：A5 电偶极子源。
    """
    lines = []
    for pos, m in zip(positions, moments):
        for c_idx, comp in enumerate(["x", "y"]):
            mag = abs(m[c_idx])
            if mag < 1e-14:
                continue
            phase = np.degrees(np.angle(m[c_idx]))
            lines.append(make_a6_line(pos, comp, magnitude=mag, phase_deg=phase))
        mag_z = abs(m[2])
        if mag_z >= 1e-14:
            phase_z = np.degrees(np.angle(m[2]))
            lines.append(make_a5_line(pos, "z", magnitude=mag_z, phase_deg=phase_z))
    return lines


def insert_sources_into_pre(template_pre_path, a6_lines, output_pre_path):
    """
    在模板 ``.pre`` 文件的 ``** Sources`` 段之后插入 A6 源卡片。

    ``Model_pred.pre`` 目前没有任何激励源，插入后即成为含偶极子阵列的
    重构模型。生成文件为纯 ASCII，避免 FEKO 解析中文注释出现问题。
    """
    # 生成单元偶极子工程时优先使用干净模板；
    # 这样即使 Model_pred.pre / .backup 里残留了上一次验证的偶极子源也不会混入。
    clean_path = template_pre_path + ".clean"
    if not os.path.exists(clean_path):
        clean_path = template_pre_path + ".backup"
    if not os.path.exists(clean_path):
        clean_path = template_pre_path
    with open(clean_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "** Sources":
            insert_idx = i
            break
    if insert_idx is None:
        raise ValueError("模板 .pre 文件中找不到 ** Sources 段")

    new_lines = lines[: insert_idx + 1]
    new_lines.append("** Added by ESR pipeline (A6 magnetic dipoles)\n")
    for a6_line in a6_lines:
        new_lines.append(a6_line + "\n")
    new_lines.append("\n")
    new_lines.extend(lines[insert_idx + 1:])

    with open(output_pre_path, "w", encoding="ascii", newline="\n") as f:
        f.writelines(new_lines)


# ---------------------------------------------------------------------------
# FEKO 求解与 .hfe 读取
# ---------------------------------------------------------------------------
def run_feko_case(pre_path, feko_bin=DEFAULT_FEKO_BIN, template_dir=None):
    """
    运行单个 FEKO 工程：先 prefeeko 生成 .fek，再用 feko 求解。

    命令流程：
    1. prefeko.exe <name>.pre
    2. feko.exe    <name>.fek

    返回 True 表示求解成功（.out 存在且返回码为 0）。
    """
    workdir = os.path.dirname(os.path.abspath(pre_path))
    base = os.path.splitext(os.path.basename(pre_path))[0]

    # 模板 .pre 通常通过 IN 卡片引用同目录的 .cfm 网格文件，
    # 运行前把网格文件复制到工作目录，否则 PREFEKO 找不到几何。
    if template_dir is not None:
        for cfm in glob.glob(os.path.join(template_dir, "*.cfm")):
            shutil.copy2(cfm, workdir)

    prefeko = os.path.join(feko_bin, "prefeko.exe")
    feko = os.path.join(feko_bin, "feko.exe")

    r1 = subprocess.run(
        [prefeko, f"{base}.pre"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if r1.returncode != 0:
        print(f"PREFEKO 失败: {pre_path}\n{r1.stderr[-2000:]}")
        return False

    r2 = subprocess.run(
        [feko, f"{base}.fek"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    out_path = os.path.join(workdir, f"{base}.out")
    ok = r2.returncode == 0 and os.path.exists(out_path)
    if not ok:
        print(f"FEKO 求解失败: {pre_path}\n{r2.stderr[-2000:]}")
    return ok


def run_feko_case_parallel(pre_path, feko_bin=DEFAULT_FEKO_BIN,
                           mpi_procs=6, log_path=None):
    """
    使用 FEKO 自带的 runfeko 并行求解器运行单个工程。

    runfeko 会按需要自动调用 PREFEKO，再以 Intel MPI 多进程方式求解，
    相比手写 prefeko + feko 串行命令，可将单次仿真时间从约 33 分钟
    缩短到约 6-10 分钟。进程数由 machines.feko 和 -np 参数共同决定。
    """
    workdir = os.path.dirname(os.path.abspath(pre_path))
    base = os.path.splitext(os.path.basename(pre_path))[0]
    runfeko = os.path.join(feko_bin, "runfeko.exe")

    cmd = [runfeko, base, "-np", str(mpi_procs)]
    log_handle = None
    if log_path:
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        r = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    finally:
        if log_handle is not None:
            log_handle.close()

    out_path = os.path.join(workdir, f"{base}.out")
    ok = r.returncode == 0 and os.path.exists(out_path)
    if ok:
        # .out 成功结束的标志是最后一行包含 “Finished”；
        # 如果 FEKO 内部出错但进程仍返回 0，这里也能识别出来。
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            ok = "Finished" in f.read()
    if not ok:
        tail = ""
        if log_path and os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8",
                      errors="replace") as f:
                tail = f.read()[-2000:]
        print(f"FEKO 并行求解失败: {pre_path}\n{tail}")
    return ok


def read_hfe_subsampled(filepath, step=5):
    """
    读取 .hfe 并均匀抽稀，返回 (coords, H, keep_indices)。
    与主程序的抽稀策略保持一致（这里只做均匀抽稀，峰值保留由上层处理）。
    """
    coords = []
    H = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                H.append([
                    complex(float(parts[3]), float(parts[4])),
                    complex(float(parts[5]), float(parts[6])),
                    complex(float(parts[7]), float(parts[8])),
                ])
            except ValueError:
                continue
    coords = np.array(coords)
    H = np.array(H, dtype=complex)
    keep = np.arange(0, len(coords), step)
    return coords[keep], H[keep], keep


# ---------------------------------------------------------------------------
# 环境感知 T_env 生成
# ---------------------------------------------------------------------------
def _unit_case_base(site_idx, comp):
    """单元偶极子工程名：tenv_s0000_x 等，避免多个并行任务冲突。"""
    return f"tenv_s{site_idx:04d}_{comp}"


def _cleanup_case_files(work_root, base):
    """
    清理某个单元偶极子工程产生的临时文件。

    只删除以该工程名开头的文件（.pre/.fek/.out/.str/.bof/.hfe/.log 等），
    不会触碰 Model_pred.pre / Model_pred.cfm 等共享文件。
    """
    for pattern in (base + ".*", base + "_NearField_*.hfe"):
        for f in glob.glob(os.path.join(work_root, pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def _read_unit_column(base, work_root, plane_files, keeps):
    """读取一个单元偶极子工程生成的三个 .hfe，拼接成 T_env 的一列。"""
    columns = []
    for p_idx, pf in enumerate(plane_files):
        real_base = os.path.splitext(os.path.basename(pf))[0]
        request_name = real_base.split("NearField_")[-1]
        hfe_path = os.path.join(
            work_root, f"{base}_NearField_{request_name}.hfe"
        )
        if not os.path.exists(hfe_path):
            fallback = os.path.join(work_root, f"{base}.hfe")
            if os.path.exists(fallback):
                hfe_path = fallback
            else:
                raise FileNotFoundError(
                    f"找不到 FEKO 磁场输出文件: {hfe_path}"
                )
        if os.path.getsize(hfe_path) == 0:
            raise RuntimeError(f"FEKO 磁场输出文件为空: {hfe_path}")
        _, H_full, _ = read_hfe_subsampled(hfe_path, step=1)
        H = H_full[keeps[p_idx]]
        columns.append(H.ravel())
    return np.concatenate(columns)


def generate_T_env_with_feko(src_positions, plane_files, template_pre_path,
                             output_dir, feko_bin=DEFAULT_FEKO_BIN,
                             subsample_step=5, subsample_indices=None,
                             max_runs=None, mpi_procs=6,
                             concurrent_jobs=2):
    """
    通过 FEKO 单元偶极子仿真生成环境感知传递矩阵 T_env。

    工作方式：
    - 在用户已有的 ``Model_EQ_Surce`` 目录中，从干净模板生成
      ``tenv_s0000_x.pre`` 这类单源工程，不新建自由空间模型；
    - 使用 ``runfeko -np <进程数>`` 并行求解，并把多个单源工程并发排队，
      以利用机器多核（建议 2 个工程并发、每个 6 个 MPI 进程）；
    - 每个工程只包含一个单位偶极子：x/y 分量用 A6 磁偶极子，
      z 分量用 A5 电偶极子；
    - 求解完成后读取三个 .hfe 并抽稀，形成 T_env 的一列；
    - 每列单独保存为 npy，支持断点续跑；全部完成后汇总为 T_env.npz。

    参数：
        mpi_procs      : 单个 FEKO 工程使用的 MPI 进程数（默认 6）
        concurrent_jobs: 同时运行的 FEKO 工程数（默认 2）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 工作目录直接使用 Model_EQ_Surce：模板里的 IN 卡片引用 Model_pred.cfm，
    # runfeko 在该目录运行即可找到几何网格。
    work_root = os.path.dirname(os.path.abspath(template_pre_path))
    columns_dir = os.path.join(output_dir, "feko_tenv_columns")
    os.makedirs(columns_dir, exist_ok=True)

    # 先读取真实扫描文件，确定观测点抽稀后的总行数和坐标
    n_obs = 0
    obs_coords_list = []
    keeps = []
    plane_offsets = []
    offset = 0
    for pf_idx, pf in enumerate(plane_files):
        c_full, _, _ = read_hfe_subsampled(pf, step=1)
        if subsample_indices is not None:
            keep = np.asarray(subsample_indices[pf_idx])
        else:
            keep = np.arange(0, len(c_full), subsample_step)
        keeps.append(keep)
        obs_coords_list.append(c_full[keep])
        n_obs += len(keep)
        plane_offsets.append(offset)
        offset += len(keep)
    obs_coords = np.vstack(obs_coords_list) if obs_coords_list else None

    # 生成任务表：每个位置 x/y/z 各一列
    comp_names = ["x", "y", "z"]
    jobs = []
    for s_idx, pos in enumerate(src_positions):
        for comp in comp_names:
            if max_runs is not None and len(jobs) >= max_runs:
                break
            jobs.append((s_idx, comp))
        if max_runs is not None and len(jobs) >= max_runs:
            break
    n_cols = len(jobs)
    print(f"T_env 任务总数: {n_cols}（位置 {len(src_positions)} 个，每点 3 分量）")

    def run_one(site_idx, comp):
        """运行单个单元偶极子仿真并保存该列（可断点续跑）。"""
        base = _unit_case_base(site_idx, comp)
        col_path = os.path.join(columns_dir, base + ".npy")
        if os.path.exists(col_path):
            return base, "resume"

        # 清理同名旧文件，避免上次失败残留影响本次结果
        _cleanup_case_files(work_root, base)
        pre_path = os.path.join(work_root, base + ".pre")
        src_line = (
            make_a6_line(src_positions[site_idx], comp, 1.0, 0.0)
            if comp in ("x", "y")
            else make_a5_line(src_positions[site_idx], comp, 1.0, 0.0)
        )
        insert_sources_into_pre(template_pre_path, [src_line], pre_path)

        log_path = pre_path + ".log"
        ok = run_feko_case_parallel(
            pre_path, feko_bin, mpi_procs=mpi_procs, log_path=log_path
        )
        if not ok:
            # 偶发许可证/内存抖动时清理后重试一次
            _cleanup_case_files(work_root, base)
            insert_sources_into_pre(template_pre_path, [src_line], pre_path)
            ok = run_feko_case_parallel(
                pre_path, feko_bin, mpi_procs=mpi_procs, log_path=log_path
            )
        if not ok:
            raise RuntimeError(f"单元偶极子仿真失败: {base}")

        col = _read_unit_column(base, work_root, plane_files, keeps)
        np.save(col_path, col)
        # 列已读取并保存，删除临时工程文件，避免目录被几百个模型占满
        _cleanup_case_files(work_root, base)
        return base, "done"

    # 并发提交任务；每个任务在独立工程名下运行，不会互相覆盖文件
    t_start = time.time()
    completed = 0
    progress_path = os.path.join(output_dir, "feko_tenv_progress.json")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(concurrent_jobs))
    ) as executor:
        futures = {
            executor.submit(run_one, s_idx, comp): (s_idx, comp)
            for s_idx, comp in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            base, status = future.result()
            completed += 1
            elapsed = time.time() - t_start
            eta_h = (
                elapsed / completed * (n_cols - completed) / 3600.0
                if completed > 0
                else 0.0
            )
            progress = {
                "completed": completed,
                "total": n_cols,
                "elapsed_seconds": round(elapsed, 1),
                "eta_hours": round(eta_h, 2),
                "updated_at": datetime.datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            print(
                f"[T_env] {completed}/{n_cols}: {base} ({status}) "
                f"耗时 {elapsed:.0f}s, 预计剩余 {eta_h:.1f}h"
            )

    # 按任务顺序汇总 T_env
    T = np.zeros((n_obs * 3, n_cols), dtype=complex)
    for col_idx, (s_idx, comp) in enumerate(jobs):
        base = _unit_case_base(s_idx, comp)
        col = np.load(os.path.join(columns_dir, base + ".npy"))
        T[:, col_idx] = col

    # 保存 T_env，与主程序输出格式一致；
    # 冒烟测试（max_runs 受限）不覆盖正式 T_env.npz，另存为 partial 文件。
    out_name = (
        "T_env.npz" if max_runs is None else f"T_env_partial_{max_runs}.npz"
    )
    out_path = os.path.join(output_dir, out_name)
    n_src = len(src_positions)
    np.savez_compressed(
        out_path,
        T=T,
        src_coords=src_positions,
        obs_coords=obs_coords,
        model=MODEL_TAG,
        moment_component_indices=np.tile([0, 1, 2], n_src)[:n_cols],
        component_types=np.tile(["M", "M", "E"], n_src)[:n_cols],
        frequency_hz=150e6,
        source="feko_unit_dipole",
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        plane_offsets=np.array(plane_offsets, dtype=int),
        subsample_step=subsample_step,
        mpi_procs=mpi_procs,
        concurrent_jobs=concurrent_jobs,
    )
    print(f"FEKO 环境感知 T_env 已保存: {out_path}, shape={T.shape}")
    return T, out_path


# ---------------------------------------------------------------------------
# 最终偶极子阵列的 FEKO 验证
# ---------------------------------------------------------------------------
def validate_final_dipoles_with_feko(dipole_txt_path, template_pre_path,
                                     output_dir, real_plane_files,
                                     feko_bin=DEFAULT_FEKO_BIN):
    """
    把最终偶极子阵列导入 Model_pred 并运行 FEKO 验证。

    步骤：
    1. 读取 equivalent_dipoles.txt；
    2. 把每个偶极子拆成源：Mx/My 用 A6 磁偶极子，Pz 用 A5 电偶极子；
    3. 生成验证用 .pre，运行 FEKO；
    4. 读取三个 .hfe，与真实源扫描数据逐点比较，输出误差报告。
    """
    os.makedirs(output_dir, exist_ok=True)

    positions = []
    moments = []
    with open(dipole_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 跳过首行表头
    for line in lines[1:]:
        vals = line.split()
        if len(vals) < 9:
            continue
        pos = [float(vals[0]), float(vals[1]), float(vals[2])]
        m = [
            complex(float(vals[3]), float(vals[4])),
            complex(float(vals[5]), float(vals[6])),
            complex(float(vals[7]), float(vals[8])),
        ]
        positions.append(pos)
        moments.append(m)
    positions = np.array(positions)
    moments = np.array(moments, dtype=complex)

    source_lines = moments_to_source_lines(positions, moments)
    val_pre = os.path.join(output_dir, "Model_pred_ESR_validation.pre")
    insert_sources_into_pre(template_pre_path, source_lines, val_pre)

    print(f"生成验证工程: {val_pre}，共 {len(source_lines)} 个偶极子源")
    template_dir = os.path.dirname(os.path.abspath(template_pre_path))
    if not run_feko_case(val_pre, feko_bin, template_dir):
        raise RuntimeError("FEKO 最终验证仿真失败")

    # 比较三个面的磁场
    report = {}
    b_all = []
    b_rec_all = []
    for pf in real_plane_files:
        real_base = os.path.splitext(os.path.basename(pf))[0]
        request_name = real_base.split("NearField_")[-1]
        case_base = os.path.splitext(os.path.basename(val_pre))[0]
        hfe_rec_path = os.path.join(
            output_dir, f"{case_base}_NearField_{request_name}.hfe"
        )
        if not os.path.exists(hfe_rec_path):
            fallback = os.path.join(output_dir, f"{case_base}.hfe")
            if os.path.exists(fallback):
                hfe_rec_path = fallback
            else:
                raise FileNotFoundError(
                    f"找不到 FEKO 验证磁场输出文件: {hfe_rec_path}"
                )
        coords_rec, H_rec, _ = read_hfe_subsampled(hfe_rec_path, step=1)
        coords_orig, H_orig, _ = read_hfe_subsampled(pf, step=1)
        rel_l2 = np.linalg.norm(H_orig - H_rec) / np.linalg.norm(H_orig)
        report[base] = round(float(rel_l2 * 100.0), 4)
        b_all.append(H_orig.ravel())
        b_rec_all.append(H_rec.ravel())

    b_all = np.concatenate(b_all)
    b_rec_all = np.concatenate(b_rec_all)
    report["overall"] = round(
        float(np.linalg.norm(b_all - b_rec_all) / np.linalg.norm(b_all) * 100.0),
        4,
    )

    import json

    with open(os.path.join(output_dir, "feko_validation_errors.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("FEKO 验证误差:", report)
    return report


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEKO 磁偶极子批处理工具"
    )
    parser.add_argument("--mode", choices=["tenv", "final"],
                        required=True,
                        help="tenv=生成环境感知 T_env；final=验证最终偶极子阵列")
    parser.add_argument("--candidates", help="候选偶极子位置 npy 文件")
    parser.add_argument("--dipoles", help="最终偶极子 TXT 文件")
    parser.add_argument("--template", required=True,
                        help="Model_pred.pre 模板路径")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录")
    parser.add_argument("--feko-bin", default=DEFAULT_FEKO_BIN,
                        help="FEKO bin 目录")
    parser.add_argument("--plane-files", default="",
                        help="逗号分隔的三个真实扫描 .hfe 文件路径")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="最大仿真次数（调试用）")
    args = parser.parse_args()

    if args.mode == "tenv":
        if not args.candidates:
            raise SystemExit("tenv 模式需要 --candidates 参数")
        plane_files = [p.strip() for p in args.plane_files.split(",") if p.strip()]
        src = np.load(args.candidates)
        generate_T_env_with_feko(
            src_positions=src,
            plane_files=plane_files,
            template_pre_path=args.template,
            output_dir=args.output_dir,
            feko_bin=args.feko_bin,
            max_runs=args.max_runs,
        )
    else:
        if not args.dipoles:
            raise SystemExit("final 模式需要 --dipoles 参数")
        plane_files = [p.strip() for p in args.plane_files.split(",") if p.strip()]
        validate_final_dipoles_with_feko(
            dipole_txt_path=args.dipoles,
            template_pre_path=args.template,
            output_dir=args.output_dir,
            real_plane_files=plane_files,
            feko_bin=args.feko_bin,
        )
