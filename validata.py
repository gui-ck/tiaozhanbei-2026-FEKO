# -*- coding: utf-8 -*-
"""主入口 2：把等效偶极子阵列导入 FEKO 并验证预测磁场。"""

import argparse
import os

from esr_core import config, validation


def main():
    parser = argparse.ArgumentParser(
        description="FEKO 等效源验证：预测磁场与真实磁场对比"
    )
    parser.add_argument(
        "--dipoles",
        default=os.path.join(config.RESULTS_DIR, "equivalent_dipoles.txt"),
    )
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--feko-bin", default=config.DEFAULT_FEKO_BIN)
    parser.add_argument("--mpi-procs", type=int, default=config.MPI_PROCS)
    parser.add_argument("--skip-feko", action="store_true",
                        help="跳过 FEKO 求解，直接读取已有验证 .hfe")
    args = parser.parse_args()

    validation.run_feko_validation(
        dipole_path=args.dipoles,
        results_dir=args.results_dir,
        feko_bin=args.feko_bin,
        mpi_procs=args.mpi_procs,
        skip_feko=args.skip_feko,
    )


if __name__ == "__main__":
    main()
