# -*- coding: utf-8 -*-
"""主入口 1：从真实近场磁场重构等效偶极子阵列。"""

import argparse

from esr_core import config, reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="等效偶极子阵列重构：数量、位置与偶极矩"
    )
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    parser.add_argument("--feko-bin", default=config.DEFAULT_FEKO_BIN)
    parser.add_argument("--mpi-procs", type=int, default=config.MPI_PROCS)
    parser.add_argument("--jobs", type=int, default=config.CONCURRENT_JOBS)
    parser.add_argument("--candidate-spacing", type=float,
                        default=config.CANDIDATE_SPACING)
    parser.add_argument("--max-support", type=int, default=config.MAX_SUPPORT)
    parser.add_argument("--new-candidates", type=int,
                        default=config.NEW_CANDIDATES)
    parser.add_argument("--error-threshold", type=float,
                        default=config.NEW_CANDIDATE_ERROR_THRESHOLD_PERCENT)
    parser.add_argument("--max-runs", type=int, default=None,
                        help="只仿真前若干列，用于调试")
    parser.add_argument("--force-tenv", action="store_true",
                        help="忽略已有 T_env_full，强制重新仿真")
    args = parser.parse_args()

    reconstruction.run_reconstruction(
        results_dir=args.results_dir,
        feko_bin=args.feko_bin,
        mpi_procs=args.mpi_procs,
        concurrent_jobs=args.jobs,
        max_support=args.max_support,
        max_runs=args.max_runs,
        force_tenv=args.force_tenv,
        candidate_spacing=args.candidate_spacing,
        n_candidates=args.new_candidates,
        error_threshold=args.error_threshold,
    )


if __name__ == "__main__":
    main()
