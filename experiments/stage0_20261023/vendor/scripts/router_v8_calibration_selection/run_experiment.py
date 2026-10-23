#!/usr/bin/env python3
"""Slurm-only bounded target-calibration selection over frozen v2 trials."""
import argparse
import os
import sys
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('target',))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('numerical experiment must run under Slurm')
    from eargtc.router_v8_calibration_selection.experiment import run_target
    run_target(args.root, args.target, args.workers)


if __name__ == '__main__':
    main()
