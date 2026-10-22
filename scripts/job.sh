#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${ENHANCEROUTER_ROOT:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export TMPDIR="$PWD/.cache/tmp" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib" TORCH_HOME="$PWD/.cache/torch"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME"
stage="$1"; run="$2"; shot="$3"
args=("$stage" --run "$run" --shot "$shot")
if [[ "$stage" == cell ]]; then args+=(--index "${SLURM_ARRAY_TASK_ID:?}"); fi
exec "${ENHANCEROUTER_PYTHON:-/home/wujingyao/miniconda3/envs/lf/bin/python}" -B -u -m enhancerouter.cli "${args[@]}"
