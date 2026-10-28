#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=12:00:00
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$PWD/.cache/tmp" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib" TORCH_HOME="$PWD/.cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/.cache/huggingface" TRITON_CACHE_DIR="$PWD/.cache/triton"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec /home/wujingyao/miniconda3/envs/lf/bin/python -B -u "$PWD/experiments/stage0_20261029/factorial_joint.py" "$@"
