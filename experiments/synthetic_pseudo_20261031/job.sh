#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="$PWD/.cache/tmp" XDG_CACHE_HOME="$PWD/.cache" MPLCONFIGDIR="$PWD/.cache/matplotlib" TORCH_HOME="$PWD/.cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PWD/.cache/huggingface" TRITON_CACHE_DIR="$PWD/.cache/triton"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec /home/wujingyao/miniconda3/envs/lf/bin/python -B -u "$PWD/experiments/synthetic_pseudo_20261031/experiment.py" "$@"
