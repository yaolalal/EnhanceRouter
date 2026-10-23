#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
name="${1:-stage0-$(date +%Y%m%d-%H%M%S)}"
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
export ENHANCEROUTER_STAGE0_RUN="runs/$name"
mkdir "$ENHANCEROUTER_STAGE0_RUN"
mkdir "$ENHANCEROUTER_STAGE0_RUN/logs"
exp=experiments/stage0_20261023
logs="$ENHANCEROUTER_STAGE0_RUN/logs"
gpu="${ENHANCEROUTER_GPU_PARTITIONS:-RTX2080,RTX3090,RTX4090}"
text_gpu="${ENHANCEROUTER_TEXT_GPU_PARTITIONS:-RTX3090,RTX4090}"
text_gres="${ENHANCEROUTER_TEXT_GRES:-gpu:1}"
prep=$(sbatch --parsable -p cpu --mem=48G -J s0-prepare -o "$logs/prepare-%j.log" "$exp/job.sh" prepare.py)
amend=$(sbatch --parsable -p cpu --mem=16G -J s0-amend --dependency=afterok:$prep -o "$logs/amend-%j.log" "$exp/job.sh" prepare_amend.py)
small=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 -J s0-small --dependency=afterok:$amend -o "$logs/small-%A_%a.log" "$exp/job.sh" small_stratified.py --index '$SLURM_ARRAY_TASK_ID')
heads=$(sbatch --parsable -p "$gpu" --gres=gpu:1 --mem=16G --time=02:00:00 --array=0-74%2 -J s0-heads --dependency=afterok:$prep -o "$logs/heads-%A_%a.log" "$exp/job.sh" frontend.py heads --index '$SLURM_ARRAY_TASK_ID')
texts=()
for ds in cora citeseer pubmed arxiv ogbn-products; do
 texts+=("$(sbatch --parsable -p "$text_gpu" --gres="$text_gres" --mem=32G --time=02:00:00 -J s0-text-$ds --dependency=afterok:$amend -o "$logs/text-$ds-%j.log" "$exp/job.sh" frontend.py text --dataset "$ds")")
done
dep=$(IFS=:; echo "${texts[*]}")
banks=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 -J s0-banks --dependency=afterok:$heads:$small:$dep -o "$logs/banks-%A_%a.log" "$exp/job.sh" frontend.py banks --index '$SLURM_ARRAY_TASK_ID')
base=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 -J s0-baseline --dependency=afterok:$banks -o "$logs/baseline-%A_%a.log" "$exp/job.sh" compute.py baseline --index '$SLURM_ARRAY_TASK_ID')
select=$(sbatch --parsable -p cpu --mem=24G -J s0-oraclesel --dependency=afterok:$base -o "$logs/select-%j.log" "$exp/job.sh" compute.py oracle-select)
extra=()
for ds in cora citeseer pubmed arxiv ogbn-products; do
 extra+=("$(sbatch --parsable -p "$text_gpu" --gres="$text_gres" --mem=32G --time=02:00:00 -J s0-extra-$ds --dependency=afterok:$select -o "$logs/extra-$ds-%j.log" "$exp/job.sh" frontend.py text --extra --dataset "$ds")")
done
dep=$(IFS=:; echo "${extra[*]}")
oracle=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 -J s0-oracle --dependency=afterok:$dep -o "$logs/oracle-%A_%a.log" "$exp/job.sh" compute.py oracle-banks --index '$SLURM_ARRAY_TASK_ID')
diag=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 -J s0-diag --dependency=afterok:$oracle -o "$logs/diag-%A_%a.log" "$exp/job.sh" compute.py diagnostics --index '$SLURM_ARRAY_TASK_ID')
report=$(sbatch --parsable -p cpu --mem=48G -J s0-report --dependency=afterok:$diag -o "$logs/report-%j.log" "$exp/job.sh" report.py)
printf 'prepare %s\namend %s\nsmall %s\nheads %s\ntext %s\nbanks %s\nbaseline %s\nselection %s\nextra %s\noracle %s\ndiagnostics %s\nreport %s\n' "$prep" "$amend" "$small" "$heads" "${texts[*]}" "$banks" "$base" "$select" "${extra[*]}" "$oracle" "$diag" "$report" > "$ENHANCEROUTER_STAGE0_RUN/jobs.txt"
cat "$ENHANCEROUTER_STAGE0_RUN/jobs.txt"
