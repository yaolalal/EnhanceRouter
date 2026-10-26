#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
export ENHANCEROUTER_STAGE0_RUN=runs/stage0-20261026-v2
run="$ENHANCEROUTER_STAGE0_RUN"; exp=experiments/stage0_20261026; logs="$run/logs"
test -f "$run/PREPARED.json"
test ! -f "$run/REMAINDER_JOB_GRAPH.json"
# Existing validated work: historical B0=119812, D1=119818, new heads=119804,
# D3 heads=119829, missing standard text forwards=119914, head-source registry=119958.
banks=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=0-74%2 --dependency=afterok:119914:119829:119958 -J er-banks75 -o "$logs/banks-%A_%a.log" "$exp/job.sh" frontend.py banks --index '$SLURM_ARRAY_TASK_ID')
base=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=15-24,65-74%2 --dependency=afterok:$banks:119812 -J er-b0-new20 -o "$logs/b0-new-%A_%a.log" "$exp/job.sh" compute.py b0 --index '$SLURM_ARRAY_TASK_ID')
d1=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=15-24,65-74%2 --dependency=afterok:$base -J er-d1-new20 -o "$logs/d1-new-%A_%a.log" "$exp/job.sh" compute.py d1 --index '$SLURM_ARRAY_TASK_ID')
d3=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=0-74%2 --dependency=afterok:$base -J er-d3-router75 -o "$logs/d3-router-%A_%a.log" "$exp/job.sh" compute.py d3 --index '$SLURM_ARRAY_TASK_ID')
d4=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=0-74%2 --dependency=afterok:$base -J er-d4-all75 -o "$logs/d4-%A_%a.log" "$exp/job.sh" compute.py d4 --index '$SLURM_ARRAY_TASK_ID')
d1report=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=02:00:00 --dependency=afterok:$d1:119818 -J er-d1-report -o "$logs/d1-report-%j.log" "$exp/job.sh" d1_report.py)
select=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=01:00:00 --dependency=afterok:$base -J er-oracle-select -o "$logs/oracle-select-%j.log" "$exp/job.sh" compute.py oracle-select)
extra=$(sbatch --parsable -p RTX3090,RTX4090,debug --gpus-per-node=1 --ntasks=1 --mem=8G --time=02:00:00 --array=0-4%1 --dependency=afterok:$select -J er-oracle-text -o "$logs/oracle-text-%A_%a.log" "$exp/job.sh" text_entry.py --extra)
oracle=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --array=0-74%2 --dependency=afterok:$extra -J er-oracle-banks -o "$logs/oracle-banks-%A_%a.log" "$exp/job.sh" compute.py oracle-banks --index '$SLURM_ARRAY_TASK_ID')
d2=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=08:00:00 --array=0-74%2 --dependency=afterok:$oracle -J er-d2-all75 -o "$logs/d2-%A_%a.log" "$exp/job.sh" compute.py d2 --index '$SLURM_ARRAY_TASK_ID')
report=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --dependency=afterok:$d1report:$d3:$d4:$d2 -J er-stage0-report -o "$logs/report-%j.log" "$exp/job.sh" report.py)
python3 - "$banks" "$base" "$d1" "$d3" "$d4" "$d1report" "$select" "$extra" "$oracle" "$d2" "$report" <<'PY'
from pathlib import Path
import sys,json,datetime
keys=['banks75','b0_new20','d1_new20','d3_router75','d4_all75','d1_report','oracle_selection','oracle_text','oracle_banks','d2_all75','stage0_report']
p=Path('runs/stage0-20261026-v2/REMAINDER_JOB_GRAPH.json')
with p.open('x') as f:json.dump(dict(created_at=datetime.datetime.now().astimezone().isoformat(),jobs=dict(zip(keys,sys.argv[1:])),phase_limit='Stage0 only. No stage1 submission or acceptance without completed diagnostics and explicit gate record.'),f,indent=2)
print(p.read_text())
PY
