#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
exp=experiments/stage0_20261023
logs=runs/stage0-20261023-v1/logs
banks=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 --job-name=s0-banks --dependency=afterok:118690:118685:118686:118687:118688:118689 -o "$logs/banks-%A_%a.log" "$exp/job.sh" frontend.py banks --index '$SLURM_ARRAY_TASK_ID')
base=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 --job-name=s0-baseline --dependency=afterok:$banks -o "$logs/baseline-%A_%a.log" "$exp/job.sh" compute.py baseline --index '$SLURM_ARRAY_TASK_ID')
selection=$(sbatch --parsable -p cpu --mem=24G --job-name=s0-oraclesel --dependency=afterok:$base -o "$logs/oracle-select-%j.log" "$exp/job.sh" compute.py oracle-select)
extra=()
for ds in cora citeseer ogbn-products; do
 extra+=("$(sbatch --parsable -p RTX3090,RTX4090 --gres=gpu:1 --mem=16G --time=12:00:00 --job-name=s0-extra-$ds --dependency=afterok:$selection -o "$logs/extra-$ds-%j.log" "$exp/job.sh" frontend.py text --extra --dataset "$ds")")
done
extra_dep=$(IFS=:; echo "${extra[*]}")
oracle=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 --job-name=s0-oraclebank --dependency=afterok:$extra_dep -o "$logs/oracle-banks-%A_%a.log" "$exp/job.sh" compute.py oracle-banks --index '$SLURM_ARRAY_TASK_ID')
diag=$(sbatch --parsable -p cpu --mem=48G --array=0-74%4 --job-name=s0-diag --dependency=afterok:$oracle -o "$logs/diagnostics-%A_%a.log" "$exp/job.sh" compute.py diagnostics --index '$SLURM_ARRAY_TASK_ID')
report=$(sbatch --parsable -p cpu --mem=48G --job-name=s0-report --dependency=afterok:$diag -o "$logs/report-%j.log" "$exp/job.sh" report.py)
python3 - "$banks" "$base" "$selection" "${extra[@]}" "$oracle" "$diag" "$report" <<'PY'
import json,sys,pathlib,time
p=pathlib.Path('runs/stage0-20261023-v1/SUBMISSION.json')
with p.open('x') as f: json.dump(dict(prepare=118678,text=[118685,118686,118687,118688,118689],heads=118690,banks=sys.argv[1],baseline=sys.argv[2],selection=sys.argv[3],extra_text=sys.argv[4:7],oracle_banks=sys.argv[7],diagnostics=sys.argv[8],report=sys.argv[9],submitted_at=time.time()),f,indent=2)
print(p.read_text())
PY
