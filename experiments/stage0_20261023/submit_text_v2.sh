#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
ids=()
for ds in cora citeseer pubmed arxiv ogbn-products; do
 ids+=("$(sbatch --parsable -p RTX2080,RTX3090,RTX4090 --gres=gpu:1 --mem=48G --time=02:00:00 -J s0-txt2-$ds --dependency=afterok:118755 -o runs/stage0-20261023-v1/logs/text-v2-$ds-%j.log experiments/stage0_20261023/job.sh frontend.py text --dataset "$ds")")
done
dep=$(IFS=:; echo "${ids[*]}")
scontrol update JobId=118737 Dependency=afterok:118690:$dep
extra=$(sbatch --parsable -p RTX2080,RTX3090,RTX4090 --gres=gpu:1 --mem=48G --time=02:00:00 -J s0-extra-arxiv --dependency=afterok:118739 -o runs/stage0-20261023-v1/logs/extra-arxiv-%j.log experiments/stage0_20261023/job.sh frontend.py text --extra --dataset arxiv)
scontrol update JobId=118743 Dependency=afterok:118740:118741:118742:118751:$extra
smoke=$(sbatch --parsable -p cpu --mem=48G --array=0-4%2 -J s0-banks-smoke --dependency=afterok:$dep -o runs/stage0-20261023-v1/logs/smoke-banks-%A_%a.log experiments/stage0_20261023/job.sh frontend.py banks --index '$SLURM_ARRAY_TASK_ID')
base=$(sbatch --parsable -p cpu --mem=48G -J s0-base-smoke --dependency=afterok:$smoke -o runs/stage0-20261023-v1/logs/smoke-base-%j.log experiments/stage0_20261023/job.sh compute.py baseline --index 0)
python3 - "${ids[@]}" "$extra" "$smoke" "$base" <<'PY'
import sys,json,pathlib
p=pathlib.Path('runs/stage0-20261023-v1/SUBMISSION_AMENDMENT_01.json')
with p.open('x') as f:json.dump(dict(text_v2=sys.argv[1:6],extra_pubmed=118751,extra_arxiv=sys.argv[6],smoke_banks=sys.argv[7],smoke_baseline=sys.argv[8],supersedes_failed_text=[118685,118686,118687,118688,118689]),f,indent=2)
print(p.read_text())
PY
