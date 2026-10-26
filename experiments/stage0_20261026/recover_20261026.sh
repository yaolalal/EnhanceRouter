#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
run=runs/stage0-20261023-v1
exp=experiments/stage0_20261023
text=$(sbatch --parsable -p RTX2080 --gpus-per-node=2 --ntasks=1 --mem=32G --time=02:00:00 --array=0-4%2 -J s0-text-fixed --dependency=afterok:119522 -o "$run/logs/text-fixed-%A_%a.log" "$exp/job.sh" text_entry.py)
scontrol update JobId=118764 Dependency=afterok:$text
scontrol update JobId=118737 Dependency=afterok:$text:118764
scontrol update JobId=118738 Dependency=afterok:118737:118765
scancel 118791
extra=$(sbatch --parsable -p RTX2080 --gpus-per-node=2 --ntasks=1 --mem=32G --time=02:00:00 --array=0-4%2 -J s0-extra-fixed --dependency=afterok:118739 -o "$run/logs/extra-fixed-%A_%a.log" "$exp/job.sh" text_entry.py --extra)
scontrol update JobId=118743 Dependency=afterok:$extra
python3 - "$text" "$extra" <<'PY'
import pathlib,json,sys,datetime
p=pathlib.Path('runs/stage0-20261023-v1/RECOVERY-20261026.json')
with p.open('x') as f:json.dump(dict(original_failed_text_array=118790,error='All five workers requested2 GPUs through --gres but were allocated1; frontend correctly refused fp16 inference on one11GB card.',allocation_probe=119521,allocation_probe_pass=True,full_forward_smoke=119522,text_array=sys.argv[1],extra_text_array=sys.argv[2],resource_fix='--gpus-per-node=2 --ntasks=1; preserve model, precision, seeds, data and all completed results',created_at=datetime.datetime.now().astimezone().isoformat()),f,indent=2)
print(p.read_text())
PY
