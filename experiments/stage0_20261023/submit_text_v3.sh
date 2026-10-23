#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
text=$(sbatch --parsable -p RTX2080 --gres=gpu:2 --mem=32G --time=02:00:00 --array=0-4%1 -J s0-text3 -o runs/stage0-20261023-v1/logs/text-v3-%A_%a.log experiments/stage0_20261023/job.sh text_entry.py)
scontrol update JobId=118764 Dependency=afterok:$text
scontrol update JobId=118737 Dependency=afterok:118690:$text:118764
scancel 118740 118741 118742 118751 118763
extra=$(sbatch --parsable -p RTX2080 --gres=gpu:2 --mem=32G --time=02:00:00 --array=0-4%1 -J s0-extra3 --dependency=afterok:118739 -o runs/stage0-20261023-v1/logs/extra-v3-%A_%a.log experiments/stage0_20261023/job.sh text_entry.py --extra)
scontrol update JobId=118743 Dependency=afterok:$extra
python3 - "$text" "$extra" <<'PY'
import json,sys,pathlib
p=pathlib.Path('runs/stage0-20261023-v1/SUBMISSION_AMENDMENT_02.json')
with p.open('x') as f:json.dump(dict(text_array=sys.argv[1],extra_text_array=sys.argv[2],gpu='2xRTX2080',quantization=False,reason='Keep frozen fp16 model entirely on GPUs; single11GB CPU offload is too slow. Preserve valid cached rows and use bounded checkpoint/requeue.',supersedes_text=[118758,118759,118760,118761,118762],supersedes_extra=[118740,118741,118742,118751,118763]),f,indent=2)
print(p.read_text())
PY
