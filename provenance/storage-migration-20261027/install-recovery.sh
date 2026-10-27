#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
test -L /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
python3 - <<'CHECK'
from pathlib import Path
import json
r=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter').resolve()
a=json.loads((r/'STORAGE_LAYOUT_20261027.json').read_text())
assert a['code_checksum_pass'] and a['active_run_checksum_pass'] and a['no_historical_files_deleted']
assert r==Path(a['active_project']).resolve()
assert Path(a['historical_archive']).is_dir()
CHECK
cp -a upload-20261027/experiments/stage0_20261027/. experiments/stage0_20261027/
cp -a upload-20261027/docs/FAIRROUTER_BASELINE.md docs/FAIRROUTER_BASELINE.md
chmod +x experiments/stage0_20261027/job.sh
bash -n experiments/stage0_20261027/job.sh
git add -- experiments/stage0_20261027 docs/FAIRROUTER_BASELINE.md
git commit -m 'Add B0-constrained numeric D1 and quota failure recovery'
printf '%s\n' 'RECOVERY_CODE_INSTALLED'
