#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
run=runs/stage0-20261026-v2
exp=experiments/stage0_20261026
# Register an explicit human choice first. Elapsed time is not confirmation.
test -f "$run/METRIC_BUDGET_REGISTRATION.json"
test ! -f "$run/REPORT_SUBMISSION.json"
resolve_dependency() {
 local scope="$1" gate="$2"
 if scontrol show job "$gate" >/dev/null 2>&1; then
  printf '%s' "--dependency=afterok:$gate"
 else
  test -f "$run/REPORT_INPUTS_READY_$scope.json" || { echo "Required $scope model gate is missing; validate prerequisites before resuming." >&2; return 1; }
 fi
}
d1dep=$(resolve_dependency d1 120213)
fulldep=$(resolve_dependency stage0 120214)
# Always run fresh hash checks, including when Slurm purges completed job IDs.
d1gate=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=4G --time=01:00:00 ${d1dep:+"$d1dep"} -J er-d1-fresh-gate -o "$run/logs/d1-fresh-gate-%j.log" "$exp/job.sh" verify_report_inputs.py d1)
fullgate=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=4G --time=01:00:00 ${fulldep:+"$fulldep"} -J er-stage0-fresh-gate -o "$run/logs/stage0-fresh-gate-%j.log" "$exp/job.sh" verify_report_inputs.py stage0)
d1report=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=02:00:00 --dependency=afterok:$d1gate -J er-d1-report -o "$run/logs/d1-report-%j.log" "$exp/job.sh" d1_report.py)
report=$(sbatch --parsable -p cpu --cpus-per-task=1 --mem=16G --time=04:00:00 --dependency=afterok:$fullgate:$d1report -J er-stage0-report -o "$run/logs/report-%j.log" "$exp/job.sh" report.py)
python3 - "$d1gate" "$fullgate" "$d1report" "$report" <<'PY'
import sys,json,datetime
from pathlib import Path
p=Path('runs/stage0-20261026-v2/REPORT_SUBMISSION.json')
with p.open('x') as f:json.dump(dict(created_at=datetime.datetime.now().astimezone().isoformat(),jobs=dict(zip(['fresh_D1_gate','fresh_Stage0_gate','D1_report','Stage0_report'],sys.argv[1:])),stage1_submitted=False),f,indent=2)
print(p.read_text())
PY
