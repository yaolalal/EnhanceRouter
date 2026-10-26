#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
# Resume the registered run with explicit stages; never invoke superseded prepare.py.
export ENHANCEROUTER_STAGE0_RUN="${ENHANCEROUTER_STAGE0_RUN:-runs/stage0-20261026-v2}"
case "${1:-status}" in
 status) squeue -u wujingyao; test ! -f "$ENHANCEROUTER_STAGE0_RUN/JOBS-20261026.json" || cat "$ENHANCEROUTER_STAGE0_RUN/JOBS-20261026.json" ;;
 *) echo 'Use the recorded Slurm dependency graph. This wrapper does not silently resubmit or overwrite jobs.' >&2; exit 2 ;;
esac
