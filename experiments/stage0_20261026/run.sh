#!/bin/bash
set -euo pipefail
cd /home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
export ENHANCEROUTER_STAGE0_RUN="${ENHANCEROUTER_STAGE0_RUN:-runs/stage0-20261026-v2}"
case "${1:-status}" in
 status)
  squeue -u wujingyao
  for name in REMAINDER_JOB_GRAPH GAMMA_JOB_GRAPH_AMENDMENT NATIVE_SEARCH_JOB_AMENDMENT; do
   test ! -f "$ENHANCEROUTER_STAGE0_RUN/$name.json" || cat "$ENHANCEROUTER_STAGE0_RUN/$name.json"
  done
  ;;
 *) echo 'Use the registered Slurm graph; this wrapper does not resubmit jobs.' >&2; exit 2 ;;
esac
