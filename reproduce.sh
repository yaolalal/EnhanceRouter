#!/bin/bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
shot="${ENHANCEROUTER_SHOT:-5}"
case "$shot" in 3|5|10) ;; *) echo 'shot must be 3, 5, or 10' >&2; exit 2;; esac
tag="${1:-acceptance-$(date +%Y%m%d-%H%M%S)}"
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 2; }
run="$root/runs/$tag"
mkdir -p "$root/runs"
mkdir "$run"
mkdir "$run/logs"
export ENHANCEROUTER_ROOT="$root"
prepare=$(sbatch --parsable --chdir="$root" --job-name=enhance-prepare --output="$run/logs/prepare-%j.log" scripts/job.sh prepare "$run" "$shot")
array=$(sbatch --parsable --chdir="$root" --dependency="afterok:$prepare" --array="0-14%${ENHANCEROUTER_CONCURRENCY:-3}" --job-name=enhance-5shot --output="$run/logs/cell-%A_%a.log" scripts/job.sh cell "$run" "$shot")
evaluate=$(sbatch --parsable --chdir="$root" --dependency="afterok:$array" --job-name=enhance-evaluate --output="$run/logs/evaluate-%j.log" scripts/job.sh evaluate "$run" "$shot")
printf 'stage\tjob_id\nprepare\t%s\narray\t%s\nevaluate\t%s\n' "$prepare" "$array" "$evaluate" > "$run/jobs.tsv"
printf 'Run: %s\nprepare=%s array=%s evaluate=%s\n' "$run" "$prepare" "$array" "$evaluate"
