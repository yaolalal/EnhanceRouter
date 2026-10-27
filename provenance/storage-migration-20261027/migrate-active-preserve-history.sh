#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
set -euo pipefail
src=/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter
dst=/data1/wujingyao-20354/wujingyao/EnhanceRouter
archive=${src}.historical-archive-20261027
test -d "$src/.git"
test ! -L "$src"
test ! -e "$archive"
test -L "$src/runs/stage0-20261026-v2"
test -f "$dst/migration-difference.txt"
test -f "$dst/migration-final-check.txt"
rg -x -q MIGRATION_VERIFIED_COMPLETE "$dst/migration-120644.log"
test ! -s "$dst/migration-difference.txt"
test ! -s "$dst/migration-final-check.txt"
python3 - <<'CHECK'
from pathlib import Path
import json,hashlib
root=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter')
freeze=json.loads((root/'runs/stage0-20261026-v2/CODE_FREEZE.json').read_text())
for rel,sha in freeze['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,rel
print('REGISTERED_CODE_RECHECK_PASS',len(freeze['files']))
CHECK
rsync -a --exclude=/runs "$src/" "$dst/"
rsync -acn --exclude=/runs --out-format='%n' "$src/" "$dst/" > "$dst/active-code-migration-difference.txt"
python3 - <<'DIFF'
from pathlib import Path
p=Path('/data1/wujingyao-20354/wujingyao/EnhanceRouter/active-code-migration-difference.txt')
rows=p.read_text().splitlines()
assert all(row=='./' for row in rows), rows
print('CODE_CHECKSUM_PASS_ROOT_DIRECTORY_METADATA_ONLY', rows)
DIFF
export ER_MIGRATE_SRC="$src" ER_MIGRATE_DST="$dst" ER_MIGRATE_ARCHIVE="$archive"
python3 - <<'PY'
import os,json,pathlib,subprocess
src=pathlib.Path(os.environ['ER_MIGRATE_SRC']);dst=pathlib.Path(os.environ['ER_MIGRATE_DST']);archive=pathlib.Path(os.environ['ER_MIGRATE_ARCHIVE'])
partial=dst/'runs'/('migration-partial-'+os.environ['SLURM_JOB_ID']);partial.mkdir()
links={}
for entry in sorted((src/'runs').iterdir()):
 if entry.name=='stage0-20261026-v2':continue
 dest=dst/'runs'/entry.name
 if dest.exists() or dest.is_symlink():dest.rename(partial/entry.name)
 target=archive/'runs'/entry.name;dest.symlink_to(target,target_is_directory=entry.is_dir());links[str(dest)]=str(target)
commit=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()
record=dict(active_project=str(dst),logical_project=str(src),historical_archive=str(archive),historical_inputs_policy='original files preserved; read-only through application write boundary',partial_copies_preserved=str(partial),links=links,code_checksum_pass=True,active_run_checksum_pass=True,original_commit=commit,no_historical_files_deleted=True,slurm_job=os.environ['SLURM_JOB_ID'])
(dst/'STORAGE_LAYOUT_20261027.json').write_text(json.dumps(record,indent=2))
PY
mv "$src" "$archive"
ln -s "$dst" "$src"
test "$(readlink -f "$src")" = "$(readlink -f "$dst")"
printf '%s\n' 'ACTIVE_PROJECT_MIGRATED_HISTORY_PRESERVED'
