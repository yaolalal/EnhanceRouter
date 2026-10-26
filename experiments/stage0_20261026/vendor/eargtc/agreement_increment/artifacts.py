"""File identities and all-target freeze boundary; contains no label loader."""
import hashlib
import json
import os
from pathlib import Path

DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
REPO = Path(__file__).resolve().parents[2]


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical agreement stages require Slurm')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def verify_record(record):
    for name, expected in record['files'].items():
        if sha256(name) != expected:
            raise ValueError(f'file identity changed: {name}')


def code_files():
    paths = list((REPO/'eargtc/agreement_increment').glob('*.py'))
    paths += list((REPO/'scripts/agreement_increment').glob('*.py'))
    paths += list((REPO/'scripts/agreement_increment').glob('*.sbatch'))
    paths += list((REPO/'tests').glob('test_agreement_increment*.py'))
    return sorted(paths)


def freeze(root):
    require_slurm()
    root = Path(root).resolve()
    if (root/'FROZEN.json').exists() or (root/'evaluation/STARTED.json').exists():
        raise FileExistsError('experiment already frozen or evaluated')
    paths = [root/'CONFIG.json', root/'REGISTRATION.json', root/'PRESERVATION_BEFORE.json']
    for dataset in DATASETS:
        dest = root/'targets'/dataset
        record = read_json(dest/'selection.json')
        if record['dataset'] != dataset or set(record['calibration_ids']) & set(record['policy_ids']):
            raise ValueError('target identity or calibration/policy separation failed')
        paths.extend(dest/name for name in ('selection.json', 'model.joblib', 'scores.pt',
                                           'validation_scores.pt', 'trials.json', 'oof.npz',
                                           'chosen.json', 'cv_manifest.json', 'SEARCH_SEALED.json', 'POLICY_STARTED.json'))
        verify_record(read_json(dest/'SEARCH_SEALED.json'))
        paths.extend(root/'banks'/dataset/name for name in ('validation.pt', 'query.pt', 'manifest.json'))
        paths.append(root/'pipeline'/f'{dataset}.pt')
    paths += code_files()
    registration = read_json(root/'REGISTRATION.json')
    verify_record(registration)
    paths.extend(Path(name) for name in registration['files'])
    verify_preservation(root)
    write_json(root/'FROZEN.json', dict(schema='agreement_increment_freeze_v1', datasets=DATASETS,
               slurm_job_id=os.environ['SLURM_JOB_ID'], test_labels_accessed=False,
               files={str(p):sha256(p) for p in paths}))


def preservation_inventory(root):
    require_slurm()
    protected = []
    for name in ('joint_cross_repair','joint_cross_repair_search','joint_five_router','joint_selector',
                 'joint_v8','router_v8','router_v8_learned','router_v8_calibration_selection'):
        protected.extend((REPO/'eargtc'/name).glob('*.py'))
        protected.extend((REPO/'scripts'/name).glob('*.*'))
    protected.extend((REPO/'artifacts/preserved_versions').glob('*.tar*'))
    snapshot = REPO/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    raw = read_json(snapshot)
    # Inventory all path strings in the preservation manifest without interpreting numerical payloads.
    def visit(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if (REPO/k).is_file(): protected.append(REPO/k)
                visit(v)
        elif isinstance(value, list):
            for v in value: visit(v)
        elif isinstance(value, str) and len(value) < 4096:
            try:
                if (REPO/value).is_file(): protected.append(REPO/value)
            except OSError:
                pass
    visit(raw)
    protected.append(snapshot)
    protected.extend((REPO/'runs/router_v8_learned_20260907_v4/pipeline').glob('*'))
    inventory = {}
    for p in sorted(set(protected)):
        if not p.is_file(): continue
        s = p.stat()
        inventory[str(p)] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns,
                                 sha256=sha256(p) if s.st_size < 16*1024*1024 else None)
    write_json(Path(root)/'PRESERVATION_BEFORE.json', dict(files=inventory,
               scope='all snapshot-listed discovered files and historical source/scripts; large files size+mtime'))


def verify_preservation(root):
    before = read_json(Path(root)/'PRESERVATION_BEFORE.json')
    for name, expected in before['files'].items():
        p = Path(name); s = p.stat()
        if s.st_size != expected['size'] or s.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'preserved file metadata changed: {name}')
        if expected['sha256'] and sha256(p) != expected['sha256']:
            raise ValueError(f'preserved file content changed: {name}')
    return dict(pass_preservation=True, file_count=len(before['files']), scope=before['scope'])
