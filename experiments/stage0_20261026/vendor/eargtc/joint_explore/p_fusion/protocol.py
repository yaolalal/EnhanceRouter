"""Immutable identities and Slurm gates for the isolated experiment."""
import hashlib
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/router_v9new_p_fusion_20260913_v1'
SOURCE = ROOT / 'runs/router_v9_new_joint_multiseed_20260912_v2'
DEPLOYMENT = (ROOT / 'runs/router_v9_new_final').resolve()
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')


def slurm():
    if not os.environ.get('SLURM_JOB_ID') or not __debug__:
        raise RuntimeError('Numerical execution requires Slurm and enabled assertions')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify(records):
    for p, h in records.items():
        if sha(p) != h:
            raise ValueError('Hash mismatch: ' + str(p))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable JSON differs: ' + str(path))
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.link(temp, path)
    temp.unlink()


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        def equal(a, b):
            if torch.is_tensor(a):
                return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
            if isinstance(a, dict):
                return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return type(a) is type(b) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a == b
        if not equal(load(path), value):
            raise ValueError('Published tensor differs: ' + str(path))
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    os.link(temp, path)
    temp.unlink()


def load(path):
    import torch
    slurm()
    return torch.load(path, weights_only=True, map_location='cpu', mmap=True)


def cell(seed, ds):
    if seed not in SEEDS or ds not in DATASETS:
        raise ValueError('Invalid historical cell')
    return RUN / f'seed{seed}' / ds


def require_all_cells(cells):
    expected = set(itertools.product(SEEDS, DATASETS))
    if len(cells) != 25 or set(map(tuple, cells)) != expected:
        raise ValueError('All 25 historical cells are required before evaluation')


def training_open():
    slurm()
    if any((RUN / name).exists() for name in ('FROZEN.json', 'EVALUATION_STARTED.json', 'evaluation')):
        raise RuntimeError('Training closed after global freeze/evaluation')
    reg = read(RUN / 'REGISTRATION.json')
    verify(reg['code_hashes'])
    if reg['inner_seeds'] != [] or reg['training_seed_rule'] != 'seed_equals_split_seed':
        raise ValueError('Inner ensembles are disabled')
    return reg


def code_records():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'scripts/joint_explore/p_fusion').glob('*'))
    paths += [ROOT / 'tests/test_p_fusion.py', ROOT / 'scripts/v9new_multiseed/final_audit.py',
              ROOT / 'scripts/v9new_persistence/persist.py']
    for folder in ('multiseed_joint', 'joint_cross_repair', 'joint_cross_repair_search',
                   'joint_five_router', 'router_bc_split_c'):
        paths += list((ROOT / 'eargtc' / folder).glob('*.py'))
    return {str(p): sha(p) for p in paths if p.is_file()}
