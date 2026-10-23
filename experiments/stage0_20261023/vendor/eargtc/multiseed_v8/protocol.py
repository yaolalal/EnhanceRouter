"""Preregister fresh banks/grids while reusing the frozen learning algorithms."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All numerical work must execute through Slurm')


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def roots(run_root, split_seed):
    run_root = Path(run_root).resolve()
    if split_seed not in range(42, 47):
        raise ValueError('Registered split seeds are 42..46')
    if not run_root.name.startswith('multiseed_reproduction_'):
        raise ValueError('A separate multiseed_reproduction_* run namespace is required')
    seed = run_root / f'seed{split_seed}'
    return seed, seed / 'router_work' / 'r8_disagreement', seed / 'router_work' / 'r8_agreement'


def code_paths():
    paths = set()
    for folder in ('eargtc/multiseed_v8', 'eargtc/router_v8_learned',
                   'eargtc/router_v8', 'eargtc/router_v8_exploration',
                   'eargtc/router_bc_split_c', 'eargtc/agreement_increment'):
        paths.update((REPO / folder).glob('*.py'))
    paths.update((REPO / 'scripts/multiseed_reproduction').glob('v8*'))
    return sorted(paths)


def check_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'Registered file changed: {path}')


def verify_preregistration(root, stage='full'):
    root = Path(root)
    pre = read(root / 'PREREGISTERED.json')
    if set(pre['code_hashes']) != {str(p) for p in code_paths()}:
        raise ValueError('Registered code inventory changed')
    check_hashes(pre['code_hashes'])
    check_hashes({p: h for p, h in pre['input_hashes'].items()
                  if stage not in ('source', 'target') or Path(p).name == 'manifest.json'})
    from eargtc.router_v8_learned.model import neural_grid
    from eargtc.router_v8_learned.experiment import tabular_grid
    if pre['neural_grid'] != neural_grid() or pre['tabular_grid'] != tabular_grid():
        raise ValueError('Frozen algorithm candidate grids changed')
    if pre['seed'] != 42 or pre['datasets'] != list(DATASETS):
        raise ValueError('Model seed or dataset roles changed')
    return pre


def register(run_root, split_seed):
    require_slurm()
    from eargtc.router_v8_learned.model import neural_grid
    from eargtc.router_v8_learned.experiment import tabular_grid
    from eargtc.router_v8_exploration.core import FRACTIONS
    from eargtc.agreement_increment.models import candidate_configs
    from eargtc.agreement_increment.policy import FRACTIONS as AGREEMENT_FRACTIONS
    _, disagreement, agreement = roots(run_root, split_seed)
    inputs = {}
    for root in (disagreement, agreement):
        for dataset in DATASETS:
            manifest_path = root / 'banks' / dataset / 'manifest.json'
            manifest = read(manifest_path)
            inputs[str(manifest_path)] = sha256(manifest_path)
            inputs.update(manifest['input_hashes'])
            for filename, digest in manifest['bank_hashes'].items():
                inputs[str(manifest_path.parent / filename)] = digest
    check_hashes(inputs)
    original = (REPO / 'eargtc/router_v8_learned/experiment.py').read_text()
    adapted = original.replace('from .data import load_bank, take', 'from eargtc.router_v8_learned.data import load_bank, take')
    adapted = adapted.replace('from .model import neural_grid, fit_head, predict_head, feature_importance', 'from eargtc.router_v8_learned.model import neural_grid, fit_head, predict_head, feature_importance')
    adapted = adapted.replace('from scripts.router_v8_learned.protocol import verify_preregistration as verify', 'from .protocol import verify_preregistration as verify')
    if (REPO / 'eargtc/multiseed_v8/experiment.py').read_text() != adapted:
        raise ValueError('Reproduction changed frozen learning/search algorithm body')
    code = {str(p): sha256(p) for p in code_paths()}
    pre = dict(schema='multiseed_router_v8_learned_v4', seed=42, split_seed=split_seed,
        datasets=list(DATASETS), neural_grid=neural_grid(), tabular_grid=tabular_grid(),
        banks_root=str(disagreement / 'banks'), coverage=FRACTIONS,
        target_folds=5, transductive_query_consistency=True, test_selection=False,
        code_hashes=code, input_hashes=inputs, protocol_hashes={},
        source_protocol='unchanged leave-target-and-held-out reciprocal-half LODO',
        adaptation='only split identities and resulting upstream experts/banks differ',
        test_labels_accessed=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(disagreement / 'PREREGISTERED.json', pre)
    candidates = []
    for base in candidate_configs():
        for scope in ('target',) if base['family'] == 'baseline' else ('target', 'pooled'):
            candidates.append(dict(id=f"{base['id']}__{scope}", model=base, scope=scope))
    config = dict(schema='agreement_increment_exploration_v1', datasets=list(DATASETS), seed=42,
        split_seed=split_seed, candidates=candidates, fractions=AGREEMENT_FRACTIONS,
        min_policy_accepted=20, independent_of_v4=True, query_consistency_training=False,
        test_labels_accessed=False)
    write(agreement / 'CONFIG.json', config)
    write(agreement / 'REGISTRATION.json', dict(files=dict(inputs, **code,
          **{str(agreement / 'CONFIG.json'): sha256(agreement / 'CONFIG.json')}),
          split_seed=split_seed, seed=42, test_labels_accessed=False,
          slurm_job_id=os.environ['SLURM_JOB_ID']))
    return dict(neural_configs=len(pre['neural_grid']), tabular_configs=len(pre['tabular_grid']),
                agreement_configs=len(candidates), split_seed=split_seed)
