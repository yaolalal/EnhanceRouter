"""Filesystem-only migration. No imports or numerical execution of source code."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EALR-GTC')
assert ROOT.name=='EnhanceRouter' and ROOT.parent==SOURCE.parent
assert not (ROOT/'provenance/source.json').exists(),'Bootstrap is one-time and immutable'


def read(p):return json.loads(Path(p).read_text())
def h(data,algorithm='sha256'):return hashlib.new(algorithm,data).hexdigest()
def write(p,v):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')


provenance={}
def copy(source,destination,text=None,transform=None):
    source=Path(source);p=ROOT/destination;p.parent.mkdir(parents=True,exist_ok=True)
    original=source.read_bytes()
    data=original if text is None else text.encode()
    # Relocate dormant historical I/O constants as well; the new runner owns all artifact access.
    replaced=data.replace(str(SOURCE).encode(),str(ROOT).encode())
    if replaced!=data:transform=(transform or '')+'; relocate absolute historical project constants'
    data=replaced
    with p.open('xb') as f:f.write(data)
    provenance[str(source)]=dict(destination=str(destination),source_md5=h(original,'md5'),source_sha256=h(original),
          copied_md5=h(data,'md5'),copied_sha256=h(data),transformation=transform or 'byte-identical copy')


regpath=SOURCE/'runs/purf_v1_ablation_20260916/REGISTRATION.json'
registration=read(regpath)
inventorypath=SOURCE/'runs/rebuttal_followup_20261022_v1/INVENTORY.json'
inventory={f"{x['shot']}/{x['seed']}/{x['dataset']}":x for x in read(inventorypath)['rows']}
standardpath=SOURCE/'runs/purf_ablation_standard_eval1000_20260923/STANDARD_EVAL_IDS.json'
standard={f"{x['seed']}/{x['dataset']}":x['node_ids'] for x in read(standardpath)}
reference=SOURCE/'runs/purf_ablation_standard_eval1000_20260923/per_seed.csv'


def bank(shot,seed,ds,branch,role):
    if shot!=5:return SOURCE/f'runs/router_v10_shots_v1_20260914/shot{shot}/seed{seed}/banks/{ds}/{branch}_{role}.pt'
    if role=='train':return SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/banks/{ds}/{branch}_train.pt'
    if seed==42:return SOURCE/f'runs/router_v10_final/inputs/{role}/{ds}/{branch}.pt'
    sub='r8_agreement' if branch=='agreement' else 'r8_disagreement'
    return SOURCE/f'runs/multiseed_reproduction_20260909_v1/seed{seed}/router_work/{sub}/banks/{ds}/{role}.pt'


cells={}
for key,r in sorted(registration['records'].items()):
    shot,seed,ds=key.split('/');shot=int(shot);seed=int(seed)
    if seed not in (42,43,44):continue
    row=inventory[key];split=read(r['paths']['split']);branches={};expected=dict(r['hashes'])
    for branch in ('agreement','disagreement'):
        model=row['branches'][branch]
        branches[branch]=dict(bundle=model['bundle'],**{role:str(bank(shot,seed,ds,branch,role)) for role in ('train','validation','query')})
        expected[model['bundle']]=model['bundle_sha256']
    configpath=f'configs/cells/{shot}/{seed}/{ds}.json';write(configpath,r['config'])
    cells[key]=dict(dataset=ds,shot=shot,seed=seed,config=r['config'],inputs=r['paths'],checkpoint=r['checkpoint'],
                    predictions=r['predictions'],selection=r['selection'],graph=split['graph_path'],
                    branches=branches,expected_sha256=expected)
    assert len(standard[f'{seed}/{ds}'])==1000
    assert set(standard[f'{seed}/{ds}'])<=set(split['unlabeled_ids'])
    assert all(Path(p).is_file() for p in [r['checkpoint'],r['predictions'],r['selection'],split['graph_path'],
                    *r['paths'].values(),*[p for b in branches.values() for p in b.values()]])
assert len(cells)==45
write('configs/inputs.json',dict(schema=1,source_root=str(SOURCE),cells=cells,standard_eval_ids=standard,
      reference_metrics=str(reference),metadata_sources=list(map(str,[regpath,inventorypath,standardpath,reference]))))
write('configs/experiment.json',dict(method='Router-v10-MS-Original + Joint/PURF V1',seeds=[42,43,44],shots=[3,5,10],
      datasets=['cora','citeseer','pubmed','arxiv','ogbn-products'],evaluation_protocol='standard_10shot_eval1000',
      selection='historical own-validation winner; preserve per-cell configs',
      split_seed_equals_joint_training_seed=True,inner_joint_ensemble=False,
      router_internal_seeds='retain exact selected historical configurations',
      standard_eval_nodes=1000,standard_eval_order='saved historical 10-shot Query order',
      standard_deviation='population ddof=0',frozen_experts=True))

# Copy exact pure numerical kernels and statically discover their local imports.
seeds=['eargtc.joint_explore.joint_purf_v10_v1.training','eargtc.joint_explore.joint_purf_v10_v1.search',
       'eargtc.joint_cross_repair.artifacts','eargtc.joint_five_router.adapter','eargtc.router_bc_split_c.calibration',
       'eargtc.router_aug_search.v6_engine','eargtc.router_aug_search.v7_engine','eargtc.router_aug_search.v7_expanded',
       'eargtc.router_v8_v7_explore.core_v2','eargtc.router_v8_exploration.core','eargtc.agreement_increment.policy',
       'eargtc.router_v10_tune.policy','eargtc.data']


def module_path(name):
    if name!='eargtc' and not name.startswith('eargtc.'):return None
    p=SOURCE/Path(*name.split('.'))
    if p.with_suffix('.py').is_file():return p.with_suffix('.py')
    if (p/'__init__.py').is_file():return p/'__init__.py'
    return None


def dependencies(text,name,package=False):
    names=set();tree=ast.parse(text)
    pkg=name if package else name.rpartition('.')[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):names.update(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):
            base=node.module or ''
            if node.level:
                try:base=importlib.util.resolve_name('.'*node.level+base,pkg)
                except (ValueError,ImportError):continue
            names.add(base);names.update(base+'.'+a.name for a in node.names if a.name!='*')
        elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('eargtc.'):
            names.add(node.value)
    return [n for n in names if module_path(n)]


queue=list(seeds);seen=set()
while queue:
    name=queue.pop()
    if name in seen:continue
    p=module_path(name)
    if p is None:continue
    seen.add(name)
    copy(p,Path('src')/p.relative_to(SOURCE))
    queue.extend(dependencies(p.read_text(),name,p.name=='__init__.py'))
    parts=name.split('.')
    for i in range(1,len(parts)):
        parent='.'.join(parts[:i])
        if module_path(parent):queue.append(parent)
        else:
            init=ROOT/'src'/Path(*parts[:i])/'__init__.py';init.parent.mkdir(parents=True,exist_ok=True)
            if not init.exists():init.write_text('')

for source,dest in [('scripts/router_v10_multiseed_tune/tune_engine.py','src/enhancerouter/tune_engine.py')]:
    p=SOURCE/source;copy(p,dest)
    # Every dynamically used kernel is included by the seed list/static closure above.
    missing=[m for m in dependencies(p.read_text(),'enhancerouter.tune_engine') if m not in seen]
    assert not missing,missing

p=SOURCE/'scripts/router_v10_multiseed_tune/policy.py'
copy(p,'src/enhancerouter/policy.py',p.read_text().replace('from common import *','import hashlib\nimport numpy as np'),
     'replace orchestration wildcard import with its exact numpy/hashlib dependencies')

# Keep the original adapter functions verbatim while removing unrelated historical orchestration imports.
p=SOURCE/'eargtc/multiseed_joint/experiment.py';text=p.read_text();tree=ast.parse(text)
functions=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validation_halves','prepare_arrays')]
assert len(functions)==2
header='''"""Exact original Joint V1 input adapter functions, extracted without math changes."""
import torch
from eargtc.joint_cross_repair.artifacts import validate_features, validate_supervision, validate_pools
from eargtc.joint_five_router.adapter import build_router_features_and_pools
from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split

'''
copy(p,'src/enhancerouter/adapter.py',header+'\n\n'.join(functions)+'\n',
     'verbatim extraction of validation_halves and prepare_arrays; retain only their required imports')

git=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True,capture_output=True)
source_commit=git.stdout.strip() if git.returncode==0 else None
write('provenance/copied-sources.json',dict(files=provenance,copied_module_count=len(seen),
      source_root=str(SOURCE),write_policy='EALR-GTC read-only; source code copied into EnhanceRouter'))
fingerprint=h(json.dumps({p:r['source_sha256'] for p,r in sorted(provenance.items())},sort_keys=True).encode())
write('provenance/source.json',dict(source_root=str(SOURCE),commit_id=source_commit,
      commit_status='available' if source_commit else 'unavailable_source_is_not_a_git_repository',
      git_diagnostic=git.stderr.strip(),source_tree_fingerprint_sha256=fingerprint,
      copied_sources_manifest_sha256=h((ROOT/'provenance/copied-sources.json').read_bytes()),
      migrated_at=datetime.now(timezone.utc).isoformat(),frozen_standard_eval_source=str(standardpath)))
print('MIGRATED',len(cells),'cells;',len(seen),'copied modules;',len(provenance),'source records')
