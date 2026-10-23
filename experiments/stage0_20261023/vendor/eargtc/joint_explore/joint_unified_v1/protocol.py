"""Immutable experiment provenance and safe frozen Router-v9-new inputs."""
import hashlib
import json
import os
from pathlib import Path
import torch
from eargtc.multiseed_joint.experiment import prepare_arrays, input_paths
from eargtc.joint_cross_repair.artifacts import validate_hidden

REPO=Path(__file__).resolve().parents[3]
RUN=REPO/'runs/joint_unified_v1_20260913'
OLD=REPO/'runs/multiseed_reproduction_20260909_v1'
REFERENCE=(REPO/'runs/router_v9_new_joint_final').resolve()
DEPLOYMENT=(REPO/'runs/router_v9_new_final').resolve()
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SEEDS=(42,43,44,45,46)

def slurm():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('All numerical work requires Slurm')

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if read(p)!=value:raise ValueError('Immutable artifact changed: '+str(p))
        return
    temp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with temp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    # Existing files are immutable; workers own distinct destinations.
    if p.exists():raise ValueError('Concurrent immutable write: '+str(p))
    os.link(temp,p);temp.unlink()

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise ValueError('Refusing to overwrite tensor: '+str(p))
    temp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    torch.save(value,temp);os.link(temp,p);temp.unlink()

def check(records):
    for path,digest in records.items():
        if sha(path)!=digest:raise ValueError('Hash mismatch: '+str(path))

def mapped(path,m):
    p=Path(path);source=Path(m['source'])
    if p.is_relative_to(source):return REFERENCE/p.relative_to(source)
    if str(p) in m['external_temporary_copies']:return REFERENCE/m['external_temporary_copies'][str(p)]
    if p.is_relative_to('/tmp'):raise ValueError('Unmapped historical temporary source')
    return p

def cell(seed,ds):return RUN/f'seed{seed}'/ds

def training_open():
    slurm()
    if (RUN/'FROZEN.json').exists() or (RUN/'EVALUATION_STARTED.json').exists():raise RuntimeError('All fitting/selection closed')

def verify_code():
    reg=read(RUN/'REGISTRATION.json');check(reg['code_hashes'])
    check({k:v for group in ('dependencies','protected') for k,v in reg[group].items() if k.endswith('.py')})
    return reg

def verify_router(query,disagreement,router,split):
    ids=query['node_ids'];did=disagreement['node_ids'];route=disagreement['route']
    if not torch.equal(ids,ids.sort().values):raise ValueError('Expected sorted deployment IDs')
    protected=torch.tensor(split['support_ids']+split['valid_ids'])
    agree=query['gnn_pred']==query['llm_pred']
    checks=[torch.equal(router['node_ids'],torch.arange(split['num_nodes'])),
            torch.equal(ids.sort().values,torch.tensor(split['unlabeled_ids']).sort().values),
            not bool(router['accepted'][protected].any()),
            torch.equal(router['accepted'][ids],query['accepted_mask']),
            torch.equal(torch.where(router['accepted'][ids],router['selected_pred'][ids],-1),query['pseudo_label']),
            router['selected_score'].dtype==query['selected_score'].dtype==torch.float64,
            torch.equal(router['selected_score'][ids],query['selected_score']),
            torch.equal(router['agreement'][ids],agree),
            torch.equal(did.sort().values,ids[~agree].sort().values),
            torch.equal(router['choose_g'][did],route==0)]
    pos=torch.searchsorted(ids,did)
    checks.extend([torch.equal(ids[pos],did),bool(((route==0)|(route==1)).all()),
                   torch.equal(router['selected_pred'][did],torch.where(route==0,query['gnn_pred'][pos],query['llm_pred'][pos])),
                   torch.equal(router['selected_pred'][ids[agree]],query['gnn_pred'][agree]),
                   torch.equal(router['trust'][ids],torch.where(agree,torch.zeros_like(query['selected_score']),query['selected_score']).float())])
    if not all(checks):raise ValueError('Frozen deployment handoff parity failed')

def register():
    training_open()
    persistence=read(REFERENCE/'PERSISTENCE.json');seal=read(REFERENCE/'PERSISTENCE_COMPLETE.json')
    if not seal['all_checks_pass'] or sha(REFERENCE/'PERSISTENCE.json')!=seal['manifest_sha256']:raise ValueError('Historical persistence seal changed')
    copied=persistence['copied_files']
    dependencies={str(REFERENCE/'PERSISTENCE.json'):seal['manifest_sha256']}
    for rel in ('FINAL_AUDIT.json','DEPLOYMENT_PARITY.json','joint/CONFIG.json','joint/evaluation/per_split.csv'):
        dependencies[str(REFERENCE/rel)]=copied[rel]
    audit=read(REFERENCE/'FINAL_AUDIT.json')
    if not audit['all_checks_pass']:raise ValueError('Historical reference did not pass audit')
    deployed=read(DEPLOYMENT/'COMPLETE.json')
    if not deployed['all_checks_pass'] or sha(DEPLOYMENT/'FROZEN.json')!=deployed['frozen_sha256']:raise ValueError('Deployment not sealed')
    dependencies[str(DEPLOYMENT/'FROZEN.json')]=deployed['frozen_sha256']
    deployment_files=read(DEPLOYMENT/'FROZEN.json')['files']
    # Authenticate all frozen deployment artifacts, including the model/validation sources.
    dependencies.update(deployment_files)
    snapshot=REPO/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    protected={str(REPO/k):v for k,v in read(snapshot)['files'].items()}
    protected[str(snapshot)]=sha(snapshot)
    records={}
    for seed in SEEDS:
        for ds in DATASETS:
            base=REFERENCE/f'seed{seed}'
            meta_rel=f'seed{seed}/joint/r9_new/{ds}/metadata.json'
            meta_path=REFERENCE/meta_rel
            dependencies[str(meta_path)]=copied[meta_rel]
            meta=read(meta_path)
            source_hashes={str(mapped(k,persistence).resolve()):v for k,v in meta['input_hashes'].items()}
            paths=input_paths(OLD,seed,'r9_new',ds)
            paths['router']=base/f'router/r9_new/{ds}/predictions.pt'
            paths={k:p.resolve() for k,p in paths.items()}
            hashes={}
            for k,p in paths.items():
                if k=='router':h=copied[str(p.relative_to(REFERENCE))]
                else:h=source_hashes[str(p)]
                hashes[str(p)]=h
            frozen=base/f'router/r9_new/{ds}/FROZEN.json'
            dependencies[str(frozen)]=copied[str(frozen.relative_to(REFERENCE))]
            handoff=read(frozen)
            if handoff['split_seed']!=seed or handoff['dataset']!=ds or handoff['query_truth_loaded'] is not False:raise ValueError('Unsafe canonical handoff')
            if handoff['predictions_sha256']!=hashes[str(paths['router'])]:raise ValueError('Canonical handoff hash differs')
            for k,v in handoff['input_hashes'].items():dependencies[str(mapped(k,persistence))]=v
            for k,v in handoff['code_hashes'].items():dependencies[str(mapped(k,persistence))]=v
            q=DEPLOYMENT/f'seed{seed}/pipeline/{ds}.pt'
            d=DEPLOYMENT/f'seed{seed}/models/{ds}/disagreement/query.pt'
            for p in (q,d):hashes[str(p)]=deployment_files[str(p)]
            records[f'{seed}/{ds}']={'paths':{k:str(v) for k,v in paths.items()},'hashes':hashes}
            dependencies.update(hashes)
    check(dependencies);check(protected)
    for key,record in records.items():
        paths=record['paths'];split=read(paths['split']);seed=int(key.split('/')[0]);ds=key.split('/')[1]
        if split['dataset']!=ds or split['split_seed']!=seed or split['test_truth_exported'] is not False:raise ValueError('Historical split identity failed')
        router=torch.load(paths['router'],weights_only=True,map_location='cpu',mmap=True)
        q=torch.load(DEPLOYMENT/f'seed{seed}/pipeline/{ds}.pt',weights_only=True,map_location='cpu')
        d=torch.load(DEPLOYMENT/f'seed{seed}/models/{ds}/disagreement/query.pt',weights_only=True,map_location='cpu')
        verify_router(q,d,router,split)
    files=list(Path(__file__).parent.glob('*.py'))+list((REPO/'scripts/joint_explore/joint_unified_v1').glob('*'))
    files+=[REPO/'docs/superpowers/specs/2026-09-13-joint-unified-v1.md',REPO/'docs/superpowers/plans/2026-09-13-joint-unified-v1.md']
    reg=dict(schema='joint_unified_v1',code_hashes={str(p):sha(p) for p in files if p.is_file()},
             inputs=records,dependencies=dependencies,protected=protected,datasets=list(DATASETS),seeds=list(SEEDS),
             training_seed='split_seed',inner_seeds=[],inner_ensemble=False,test_truth_loaded=False,
             upstream_selection='posthoc query metrics',all_25_router_parity=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(RUN/'REGISTRATION.json',reg)
    print('REGISTERED: all 25 frozen handoffs and protected files verified',flush=True)

def load(seed,ds):
    slurm();reg=verify_code();record=reg['inputs'][f'{seed}/{ds}'];check(record['hashes'])
    paths=record['paths'];split=read(paths['split'])
    arrays={k:torch.load(paths[k],map_location='cpu',weights_only=True,mmap=True) for k in ('upstream','embeddings','supervision','router')}
    f,s,p=prepare_arrays(split,arrays['upstream'],arrays['embeddings'],arrays['supervision'],arrays['router'],ds)
    hidden=torch.load(paths['hidden'],map_location='cpu',weights_only=True,mmap=True)
    validate_hidden(hidden,split['num_nodes'])
    if hidden['metadata']['dataset']!=ds:raise ValueError('Hidden identity mismatch')
    return f,s,p,hidden['hidden'],split,arrays['router']
