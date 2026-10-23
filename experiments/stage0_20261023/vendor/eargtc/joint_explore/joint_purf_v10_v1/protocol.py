"""Immutable experiment provenance and safe frozen Router-v10 inputs."""
import hashlib
import json
import os
from pathlib import Path
import torch
from eargtc.multiseed_joint.experiment import prepare_arrays, input_paths
from eargtc.joint_cross_repair.artifacts import validate_hidden

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'runs/joint_purf_v10_v1_20260914'
VARIANT=os.environ.get('PURF_ROUTER','original')
if VARIANT not in ('original','min1'):raise ValueError('Unknown Router variant')
RUN=ROOT/VARIANT
PREVIOUS=REPO/'runs/joint_unified_v1_20260913'
ORIGINAL=REPO/'runs/router_v10_observed_best_five_seeds_20260913_v1'
MIN1=REPO/'runs/router_v10_citeseer_min_accept_ablation_20260913_v1'
OLD=REPO/'runs/multiseed_reproduction_20260909_v1'
REFERENCE=(REPO/'runs/router_v9_new_joint_final').resolve()
ALL_DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
DATASETS=ALL_DATASETS if VARIANT=='original' else ('citeseer',)
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

def cell(seed,ds):return RUN/f'seed{seed}'/ds

def training_open():
    slurm()
    if any((r/f).exists() for r in (ROOT,RUN) for f in ('FROZEN.json','EVALUATION_STARTED.json')):raise RuntimeError('All fitting/selection closed')

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

def load(seed,ds):
    slurm();reg=verify_code();record=reg['inputs'][f'{seed}/{ds}'];check(record['hashes'])
    paths=record['paths'];split=read(paths['split'])
    arrays={k:torch.load(paths[k],map_location='cpu',weights_only=True,mmap=True) for k in ('upstream','embeddings','supervision','router')}
    f,s,p=prepare_arrays(split,arrays['upstream'],arrays['embeddings'],arrays['supervision'],arrays['router'],ds)
    hidden=torch.load(paths['hidden'],map_location='cpu',weights_only=True,mmap=True)
    validate_hidden(hidden,split['num_nodes'])
    if hidden['metadata']['dataset']!=ds:raise ValueError('Hidden identity mismatch')
    return f,s,p,hidden['hidden'],split,arrays['router']

def source_root(seed,ds,variant=VARIANT):
    return MIN1/f'seed{seed}/validation_min1' if variant=='min1' and ds=='citeseer' else ORIGINAL/f'seed{seed}'

def canonical_router(query,disagreement,upstream,split,validation=None):
    """Convert frozen query decisions, and explicit validation routes, to safe full-node inputs."""
    slurm();n=split['num_nodes'];ids=query['node_ids']
    gp=upstream['gnn_logits'].argmax(1);lp=upstream['llm_logits'].argmax(1)
    if not torch.equal(gp[ids],query['gnn_pred']) or not torch.equal(lp[ids],query['llm_pred']):
        raise ValueError('Router experts differ from historical Joint experts')
    agree=gp==lp;choose=torch.ones(n,dtype=torch.bool)
    choose[disagreement['node_ids']]=disagreement['route']==0
    scores=torch.zeros(n,dtype=torch.float64);scores[ids]=query['selected_score']
    if validation is not None:
        v=validation['node_ids'];route=validation['route'];sc=validation['score']
        expected=torch.tensor(split['valid_ids']);expected=expected[~agree[expected]]
        if not torch.equal(v.sort().values,expected.sort().values) or not bool(((route==0)|(route==1)).all()):
            raise ValueError('Validation disagreement routing identity failed')
        choose[v]=route==0;scores[v]=sc
    accepted=torch.zeros(n,dtype=torch.bool);accepted[ids]=query['accepted_mask']
    r=dict(node_ids=torch.arange(n),choose_g=choose,trust=torch.where(agree,torch.zeros_like(scores),scores).float(),
        accepted=accepted,selected_pred=torch.where(choose,gp,lp),agreement=agree,selected_score=scores)
    verify_router(query,disagreement,r,split)
    return r

def register():
    training_open();torch.set_num_threads(1) # Match the frozen Router export engine exactly.
    old_complete=read(PREVIOUS/'COMPLETE.json')
    if not old_complete['all_checks_pass']:raise ValueError('Previous experiment incomplete')
    check({str(PREVIOUS/k):v for k,v in old_complete['files'].items()})
    old_audit=read(PREVIOUS/'FINAL_AUDIT.json')
    if sha(PREVIOUS/'FROZEN.json')!=old_audit['frozen_sha256']:raise ValueError('Previous freeze seal mismatch')
    old_freeze=read(PREVIOUS/'FROZEN.json');check({str(PREVIOUS/'REGISTRATION.json'):old_freeze['files'][str(PREVIOUS/'REGISTRATION.json')]})
    old=read(PREVIOUS/'REGISTRATION.json');dependencies=dict(old['dependencies']);protected=dict(old['protected'])
    # Protect the entire completed previous experiment through its immutable manifests.
    protected.update(old['code_hashes']);protected.update(old_freeze['files'])
    protected.update({str(PREVIOUS/k):v for k,v in old_complete['files'].items()})
    protected[str(PREVIOUS/'COMPLETE.json')]=sha(PREVIOUS/'COMPLETE.json')
    cert=read(ORIGINAL/'COMPLETE.json')
    if not cert['all_checks_pass'] or sha(ORIGINAL/'FROZEN.json')!=cert['frozen_sha256']:raise ValueError('Original seal mismatch')
    deps=read(ORIGINAL/'FROZEN.json')['files'];dependencies.update(deps)
    source_manifest=read(ORIGINAL/'SOURCE_MANIFEST.json');dependencies.update(source_manifest['inputs'])
    mc=read(MIN1/'COMPLETE.json')
    if not mc['all_checks_pass'] or not mc['frozen_heads_calibration_routes_unchanged']:raise ValueError('Min1 incomplete')
    check(mc['files']);dependencies.update(mc['files']);dependencies.update(read(MIN1/'OUTPUTS_FROZEN.json')['files'])
    dependencies.update(read(MIN1/'REGISTRATION.json')['inputs'])
    for base in (ORIGINAL,MIN1):dependencies[str(base/'COMPLETE.json')]=sha(base/'COMPLETE.json')
    # Bind the Router inference engines and their actual imported source files.
    import sys
    runtime_dir=REPO/'scripts/router_v10_multiseed_tune'
    for p in runtime_dir.glob('*.py'):dependencies[str(p)]=sha(p)
    for p in (REPO/'runs/router_v10_final/code').rglob('*.py'):dependencies.setdefault(str(p),sha(p))
    tune_reg=read(REPO/'runs/router_v10_multiseed_tune_20260913_v1/REGISTRATION.json')
    dependencies.update(tune_reg['inputs']);dependencies.update(tune_reg['code'])
    check(dependencies);check(protected)
    sys.path.insert(0,str(runtime_dir))
    from common import predict_bundle,QUERY_KEYS
    import joblib
    choices=read(ORIGINAL/'CHOICES.json')['rows'];records={}
    for seed in SEEDS:
        for ds in DATASETS:
            k=f'{seed}/{ds}';record=old['inputs'][k];paths={a:b for a,b in record['paths'].items() if a!='router'}
            hashes={p:record['hashes'][p] for p in paths.values()};check(hashes)
            split=read(paths['split'])
            if split['dataset']!=ds or split['split_seed']!=seed or split['test_truth_exported'] is not False:raise ValueError('Split identity failed')
            historical=REPO/f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/splits/{ds}_5shot_seed{seed}.json'
            if historical.read_bytes()!=Path(paths['split']).read_bytes():raise ValueError('Router and Joint split bytes differ')
            chosen=next(r for r in choices if r['seed']==seed and r['dataset']==ds and r['branch']=='disagreement')
            modelpath=Path(chosen['model_source']);check({str(modelpath):chosen['model_sha256']})
            bundle=joblib.load(modelpath)
            # Feed only features to frozen heads, excluding validation target/state fields.
            def replay(role):
                bp=(REPO/'runs/router_v10_final/inputs'/role/ds/'disagreement.pt' if seed==42 else
                    OLD/f'seed{seed}/router_work/r8_disagreement/banks/{ds}/{role}.pt')
                digest=dependencies.get(str(bp))
                if digest is None:raise ValueError('Unbound Router bank '+str(bp))
                check({str(bp):digest});hashes[str(bp)]=digest
                b=torch.load(bp,weights_only=True,map_location='cpu');allowed=set(QUERY_KEYS)
                if not allowed.issubset(b):raise ValueError('Router bank lacks features')
                if role=='query' and (not set(b)<=allowed|{'split'} or ('split' in b and b['split']!='query')):raise ValueError('Unsafe query bank schema')
                clean={a:b[a] for a in allowed}
                expected=set(split['valid_ids'] if role=='validation' else split['unlabeled_ids'])
                if not set(clean['ids'].tolist())<=expected:raise ValueError('Router bank outside historical population')
                score=predict_bundle(bundle,clean)
                return clean,score
            vb,vs=replay('validation');qb,qs=replay('query')
            validation=dict(node_ids=vb['ids'],route=torch.as_tensor(vs['route'],dtype=torch.long),score=torch.as_tensor(vs['score'],dtype=torch.float64))
            src=source_root(seed,ds);qp=src/f'pipeline/{ds}.pt';dp=src/f'branches/{ds}/disagreement.pt'
            for p in (qp,dp):hashes[str(p)]=dependencies[str(p)]
            q=torch.load(qp,weights_only=True,map_location='cpu');d=torch.load(dp,weights_only=True,map_location='cpu')
            if not torch.equal(qb['ids'],d['node_ids']) or not torch.equal(torch.as_tensor(qs['route']),d['route']):raise ValueError('Frozen query routes cannot be replayed')
            torch.testing.assert_close(torch.as_tensor(qs['score'],dtype=torch.float64),d['selected_score'],atol=0,rtol=0)
            upstream=torch.load(paths['upstream'],weights_only=True,map_location='cpu',mmap=True)
            if not torch.equal(upstream['gnn_logits'].argmax(1)[vb['ids']],vb['gnn_pred']) or not torch.equal(upstream['llm_logits'].argmax(1)[vb['ids']],vb['llm_pred']):raise ValueError('Validation experts differ')
            router=canonical_router(q,d,upstream,split,validation)
            out=cell(seed,ds)/'router';out.mkdir(parents=True,exist_ok=True)
            if not (out/'predictions.pt').exists():save(out/'predictions.pt',router)
            else:
                cached=torch.load(out/'predictions.pt',weights_only=True,map_location='cpu')
                if cached.keys()!=router.keys() or not all(torch.equal(cached[x],router[x]) for x in router):raise ValueError('Canonical cache differs')
            write(out/'FROZEN.json',dict(dataset=ds,split_seed=seed,source=str(src),query_sha256=sha(qp),disagreement_sha256=sha(dp),
                model_sha256=sha(modelpath),predictions_sha256=sha(out/'predictions.pt'),query_truth_loaded=False,validation_routes_replayed=True,query_routes_scores_exact=True))
            paths['router']=str(out/'predictions.pt');hashes[paths['router']]=sha(paths['router']);hashes[str(modelpath)]=sha(modelpath)
            records[k]=dict(paths=paths,hashes=hashes);dependencies.update(hashes)
            print('HANDOFF',VARIANT,seed,ds,int(router['accepted'].sum()),flush=True)
    for module in tuple(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and f.endswith('.py') and Path(f).is_absolute() and Path(f).is_file() and Path(f).resolve().is_relative_to(REPO):
            p=Path(f).resolve();dependencies.setdefault(str(p),sha(p))
    files=list(Path(__file__).parent.glob('*.py'))+list((REPO/'scripts/joint_explore/joint_purf_v10_v1').glob('*'))
    files.append(REPO/'docs/superpowers/plans/2026-09-14-joint-purf-v10-v1.md')
    # Current code belongs to code_hashes, not historical dependencies.
    code={str(p):sha(p) for p in files if p.is_file()}
    for p in code:dependencies.pop(p,None)
    check(dependencies);check(protected)
    write(RUN/'REGISTRATION.json',dict(schema='joint_purf_v10_v1',variant=VARIANT,code_hashes=code,inputs=records,
        dependencies=dependencies,protected=protected,seeds=list(SEEDS),datasets=list(DATASETS),training_seed='split_seed',inner_seeds=[],inner_ensemble=False,
        test_truth_loaded=False,cross_split_selection=False,upstream_selection='historical query-observed branch choices; Min1 acceptance chosen on validation',
        shared_non_citeseer='original',slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('REGISTERED',VARIANT,flush=True)
