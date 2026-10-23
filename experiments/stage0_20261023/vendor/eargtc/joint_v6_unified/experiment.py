"""Finite validation-only search with immutable, provenance-bound artifacts."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/joint_v6_unified_20260910_v1'
DATASETS = ('cora','citeseer','pubmed','arxiv','ogbn-products')
FAMILIES = ('direct','residual','attention')
SEEDS = (42,43,44)
LABEL_ROOT = ROOT / 'runs/router_bc_improved_structural_oof_fixed_20260729_v1'


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('All numerical Joint work requires Slurm')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(value,f)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_files(records):
    for path,h in records.items():
        if sha256(path)!=h:raise ValueError('Hash mismatch: '+str(path))


def recipes(smoke=False):
    return [dict(id=f'{family}_{name}',family=family,width=width,dropout=dropout,
                 pseudo_weight=weight,lr=.003,weight_decay=.001,batch_size=512,
                 max_epochs=2 if smoke else 200,validate_every=1 if smoke else 5,patience=8)
            for family in FAMILIES
            for name,width,dropout,weight in [('p025',64,.2,.25),('p1',64,.2,1.),
                                             ('p4',64,.2,4.),('wide',128,.3,1.)]]


def probability(p):
    require_slurm()
    if not isinstance(p,torch.Tensor) or p.ndim!=2 or not len(p):
        raise ValueError('Expected nonempty [N,C] probabilities')
    if not bool(torch.isfinite(p).all()) or bool((p<0).any()):
        raise ValueError('Invalid probability values')
    if not torch.allclose(p.sum(1),torch.ones(len(p),device=p.device,dtype=p.dtype),atol=2e-5,rtol=0):
        raise ValueError('Probabilities must sum to one')
    return p


def score(p,y):
    require_slurm();probability(p)
    if y.dtype!=torch.long or y.shape!=(len(p),) or bool(((y<0)|(y>=p.shape[1])).any()):
        raise ValueError('Invalid validation targets')
    p=p.double()
    return dict(accuracy=float((p.argmax(1)==y).double().mean()),
                nll=float(-p[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()))


def choose(probabilities,targets):
    require_slurm()
    priorities={'gnn':0,'llm':1,'average':2,'direct':3,'residual':4,'attention':5}
    if not probabilities or not set(probabilities)<=set(priorities):raise ValueError('Unknown candidates')
    rows=[dict(name=k,**score(p,targets)) for k,p in probabilities.items()]
    winner=min(rows,key=lambda r:(-r['accuracy'],r['nll'],priorities[r['name']]))
    return dict(winner,trials=rows)


def shortlist(trials):
    require_slurm();out={}
    for family in FAMILIES:
        rows=[r for r in trials if r['family']==family]
        if not rows:raise ValueError('Missing architecture')
        out[family]=min(rows,key=lambda r:(-r['accuracy'],r['nll'],r['id']))['id']
    return out


def ensemble(values):
    require_slurm()
    if len(values)!=3:raise ValueError('Three seeds required')
    for p in values:probability(p)
    if any(p.shape!=values[0].shape for p in values):raise ValueError('Seed shape mismatch')
    return torch.stack(values).mean(0)


def _source_files():
    paths=[]
    for folder in ('eargtc/joint_v6_unified','scripts/joint_v6_unified'):
        paths.extend(sorted((ROOT/folder).glob('*.py')))
        paths.extend(sorted((ROOT/folder).glob('*.sbatch')))
    paths.extend([ROOT/'eargtc/router_v8_v5/pipeline.py',
                  ROOT/'eargtc/joint_cross_repair/artifacts.py',
                  ROOT/'docs/joint_v6_unified_work/SPEC.md'])
    return {str(p.resolve()):sha256(p) for p in paths}


def register(run_root=DEFAULT_RUN,smoke=False):
    require_slurm()
    from .data import load_data
    root=Path(run_root).resolve()
    if (root/'CONFIG.json').exists():raise FileExistsError('Registered search already exists')
    root.mkdir(parents=True,exist_ok=True)
    datasets=['cora'] if smoke else list(DATASETS)
    inputs={};metadata={}
    for ds in datasets:
        train,val,inf,meta=load_data(ds)
        inputs.update(meta['input_hashes']);metadata[ds]=meta
        del train,val,inf
    snapshot=ROOT/'artifacts/preserved_versions/joint_v8_fixed_v2_20260908/SNAPSHOT.json'
    protected={str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()}
    verify_files(protected)
    inputs[str(snapshot)]=sha256(snapshot)
    # Bind the eventual truth source by bytes, without parsing any labels.
    truth_hashes={str(LABEL_ROOT/d/'decisions.jsonl'):sha256(LABEL_ROOT/d/'decisions.jsonl') for d in datasets}
    config=dict(schema='joint_v6_unified_v1',smoke=bool(smoke),datasets=datasets,seeds=list(SEEDS),
                recipes=recipes(smoke),router_root=str((ROOT/'runs/router_v8_v6_final').resolve()),
                protocol='full_query_transductive_pseudo_labels_only',source_hashes=_source_files(),
                input_hashes=inputs,label_file_hashes=truth_hashes,metadata=metadata,
                protected_snapshot=str(snapshot),protected_snapshot_sha256=sha256(snapshot),
                torch_version=str(torch.__version__),slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'CONFIG.json',config)
    print(json.dumps({'registered':str(root),'datasets':datasets,'configs_per_dataset':12}),flush=True)
    return config


def verify_registration(root,inputs=False):
    config=read_json(Path(root)/'CONFIG.json')
    if config.get('schema')!='joint_v6_unified_v1' or config['recipes']!=recipes(config['smoke']):
        raise ValueError('Unexpected registered configuration')
    expected=['cora'] if config['smoke'] else list(DATASETS)
    if config['datasets']!=expected or config['seeds']!=list(SEEDS):raise ValueError('Population mismatch')
    if config['router_root']!=str((ROOT/'runs/router_v8_v6_final').resolve()):raise ValueError('Wrong Router source')
    verify_files(config['source_hashes'])
    if set(config['source_hashes']) != set(_source_files()):
        raise ValueError('Registered source file set changed')
    if inputs:verify_files(config['input_hashes'])
    return config


def audit_member(member,cfg,seed,train,val):
    """Replay every epoch's coverage, validation schedule, and checkpoint ranking."""
    require_slurm()
    try:
        if member['config']!=cfg or member['seed']!=seed:
            raise ValueError('Member identity mismatch')
        history=member['history']
        if not history or [r['epoch'] for r in history]!=list(range(len(history))):
            raise ValueError('Epoch history is not contiguous from zero')
        if not 1<=len(history)-1<=cfg['max_epochs']:
            raise ValueError('Invalid completed epoch count')
        best=None;stale=0
        for i,row in enumerate(history):
            for pool in ('gold','pseudo'):
                count=len(train[pool+'_ids']) if i else 0
                batches=math.ceil(count/cfg['batch_size'])
                if row[pool+'_seen']!=count or row[pool+'_unique']!=count or row[pool+'_batches']!=batches:
                    raise ValueError('Incomplete or duplicate full epoch coverage')
            if i:
                losses=[row[k] for k in ('train_loss','gold_loss','pseudo_loss')]
                if any(not math.isfinite(v) or v<0 for v in losses):raise ValueError('Invalid logged losses')
                expected=row['gold_loss']+cfg['pseudo_weight']*row['pseudo_loss']
                if not math.isclose(row['train_loss'],expected,rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError('Unequal or unexpected loss weighting')
            is_validation=i%cfg['validate_every']==0
            if not is_validation:
                if row['accuracy'] is not None or row['nll'] is not None:raise ValueError('Unexpected validation epoch')
                continue
            acc,nll=row['accuracy'],row['nll']
            if not math.isfinite(acc) or not 0<=acc<=1 or not math.isfinite(nll) or nll<0:
                raise ValueError('Invalid validation history')
            candidate=dict(accuracy=acc,nll=nll,epoch=i)
            if best is None or (-acc,nll,i)<(-best['accuracy'],best['nll'],best['epoch']):
                best=candidate;stale=0
            else:stale+=1
            if stale>=cfg['patience'] and i!=len(history)-1:
                raise ValueError('Training continued past registered early stop')
        if len(history)-1<cfg['max_epochs'] and stale<cfg['patience']:
            raise ValueError('Training ended before full registered stopping rule')
        if member['best']!=best:raise ValueError('Checkpoint was not the validation best')
        actual=score(member['validation_probs'],val['targets'])
        if any(not math.isclose(actual[k],best[k],abs_tol=1e-10,rel_tol=1e-10) for k in ('accuracy','nll')):
            raise ValueError('Saved checkpoint score differs from history')
        if not member['state'] or any(v.device.type!='cpu' or not bool(torch.isfinite(v).all()) for v in member['state'].values()):
            raise ValueError('Invalid CPU checkpoint state')
    except (KeyError,TypeError,AttributeError) as error:
        raise ValueError('Malformed member audit record') from error
    return dict(epochs=len(history)-1,best_epoch=best['epoch'],full_pool_coverage=True)


def _member(root,train,val,cfg,seed,device):
    from .training import train_member
    path=root/'members'/f"{cfg['id']}_s{seed}.pt"
    mark=path.with_suffix('.json')
    if mark.exists():
        record=read_json(mark);verify_files(record['files'])
        if record['config_sha256']!=sha256(root.parent/'CONFIG.json'):raise ValueError('Unbound member')
        result=torch.load(path,weights_only=True,map_location='cpu')
    else:
        if path.exists():raise FileExistsError('Incomplete member artifact; preserve and investigate')
        start=time.monotonic();result=train_member(train,val,cfg,seed,device)
        save_tensor(path,result)
        write_json(mark,dict(config_sha256=sha256(root.parent/'CONFIG.json'),files={str(path):sha256(path)},
                             seed=seed,config_id=cfg['id'],seconds=time.monotonic()-start))
    audit_member(result,cfg,seed,train,val)
    return result


def _raw(train,ids):
    g=train['logp_g'][ids].exp().float();l=train['logp_l'][ids].exp().float()
    return {'gnn':g,'llm':l,'average':(g+l)*.5}


def run_cell(run_root,dataset,device='cuda'):
    require_slurm()
    from .data import load_data
    from .training import predict
    root=Path(run_root).resolve()
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise ValueError('Search closed')
    config=verify_registration(root)
    if dataset not in config['datasets']:raise ValueError('Unregistered dataset')
    cell=root/dataset;cell.mkdir(exist_ok=True)
    if (cell/'COMPLETE.json').exists():
        verify_files(read_json(cell/'COMPLETE.json')['files']);return
    train,val,inf,meta=load_data(dataset)
    if meta!=config['metadata'][dataset]:raise ValueError('Inputs changed after registration')
    trials=[];by_id={c['id']:c for c in config['recipes']}
    for cfg in config['recipes']:
        member=_member(cell,train,val,cfg,42,device)
        s=score(member['validation_probs'],val['targets'])
        trials.append(dict(id=cfg['id'],family=cfg['family'],**s))
        print(json.dumps({'dataset':dataset,'screen':cfg['id'],**s}),flush=True)
    selected=shortlist(trials)
    final_val={};artifacts={}
    for family,cid in selected.items():
        members=[_member(cell,train,val,by_id[cid],seed,device) for seed in SEEDS]
        final_val[family]=ensemble([m['validation_probs'] for m in members])
        path=cell/(family+'.pt')
        if path.exists():raise FileExistsError('Partial finalist output exists; preserve and investigate')
        full=[predict(m['state'],by_id[cid],train,inf['node_ids'],device) for m in members]
        for p,m in zip(full,members):
            probability(p)
            if not torch.allclose(p[val['ids']],m['validation_probs'],atol=2e-5,rtol=1e-5):
                raise ValueError('Full prediction disagrees with validation checkpoint')
        save_tensor(path,{'node_ids':inf['node_ids'],'seeds':torch.stack(full)})
        artifacts[str(path)]=sha256(path)
    raw=_raw(train,val['ids']);selection=choose({**raw,**final_val},val['targets'])
    selection.update(shortlist=selected,screen_trials=trials,validation_ids_sha256=hashlib.sha256(val['ids'].numpy().tobytes()).hexdigest())
    write_json(cell/'selection.json',selection)
    save_tensor(cell/'validation.pt',{'ids':val['ids'],'candidates':{**raw,**final_val}})
    save_tensor(cell/'raw.pt',{'node_ids':inf['node_ids'],**_raw(train,inf['node_ids'])})
    for name in ('selection.json','validation.pt','raw.pt'):
        p=cell/name;artifacts[str(p)]=sha256(p)
    for p in sorted((cell/'members').glob('*')):artifacts[str(p)]=sha256(p)
    if len(list((cell/'members').glob('*.json')))!=18:raise ValueError('Expected eighteen trained members')
    verify_files(config['source_hashes']);verify_files(meta['input_hashes'])
    write_json(cell/'COMPLETE.json',dict(dataset=dataset,files=artifacts,config_sha256=sha256(root/'CONFIG.json'),
                                      slurm_job_id=os.environ['SLURM_JOB_ID'],test_truth_read=False,member_count=18))
    print(json.dumps({'dataset':dataset,'complete':True,'selected':selection['name'],'validation_accuracy':selection['accuracy']}),flush=True)


def audit_cell(root,dataset,device='cpu'):
    from .data import load_data
    from .training import predict
    config=read_json(root/'CONFIG.json');cell=root/dataset
    complete=read_json(cell/'COMPLETE.json')
    if complete['dataset']!=dataset or complete['config_sha256']!=sha256(root/'CONFIG.json') or complete['test_truth_read'] is not False:
        raise ValueError('Cell is not linked to registered search')
    verify_files(complete['files'])
    train,val,inf,meta=load_data(dataset)
    if meta!=config['metadata'][dataset]:raise ValueError('Data audit changed')
    sel=read_json(cell/'selection.json');by_id={c['id']:c for c in config['recipes']}
    trials=[];members={}
    expected_names=set()
    for cfg in config['recipes']:
        name=f"{cfg['id']}_s42";expected_names.add(name)
        member=torch.load(cell/'members'/(name+'.pt'),weights_only=True,map_location='cpu')
        audit_member(member,cfg,42,train,val)
        trials.append(dict(id=cfg['id'],family=cfg['family'],**score(member['validation_probs'],val['targets'])))
    picked=shortlist(trials)
    if picked!=sel['shortlist'] or trials!=sel['screen_trials']:raise ValueError('Shortlist does not replay')
    candidates=_raw(train,val['ids']);raw=torch.load(cell/'raw.pt',weights_only=True,map_location='cpu')
    if not torch.equal(raw['node_ids'],inf['node_ids']):raise ValueError('Raw node order mismatch')
    for name,p in _raw(train,inf['node_ids']).items():
        if not torch.equal(raw[name],p):raise ValueError('Original expert changed')
    for family,cid in picked.items():
        payload=torch.load(cell/(family+'.pt'),weights_only=True,map_location='cpu',mmap=True)
        if not torch.equal(payload['node_ids'],inf['node_ids']):raise ValueError('Finalist node order mismatch')
        if payload['seeds'].shape!=(3,len(inf['node_ids']),train['logp_g'].shape[1]):raise ValueError('Seed/node coverage mismatch')
        vv=[]
        for j,seed in enumerate(SEEDS):
            name=f'{cid}_s{seed}';expected_names.add(name)
            member=torch.load(cell/'members'/(name+'.pt'),weights_only=True,map_location='cpu')
            audit_member(member,by_id[cid],seed,train,val)
            replay=predict(member['state'],by_id[cid],train,inf['node_ids'],device)
            if not torch.allclose(replay,payload['seeds'][j],atol=2e-5,rtol=1e-5):raise ValueError('Checkpoint full prediction mismatch')
            if not torch.allclose(replay[val['ids']],member['validation_probs'],atol=2e-5,rtol=1e-5):raise ValueError('Checkpoint validation mismatch')
            vv.append(member['validation_probs']);probability(payload['seeds'][j])
        candidates[family]=ensemble(vv)
    actual={p.stem for p in (cell/'members').glob('*.pt')}
    if actual!=expected_names or len(actual)!=18:raise ValueError('Unexpected member population')
    selection=choose(candidates,val['targets'])
    if any(sel[k]!=selection[k] for k in ('name','accuracy','nll','trials')):raise ValueError('Final selection does not replay')
    validation=torch.load(cell/'validation.pt',weights_only=True,map_location='cpu')
    if not torch.equal(validation['ids'],val['ids']):raise ValueError('Validation node order mismatch')
    for name,p in candidates.items():
        if not torch.equal(validation['candidates'][name],p):raise ValueError('Saved validation prediction mismatch')
    return dict(dataset=dataset,member_count=18,full_checkpoint_replay=True,validation_selection_replayed=True,
                selected=sel['name'],validation_accuracy=sel['accuracy'],test_truth_read=False)


def freeze(run_root=DEFAULT_RUN,device='cpu'):
    require_slurm();root=Path(run_root).resolve()
    config=verify_registration(root,inputs=True)
    if config['smoke'] or config['datasets']!=list(DATASETS):raise ValueError('All five formal datasets required')
    if (root/'FROZEN.json').exists() or (root/'evaluation').exists():raise FileExistsError('Already sealed')
    # Fail on an incomplete fifth cell before any expensive checkpoint replay.
    for ds in DATASETS:
        if not (root/ds/'COMPLETE.json').is_file():raise ValueError('Missing completed cell: '+ds)
    rows=[audit_cell(root,ds,device) for ds in DATASETS]
    snapshot=Path(config['protected_snapshot'])
    if sha256(snapshot)!=config['protected_snapshot_sha256']:raise ValueError('Preserved snapshot changed')
    protected={str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()};verify_files(protected)
    verify_files(config['input_hashes']);verify_files(config['source_hashes'])
    audit=dict(all_checks_pass=True,cells=rows,protected_files=len(protected),test_truth_read=False,
               slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(root/'PRETEST_AUDIT.json',audit)
    paths=[root/'CONFIG.json',root/'PRETEST_AUDIT.json']+[root/d/'COMPLETE.json' for d in DATASETS]
    write_json(root/'FROZEN.json',dict(datasets=list(DATASETS),files={str(p):sha256(p) for p in paths},
                                     slurm_job_id=os.environ['SLURM_JOB_ID']))
    return audit


def verify_frozen(root):
    require_slurm();root=Path(root).resolve()
    locked=read_json(root/'FROZEN.json')
    if locked['datasets']!=list(DATASETS):raise ValueError('All five datasets required')
    verify_files(locked['files']);config=verify_registration(root,inputs=True)
    if config['smoke']:raise ValueError('Smoke cannot evaluate')
    audit=read_json(root/'PRETEST_AUDIT.json')
    if audit.get('all_checks_pass') is not True or len(audit['cells'])!=5:raise ValueError('Missing audit')
    snapshot=Path(config['protected_snapshot'])
    verify_files({str(snapshot):config['protected_snapshot_sha256']})
    verify_files({str(ROOT/p):h for p,h in read_json(snapshot)['files'].items()})
    for ds in DATASETS:
        complete=read_json(root/ds/'COMPLETE.json')
        if (complete['dataset']!=ds or complete['config_sha256']!=sha256(root/'CONFIG.json')
                or complete['test_truth_read'] is not False or complete['member_count']!=18):
            raise ValueError('Invalid completed cell')
        verify_files(complete['files'])
    return config
