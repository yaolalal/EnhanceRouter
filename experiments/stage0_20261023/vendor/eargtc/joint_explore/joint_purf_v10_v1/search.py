"""Finite per-split adaptive search: never pool validation labels across splits."""
import hashlib
import itertools
import json
import os
import torch
from .protocol import RUN,DATASETS,SEEDS,read,write,save,sha,check,cell,load,training_open,verify_code
from .training import DEFAULTS,train,make_model,predict


def cid(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
def dedup(configs):return list({cid(c):c for c in configs}.values())
def candidate_path(seed,ds,config):return cell(seed,ds)/'candidates'/cid(config)
def key(seed,ds):return f'{seed}/{ds}'

def core():
    return [dict(DEFAULTS,dim=d,cross=c,lr=lr) for d,c,lr in itertools.product((16,32,64),(0.,.25,1.),(.0003,.001,.003))]

def schemes(anchor):
    out=[dict(anchor,gate_lr=g,warmup=w,gate_gold=s) for g,w,s in itertools.product((1.,.1),(0,50),(False,True))]
    out += [dict(anchor,aux=v) for v in (.1,.3)]
    out += [dict(anchor,keep=v) for v in (.1,.5)]
    return dedup(out)

def refinement(winner):
    out=[winner]
    for k,values in [('dim',(16,32,64)),('cross',(0.,.25,1.)),('lr',(.0003,.001,.003))]:
        out += [dict(winner,**{k:v}) for v in values]
    out += [dict(winner,aux=.1),dict(winner,keep=.1),dict(winner,aux=.1,keep=.1)]
    return dedup(out)

def removals(winner):
    out=[winner]
    for k,v in [('gate_lr',1.),('warmup',0),('gate_gold',False),('aux',0.),('keep',0.),('cross',0.)]:
        out.append(dict(winner,**{k:v}))
    # Check both auxiliary regularizers as a pair around the selected structural/optimizer setting.
    out += [dict(winner,aux=a,keep=k) for a,k in itertools.product((0.,.1),(0.,.1))]
    return dedup(out)

def result(seed,ds,cfg):
    path=candidate_path(seed,ds,cfg);r=read(path/'result.json')
    if r['config']!=cfg or r['training_seed']!=seed or r['inner_ensemble'] is not False:raise ValueError('Candidate identity changed')
    check({str(path/'checkpoint.pt'):r['checkpoint_sha256']})
    member=torch.load(path/'checkpoint.pt',weights_only=True,map_location='cpu')
    if member['config']!=cfg or member['training_seed']!=seed or member['inner_ensemble'] is not False or member['checkpoints']['accuracy']!=r['selected']:
        raise ValueError('Candidate checkpoint disagrees with result metadata')
    return r

def rank(seed,ds,configs):
    rows=[]
    for cfg in dedup(configs):
        r=result(seed,ds,cfg)
        rows.append(dict(id=cid(cfg),config=cfg,accuracy=r['selected']['accuracy'],
                         nll=r['selected']['nll'],step=r['selected']['step'],training_seed=seed))
    return sorted(rows,key=lambda r:(-r['accuracy'],r['nll'],r['id']))

def checked_stage(stage):
    spec=read(RUN/f'stage{stage}.json')
    for seed,ds in itertools.product(SEEDS,DATASETS):
        done=read(RUN/f'stage{stage}_complete'/f'{seed}_{ds}.json')
        expected={cid(c):sha(candidate_path(seed,ds,c)/'result.json') for c in spec['candidates'][key(seed,ds)]}
        if done['stage']!=stage or done['seed']!=seed or done['dataset']!=ds or done['candidates']!=expected:
            raise ValueError('Stage completion evidence changed')
    return spec

def plan(stage):
    training_open();verify_code();bycell={};evidence={}
    previous=[checked_stage(i) for i in range(1,stage)]
    for seed,ds in itertools.product(SEEDS,DATASETS):
        k=key(seed,ds)
        if stage==1:configs=core()
        elif stage==2:
            ranks=rank(seed,ds,previous[0]['candidates'][k]);configs=schemes(ranks[0]['config']);evidence[k]=ranks
        elif stage==3:
            ranks=rank(seed,ds,previous[1]['candidates'][k]);configs=refinement(ranks[0]['config']);evidence[k]=ranks
        elif stage==4:
            configs0=[c for stage0 in previous for c in stage0['candidates'][k]]
            ranks=rank(seed,ds,configs0);configs=removals(ranks[0]['config']);evidence[k]=ranks
        else:raise ValueError('Unknown stage')
        bycell[k]=configs
    write(RUN/f'stage{stage}.json',dict(stage=stage,seeds=list(SEEDS),candidates=bycell,selection_evidence=evidence,
          criterion='own_split_validation_accuracy_then_nll_then_id',cross_split_selection=False,test_truth_loaded=False))
    print('PLANNED',stage,{k:len(v) for k,v in bycell.items()},flush=True)

def worker(stage,index):
    training_open();verify_code();spec=read(RUN/f'stage{stage}.json')
    seed,ds=list(itertools.product(SEEDS,DATASETS))[index]
    f,s,p,h,split,router=load(seed,ds)
    write(cell(seed,ds)/'pools.json',dict(training_seed=seed,counts={k:len(v) for k,v in p.items()},
           gold_count=len(s['gold_ids']),validation_count=len(s['selection_ids']),test_truth_loaded=False))
    for cfg in spec['candidates'][key(seed,ds)]:
        training_open();path=candidate_path(seed,ds,cfg)
        if (path/'result.json').exists():result(seed,ds,cfg);continue
        if (path/'checkpoint.pt').exists():raise ValueError('Incomplete candidate checkpoint requires recovery audit: '+str(path))
        member=train(f,s,p,h,cfg,seed);save(path/'checkpoint.pt',member)
        selected=member['checkpoints']['accuracy']
        write(path/'result.json',dict(config=cfg,training_seed=seed,split_seed=seed,inner_ensemble=False,
              selected=selected,checkpoints=member['checkpoints'],history=member['history'],
              train_seconds=member['train_seconds'],parameter_count=member['parameter_count'],
              checkpoint_sha256=sha(path/'checkpoint.pt'),test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
        print('TRIAL',stage,seed,ds,cid(cfg),selected,round(member['train_seconds'],2),flush=True)
    write(RUN/f'stage{stage}_complete'/f'{seed}_{ds}.json',dict(stage=stage,seed=seed,dataset=ds,
          candidates={cid(c):sha(candidate_path(seed,ds,c)/'result.json') for c in spec['candidates'][key(seed,ds)]},test_truth_loaded=False))

def select():
    training_open();verify_code();stages=[checked_stage(i) for i in range(1,5)]
    selection={}
    for seed,ds in itertools.product(SEEDS,DATASETS):
        k=key(seed,ds);ranks=rank(seed,ds,[c for stage in stages for c in stage['candidates'][k]])
        selection[k]=dict(winner=ranks[0],ranking=ranks)
    write(RUN/'SELECTION.json',dict(cells=selection,seeds=list(SEEDS),inner_seeds=[],test_truth_loaded=False,
          scope='best among registered candidates in each own historical split',cross_split_selection=False,
          rule='own_split_validation_accuracy_then_nll_then_id'))
    print('SELECTED',{k:v['winner']['config'] for k,v in selection.items()},flush=True)

def export(index):
    training_open();verify_code();seed,ds=list(itertools.product(SEEDS,DATASETS))[index]
    selection=read(RUN/'SELECTION.json');cfg=selection['cells'][key(seed,ds)]['winner']['config']
    dst=cell(seed,ds)/'final'
    if (dst/'complete.json').exists():check(read(dst/'complete.json')['files']);return
    f,s,p,h,split,router=load(seed,ds);r=result(seed,ds,cfg)
    member=torch.load(candidate_path(seed,ds,cfg)/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=make_model(f,h,cfg);model.load_state_dict(member['states']['accuracy'])
    prob=predict(model,f,h,f['node_ids'])
    torch.testing.assert_close(prob[s['selection_ids']],member['validation_probability'],rtol=0,atol=2e-6)
    save(dst/'predictions.pt',dict(node_ids=f['node_ids'],probability=prob,training_seed=seed,inner_ensemble=False))
    default=result(seed,ds,DEFAULTS)
    dm=torch.load(candidate_path(seed,ds,DEFAULTS)/'checkpoint.pt',weights_only=True,map_location='cpu')
    model0=make_model(f,h,DEFAULTS);model0.load_state_dict(dm['states']['accuracy'])
    save(dst/'default.pt',dict(probability=predict(model0,f,h,f['node_ids']),candidate_sha256=default['checkpoint_sha256'],training_seed=seed,inner_ensemble=False))
    save(dst/'model.pt',dict(state=model.state_dict(),config=cfg,training_seed=seed,
                           gnn_dim=f['gnn_embeddings'].shape[1],llm_dim=h.shape[1],classes=prob.shape[1]))
    a,b=model.coefficients()
    write(dst/'selection.json',dict(config=cfg,candidate_id=cid(cfg),validation=r['selected'],training_seed=seed,
          alpha=float(a.detach()),beta=float(b.detach()),weights=model.weights().detach().tolist(),
          parameter_count=r['parameter_count'],candidate_sha256=r['checkpoint_sha256'],test_truth_loaded=False))
    write(dst/'complete.json',dict(files={str(p):sha(p) for p in dst.iterdir() if p.is_file()},all_checks_pass=True))
    print('EXPORTED',seed,ds,flush=True)
