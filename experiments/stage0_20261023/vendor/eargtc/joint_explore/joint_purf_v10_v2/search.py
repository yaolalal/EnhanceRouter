"""Finite own-split search with explicit inheritance, fixed labels and freeze barrier."""
import hashlib,itertools,json,os
import torch
from .protocol import RUN,PREVIOUS,CELLS,SEEDS,DATASETS,cell,read,write,save,sha,check,load,training_open,verify_code,verify_all,diagnostic
from .training import train,expand_v1,make_model,predict,score

def cid(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
def dedup(configs):return list({cid(c):c for c in configs}.values())
def key(seed,ds):return f'{seed}/{ds}'
def candidate_path(seed,ds,c):return cell(seed,ds)/'candidates'/cid(c)
def rank_rows(rows):return sorted(rows,key=lambda r:(-r['accuracy'],r['nll'],0 if r['origin']=='inherited_v1' else 1,r['id']))

def catalog(stage,anchor,eligible):
    out=[dict(anchor)]
    if stage==1:
        out += [dict(anchor,max_steps=n,lr=lr) for n,lr in itertools.product((500,1000,2000),(anchor['lr']/3,anchor['lr']))]
    elif stage==2:
        out += [dict(anchor,max_steps=n,gate_mode='learned',gate_lr=g) for n,g in itertools.product((500,1000,2000),(1.,3.,10.))]
        out += [dict(anchor,max_steps=n,gate_mode='fixed',gate_lr=1.) for n in (500,1000,2000)]
    elif stage==3:
        out += [dict(anchor,aux_gold=g,aux_pseudo=p) for g,p in itertools.product((0.,.1,.3),repeat=2)]
        out += [dict(anchor,aux_gold=a,aux_pseudo=a) for a in (.05,.5)]
    elif stage==4:
        out += [dict(anchor,aux_gold=0.),dict(anchor,aux_pseudo=0.),dict(anchor,aux_gold=0.,aux_pseudo=0.),
                dict(anchor,keep=0.),dict(anchor,gate_mode='fixed',gate_lr=1.),dict(anchor,max_steps=500)]
        if eligible:out += [dict(anchor,class_power=p) for p in (.25,.5)]
    else:raise ValueError('Invalid stage')
    return dedup(out)

def result(seed,ds,cfg):
    dest=candidate_path(seed,ds,cfg);r=read(dest/'result.json');check({str(dest/'checkpoint.pt'):r['checkpoint_sha256']})
    member=torch.load(dest/'checkpoint.pt',weights_only=True,map_location='cpu')
    if any(r[k]!=v for k,v in [('config',cfg),('training_seed',seed),('split_seed',seed),('inner_ensemble',False)]):raise ValueError('Candidate identity differs')
    if member['config']!=cfg or member['training_seed']!=seed or member['split_seed']!=seed or member['inner_ensemble'] is not False or member['checkpoints']['accuracy']!=r['selected']:
        raise ValueError('Candidate payload differs')
    return r

def rank(seed,ds,configs):
    rows=[]
    for cfg in dedup(configs):
        r=result(seed,ds,cfg);v=r['selected']
        rows.append(dict(id=cid(cfg),config=cfg,accuracy=v['accuracy'],nll=v['nll'],step=v['step'],origin=r['origin'],training_seed=seed))
    return rank_rows(rows)

def verify_selected_model(final_obj,member,config,seed):
    if final_obj.get("config")!=config or final_obj.get("training_seed")!=seed:raise ValueError("Final model identity differs")
    if (member.get("config")!=config or member.get("training_seed")!=seed or member.get("split_seed")!=seed
            or member.get("inner_ensemble") is not False):raise ValueError("Selected member identity differs")
    final_state=final_obj.get("state");selected=member.get("states",{}).get("accuracy")
    if not isinstance(final_state,dict) or not isinstance(selected,dict) or set(final_state)!=set(selected):raise ValueError("Selected model state keys differ")
    if any(not torch.is_tensor(final_state[k]) or not torch.is_tensor(selected[k]) or not torch.equal(final_state[k],selected[k]) for k in final_state):
        raise ValueError("Selected model state differs")

def persist(seed,ds,cfg,member,origin):
    dest=candidate_path(seed,ds,cfg);save(dest/'checkpoint.pt',member)
    write(dest/'result.json',dict(config=cfg,training_seed=seed,split_seed=seed,inner_ensemble=False,origin=origin,
        selected=member['checkpoints']['accuracy'],checkpoints=member['checkpoints'],history=member['history'],
        train_seconds=member['train_seconds'] if origin=='trained' else 0.,parameter_count=member['parameter_count'],
        checkpoint_sha256=sha(dest/'checkpoint.pt'),test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))

def initialize(index):
    training_open();reg=verify_code();seed,ds=CELLS[index];record=reg['inputs'][key(seed,ds)]
    f,s,p,h,split,router=load(seed,ds);torch.set_num_threads(2)
    old=torch.load(record['baseline_checkpoint'],weights_only=True,map_location='cpu');cfg=expand_v1(old['config'])
    model=make_model(f,h,cfg);model.load_state_dict(old['states']['accuracy']);prob=predict(model,f,h,s['selection_ids'])
    torch.testing.assert_close(prob,old['validation_probability'],atol=0,rtol=0)
    if not torch.equal(old['validation_ids'],s['selection_ids']):raise ValueError('Inherited validation IDs differ')
    observed=score(prob,s['selection_targets']);v=old['checkpoints']['accuracy']
    if any(abs(observed[k]-v[k])>1e-12 for k in ('accuracy','nll')):raise ValueError('Inherited validation metric differs')
    member=dict(old,config=cfg,inherited_from=record['baseline_checkpoint'],inherited_sha256=sha(record['baseline_checkpoint']))
    persist(seed,ds,cfg,member,'inherited_v1')
    write(cell(seed,ds)/'BASELINE.json',dict(config=cfg,candidate_id=cid(cfg),source=record['baseline_checkpoint'],source_sha256=sha(record['baseline_checkpoint'])))
    write(cell(seed,ds)/'DIAGNOSTICS.json',diagnostic(f,s,p,prob))
    write(cell(seed,ds)/'pools.json',dict(counts={k:len(v) for k,v in p.items()},gold_count=len(s['gold_ids']),validation_count=len(s['selection_ids']),test_truth_loaded=False))
    files={str(x):sha(x) for x in (cell(seed,ds)/'BASELINE.json',cell(seed,ds)/'DIAGNOSTICS.json',cell(seed,ds)/'pools.json',candidate_path(seed,ds,cfg)/'checkpoint.pt',candidate_path(seed,ds,cfg)/'result.json')}
    write(cell(seed,ds)/'INITIALIZED.json',dict(files=files,all_checks_pass=True,training_seed=seed,inner_seeds=[]))
    print('INITIALIZED',seed,ds,flush=True)

def checked_stage(stage):
    spec=read(RUN/f'stage{stage}.json')
    for seed,ds in CELLS:
        done=read(RUN/f'stage{stage}_complete'/f'{seed}_{ds}.json')
        expected={cid(c):sha(candidate_path(seed,ds,c)/'result.json') for c in spec['candidates'][key(seed,ds)]}
        if done['stage']!=stage or done['seed']!=seed or done['dataset']!=ds or done['candidates']!=expected:raise ValueError('Stage completion differs')
    return spec

def prior_configs(seed,ds,stages):
    return [read(cell(seed,ds)/'BASELINE.json')['config']]+[c for st in stages for c in st['candidates'][key(seed,ds)]]

def plan(stage):
    training_open();verify_code();previous=[checked_stage(i) for i in range(1,stage)];candidates={};evidence={};diagnostics={}
    for seed,ds in CELLS:
        check(read(cell(seed,ds)/'INITIALIZED.json')['files'])
        k=key(seed,ds);rows=rank(seed,ds,prior_configs(seed,ds,previous));diag=read(cell(seed,ds)/'DIAGNOSTICS.json')
        candidates[k]=catalog(stage,rows[0]['config'],diag['class_weight_eligible']);evidence[k]=rows
        diagnostics[k]=sha(cell(seed,ds)/'DIAGNOSTICS.json')
    write(RUN/f'stage{stage}.json',dict(stage=stage,candidates=candidates,selection_evidence=evidence,diagnostic_hashes=diagnostics,
        criterion='own_validation_accuracy_then_nll_then_inherited_then_id',cross_split_selection=False,test_truth_loaded=False))
    print('PLANNED',stage,{k:len(c) for k,c in candidates.items()},flush=True)

def worker(stage,index):
    training_open();verify_code();seed,ds=CELLS[index];spec=read(RUN/f'stage{stage}.json');f,s,p,h,split,router=load(seed,ds)
    for cfg in spec['candidates'][key(seed,ds)]:
        training_open();dest=candidate_path(seed,ds,cfg)
        if (dest/'result.json').exists():result(seed,ds,cfg);continue
        if (dest/'checkpoint.pt').exists():raise ValueError('Partial candidate requires explicit recovery audit')
        member=train(f,s,p,h,cfg,seed);persist(seed,ds,cfg,member,'trained')
        print('TRIAL',stage,seed,ds,cid(cfg),member['checkpoints']['accuracy'],round(member['train_seconds'],2),flush=True)
    write(RUN/f'stage{stage}_complete'/f'{seed}_{ds}.json',dict(stage=stage,seed=seed,dataset=ds,
        candidates={cid(c):sha(candidate_path(seed,ds,c)/'result.json') for c in spec['candidates'][key(seed,ds)]},test_truth_loaded=False))

def select():
    training_open();verify_code();stages=[checked_stage(i) for i in range(1,5)];cells={}
    for seed,ds in CELLS:
        rows=rank(seed,ds,prior_configs(seed,ds,stages));cells[key(seed,ds)]=dict(winner=rows[0],ranking=rows)
    write(RUN/'SELECTION.json',dict(cells=cells,inner_seeds=[],cross_split_selection=False,test_truth_loaded=False,
        rule='own_validation_accuracy_then_nll_then_inherited_then_id'))
    print('SELECTED25 cells',flush=True)

def export(index):
    training_open();verify_code();torch.set_num_threads(2);seed,ds=CELLS[index]
    row=read(RUN/'SELECTION.json')['cells'][key(seed,ds)]['winner'];cfg=row['config'];r=result(seed,ds,cfg)
    f,s,p,h,split,router=load(seed,ds);member=torch.load(candidate_path(seed,ds,cfg)/'checkpoint.pt',weights_only=True,map_location='cpu')
    model=make_model(f,h,cfg);model.load_state_dict(member['states']['accuracy']);prob=predict(model,f,h,f['node_ids'])
    torch.testing.assert_close(prob[s['selection_ids']],member['validation_probability'],atol=2e-6,rtol=0)
    dest=cell(seed,ds)/'final'
    save(dest/'predictions.pt',dict(node_ids=f['node_ids'],probability=prob,training_seed=seed,inner_ensemble=False))
    save(dest/'model.pt',dict(state=model.state_dict(),config=cfg,training_seed=seed,origin=row['origin']))
    a,b=model.coefficients()
    write(dest/'selection.json',dict(config=cfg,candidate_id=cid(cfg),validation=r['selected'],origin=row['origin'],training_seed=seed,
        alpha=float(a.detach()),beta=float(b.detach()),weights=model.weights().detach().tolist(),parameter_count=r['parameter_count'],
        candidate_sha256=r['checkpoint_sha256'],test_truth_loaded=False))
    write(dest/'complete.json',dict(files={str(x):sha(x) for x in dest.iterdir() if x.is_file()},all_checks_pass=True))
    print('EXPORTED',seed,ds,flush=True)

def freeze():
    training_open();verify_all();stages=[checked_stage(i) for i in range(1,5)];selected=read(RUN/'SELECTION.json');files={}
    for seed,ds in CELLS:
        rows=rank(seed,ds,prior_configs(seed,ds,stages))
        if selected['cells'][key(seed,ds)]['ranking']!=rows:raise ValueError('Selection reconstruction differs')
        f,s,p,h,split,router=load(seed,ds)
        for row in rows:
            dest=candidate_path(seed,ds,row['config']);m=torch.load(dest/'checkpoint.pt',weights_only=True,map_location='cpu')
            if not torch.equal(m['validation_ids'],s['selection_ids']):raise ValueError('Candidate validation IDs differ')
            value=score(m['validation_probability'],s['selection_targets'])
            if any(abs(value[k]-row[k])>1e-12 for k in ('accuracy','nll')):raise ValueError('Candidate score cannot be reproduced')
        final=cell(seed,ds)/'final';receipt=read(final/'complete.json');check(receipt['files'])
        selection=read(final/'selection.json')
        if selection['candidate_id']!=rows[0]['id'] or selection['training_seed']!=seed:raise ValueError('Export selection differs')
        final_obj=torch.load(final/"model.pt",weights_only=True,map_location="cpu")
        selected_member=torch.load(candidate_path(seed,ds,rows[0]["config"])/"checkpoint.pt",weights_only=True,map_location="cpu")
        verify_selected_model(final_obj,selected_member,rows[0]["config"],seed)
        pred=torch.load(final/'predictions.pt',weights_only=True,map_location='cpu')
        if not torch.equal(pred['node_ids'],f['node_ids']):raise ValueError('Prediction node order differs')
        torch.testing.assert_close(pred['probability'].sum(1),torch.ones(len(f['node_ids'])),atol=2e-6,rtol=0)
        value=score(pred['probability'][s['selection_ids']],s['selection_targets'])
        if abs(value['accuracy']-rows[0]['accuracy'])>1e-12 or abs(value['nll']-rows[0]['nll'])>2e-6:raise ValueError('Export validation differs')
        for path in cell(seed,ds).rglob('*'):
            if path.is_file():files[str(path)]=sha(path)
        print('FREEZE VERIFIED',seed,ds,len(rows),flush=True)
    for path in [RUN/'REGISTRATION.json',RUN/'SELECTION.json']+[RUN/f'stage{i}.json' for i in range(1,5)]:files[str(path)]=sha(path)
    for i in range(1,5):
        for path in (RUN/f'stage{i}_complete').glob('*.json'):files[str(path)]=sha(path)
    write(RUN/'FROZEN.json',dict(files=files,all_checks_pass=True,cell_count=25,inner_seeds=[],training_seed_equals_split_seed=True,
        pseudo_labels_unchanged=True,cross_split_selection=False,test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('GLOBAL FREEZE COMPLETE; all new fitting/selection closed',flush=True)
