"""Fresh own-shot V1 then V2 search; immutable per-cell classifier artifacts."""
import argparse
import fcntl
import itertools
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from eargtc.joint_explore.joint_purf_v10_v1.protocol import read, write, save, sha, check, slurm
from eargtc.joint_explore.joint_purf_v10_v1 import search as v1search
from eargtc.joint_explore.joint_purf_v10_v1 import training as v1train
from eargtc.joint_explore.joint_purf_v10_v2 import search as v2search
from eargtc.joint_explore.joint_purf_v10_v2 import training as v2train
from eargtc.joint_explore.joint_purf_v10_v2.protocol import diagnostic
from eargtc.multiseed_joint.experiment import prepare_arrays
from eargtc.joint_cross_repair.artifacts import validate_hidden

REPO=Path(__file__).resolve().parents[2]
RUN=REPO/'runs/router_v10_shots_v1_20260914'
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SEEDS=(42,43,44)
SHOTS=(3,10)
CELLS=tuple(itertools.product(SHOTS,SEEDS,DATASETS))

def training_open(root=RUN):
    slurm()
    if any((root/name).exists() for name in ('DOWNSTREAM_FROZEN.json','EVALUATION_STARTED.json','COMPLETE.json')):
        raise RuntimeError('Downstream fitting is closed after global freeze')

def validate_split(split,shot,seed,dataset):
    if (split['dataset']!=dataset or split['split_seed']!=seed or split.get('shots',split.get('shot'))!=shot
            or split.get('test_truth_exported') is not False):raise ValueError('Split identity or truth contract differs')
    fields=('support_ids','valid_ids','unlabeled_ids')
    groups=[set(split[k]) for k in fields]
    if any(len(g)!=len(split[k]) for g,k in zip(groups,fields)):raise ValueError('Duplicate split IDs')
    if any(a&b for a,b in itertools.combinations(groups,2)):raise ValueError('Support/validation/query overlap')
    if set.union(*groups)!=set(range(split['num_nodes'])):raise ValueError('Incomplete population partition')
    if split['unlabeled_ids']!=split['evaluation_full_ids']:raise ValueError('Full evaluation differs from query')
    if split['evaluation_test_ids']!=split['evaluation_1000_ids']:raise ValueError('Fixed evaluation order changed')
    ids=split['evaluation_test_ids']
    if len(ids)!=len(set(ids)) or not ids or not set(ids)<=groups[2]:raise ValueError('Invalid fixed query subset')

def cell(root,shot,seed,ds):return root/f'shot{shot}/seed{seed}/downstream/{ds}'
def paths(root,shot,seed,ds):
    upstream=root/f'shot{shot}/seed{seed}/upstream/{ds}'
    return dict(split=root/f'shot{shot}/splits/{ds}_{shot}shot_seed{seed}.json',
                upstream=upstream/'pipeline/predictions.pt', embeddings=upstream/'embeddings.pt',
                supervision=upstream/'supervision.pt', router=root/f'shot{shot}/router/seed{seed}/joint_inputs/{ds}.pt',
                hidden=REPO/f'runs/router_v7_decision_llm_hidden_20260827_v1/{ds}/hidden.pt')

def registration(root):
    slurm();reg=read(root/'DOWNSTREAM_REGISTRATION.json');check(reg['code_hashes'])
    if sha(root/'ROUTER_FROZEN.json')!=reg['router_frozen_sha256']:raise ValueError('Router freeze changed')
    return reg

def register(root=RUN):
    training_open(root);frozen=read(root/'ROUTER_FROZEN.json')
    if frozen.get('all_checks_pass') is not True or frozen.get('cell_count')!=30:raise ValueError('All30 Router freeze required')
    check(frozen['files']);records={}
    for shot,seed,ds in CELLS:
        pp=paths(root,shot,seed,ds);split=read(pp['split']);validate_split(split,shot,seed,ds)
        if str(pp['router']) not in frozen['files']:raise ValueError('Canonical router not globally bound')
        complete=pp['supervision'].parent/'COMPLETE.json';up=read(complete)
        if (up.get("dataset"),up.get("split_seed"),up.get("shot"))!=(ds,seed,shot):raise ValueError("Upstream completion identity differs")
        if up.get('all_checks_pass') is not True or up.get('test_truth_exported') is not False:raise ValueError('Unsafe upstream completion')
        bound={str(complete.parent/k):v for k,v in up['files'].items()}
        for k in ('upstream','supervision','embeddings'):
            if str(pp[k]) not in bound or sha(pp[k])!=bound[str(pp[k])]:raise ValueError('Unbound upstream '+k)
        records[f'{shot}/{seed}/{ds}']=dict(paths={k:str(v) for k,v in pp.items()},hashes={str(v):sha(v) for v in pp.values()},upstream_complete_sha256=sha(complete))
    code={str(Path(__file__).resolve()):sha(Path(__file__).resolve())}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and path.endswith('.py') and Path(path).is_absolute() and Path(path).is_file() and Path(path).is_relative_to(REPO):code[str(Path(path))]=sha(path)
    write(root/'DOWNSTREAM_REGISTRATION.json',dict(inputs=records,code_hashes=code,router_frozen_sha256=sha(root/'ROUTER_FROZEN.json'),
        cell_count=30,classifier='Joint-PURF-v2',fresh_v1_four_stages=True,fresh_v2_four_stages=True,
        training_seed_equals_split_seed=True,inner_seeds=[],cross_split_selection=False,test_truth_loaded=False,
        historical_5shot_checkpoint_transfer=False,slurm_job_id=os.environ['SLURM_JOB_ID']))

def load(root,shot,seed,ds):
    reg=registration(root);record=reg['inputs'][f'{shot}/{seed}/{ds}'];check(record['hashes']);pp=record['paths']
    split=read(pp['split']);validate_split(split,shot,seed,ds)
    values={k:torch.load(pp[k],map_location='cpu',weights_only=True,mmap=True) for k in ('upstream','embeddings','supervision','router')}
    f,s,p=prepare_arrays(split,values['upstream'],values['embeddings'],values['supervision'],values['router'],ds)
    hidden=torch.load(pp['hidden'],map_location='cpu',weights_only=True,mmap=True);validate_hidden(hidden,split['num_nodes'])
    if hidden['metadata']['dataset']!=ds:raise ValueError('Hidden dataset differs')
    return f,s,p,hidden['hidden'],split,values['router']

def result(dest,cfg,seed):
    r=read(dest/'result.json');check({str(dest/'checkpoint.pt'):r['checkpoint_sha256']})
    if r['config']!=cfg or r['training_seed']!=seed or r['test_truth_loaded'] is not False:raise ValueError('Candidate identity differs')
    return r

def fit(root,dest,cfg,seed,f,s,p,h,version,member=None):
    training_open(root)
    if (dest/'result.json').exists():return result(dest,cfg,seed)
    if (dest/'checkpoint.pt').exists():raise ValueError('Partial candidate needs explicit recovery: '+str(dest))
    inherited=member is not None
    if member is None:member=(v1train if version==1 else v2train).train(f,s,p,h,cfg,seed)
    save(dest/'checkpoint.pt',member)
    r=dict(config=cfg,training_seed=seed,split_seed=seed,inner_ensemble=False,selected=member['checkpoints']['accuracy'],
           origin='inherited_v1' if inherited else 'trained',checkpoint_sha256=sha(dest/'checkpoint.pt'),
           train_seconds=0. if inherited else member['train_seconds'],test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(dest/'result.json',r);print('CLASSIFIER',version,seed,dest.name,r['selected'],flush=True)
    return r

def rank(base,configs,seed,version):
    rows=[]
    for cfg in (v1search if version==1 else v2search).dedup(configs):
        cid=(v1search if version==1 else v2search).cid(cfg);r=result(base/cid,cfg,seed)
        rows.append(dict(id=cid,config=cfg,accuracy=r['selected']['accuracy'],nll=r['selected']['nll'],origin=r['origin']))
    return (sorted(rows,key=lambda r:(-r['accuracy'],r['nll'],r['id'])) if version==1 else v2search.rank_rows(rows))

def run_cell(root,index):
    training_open(root);shot,seed,ds=CELLS[index];dest=cell(root,shot,seed,ds);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (dest/'COMPLETE.json').exists():check(read(dest/'COMPLETE.json')['files']);return
        f,s,p,h,split,router=load(root,shot,seed,ds)
        write(dest/'pools.json',dict(counts={k:len(v) for k,v in p.items()},gold_count=len(s['gold_ids']),validation_count=len(s['selection_ids']),test_truth_loaded=False))
        previous=[];base=dest/'v1/candidates'
        for stage in range(1,5):
            if stage==1:configs=v1search.core()
            elif stage==2:configs=v1search.schemes(rank(base,previous[0],seed,1)[0]['config'])
            elif stage==3:configs=v1search.refinement(rank(base,previous[1],seed,1)[0]['config'])
            else:configs=v1search.removals(rank(base,[c for g in previous for c in g],seed,1)[0]['config'])
            write(dest/f'v1/stage{stage}.json',dict(configs=configs,test_truth_loaded=False))
            for cfg in configs:fit(root,base/v1search.cid(cfg),cfg,seed,f,s,p,h,1)
            previous.append(configs)
        oldwinner=rank(base,[c for g in previous for c in g],seed,1)[0]
        old=torch.load(base/oldwinner['id']/'checkpoint.pt',weights_only=True,map_location='cpu');cfg=v2train.expand_v1(old['config'])
        # Preserve exact V1 checkpoint parity when promoting it to the V2 candidate set.
        model=v2train.make_model(f,h,cfg);model.load_state_dict(old['states']['accuracy'])
        torch.testing.assert_close(v2train.predict(model,f,h,s['selection_ids']),old['validation_probability'],atol=0,rtol=0)
        base=dest/'v2/candidates';promoted=dict(old,config=cfg)
        fit(root,base/v2search.cid(cfg),cfg,seed,f,s,p,h,2,member=promoted)
        diag=diagnostic(f,s,p,old['validation_probability']);write(dest/'DIAGNOSTICS.json',diag)
        previous=[cfg]
        for stage in range(1,5):
            anchor=rank(base,previous,seed,2)[0];configs=v2search.catalog(stage,anchor['config'],diag['class_weight_eligible'])
            write(dest/f'v2/stage{stage}.json',dict(configs=configs,anchor=anchor,test_truth_loaded=False))
            for cfg in configs:fit(root,base/v2search.cid(cfg),cfg,seed,f,s,p,h,2)
            previous.extend(configs)
        ranked=rank(base,previous,seed,2);winner=ranked[0];member=torch.load(base/winner['id']/'checkpoint.pt',weights_only=True,map_location='cpu')
        model=v2train.make_model(f,h,winner['config']);model.load_state_dict(member['states']['accuracy']);prob=v2train.predict(model,f,h,f['node_ids'])
        torch.testing.assert_close(prob[s['selection_ids']],member['validation_probability'],atol=2e-6,rtol=0)
        training_open(root);final=dest/'final'
        save(final/'predictions.pt',dict(node_ids=f['node_ids'],probability=prob,training_seed=seed,split_seed=seed,shots=shot,inner_ensemble=False))
        save(final/'model.pt',dict(state=model.state_dict(),config=winner['config'],training_seed=seed))
        write(final/'selection.json',dict(winner=winner,ranking=ranked,v1_winner=oldwinner,test_truth_loaded=False))
        files={str(x):sha(x) for x in dest.rglob('*') if x.is_file() and x.name not in ('.lock','COMPLETE.json')}
        write(dest/'COMPLETE.json',dict(files=files,all_checks_pass=True,shot=shot,seed=seed,dataset=ds,test_truth_loaded=False,slurm_job_id=os.environ['SLURM_JOB_ID']))

def freeze(root):
    training_open(root);registration(root);files={str(root/'DOWNSTREAM_REGISTRATION.json'):sha(root/'DOWNSTREAM_REGISTRATION.json')}
    for shot,seed,ds in CELLS:
        dest=cell(root,shot,seed,ds);receipt=read(dest/'COMPLETE.json');check(receipt['files'])
        if receipt['all_checks_pass'] is not True or receipt['test_truth_loaded'] is not False:raise ValueError('Incomplete downstream cell')
        f,s,p,h,split,router=load(root,shot,seed,ds)
        pred=torch.load(dest/'final/predictions.pt',weights_only=True,map_location='cpu');obj=torch.load(dest/'final/model.pt',weights_only=True,map_location='cpu')
        selected=read(dest/'final/selection.json')['winner'];cp=dest/'v2/candidates'/selected['id']/'checkpoint.pt'
        member=torch.load(cp,weights_only=True,map_location='cpu');v2search.verify_selected_model(obj,member,selected['config'],seed)
        if pred['training_seed']!=seed or pred['shots']!=shot or not torch.equal(pred['node_ids'],f['node_ids']):raise ValueError('Prediction identity changed')
        model=v2train.make_model(f,h,obj['config']);model.load_state_dict(obj['state'])
        torch.testing.assert_close(pred['probability'],v2train.predict(model,f,h,f['node_ids']),atol=0,rtol=0)
        files.update(receipt['files']);files[str(dest/'COMPLETE.json')]=sha(dest/'COMPLETE.json')
        print('CLASSIFIER FREEZE VERIFIED',shot,seed,ds,flush=True)
    write(root/'DOWNSTREAM_FROZEN.json',dict(files=files,cell_count=30,all_checks_pass=True,test_truth_loaded=False,inner_seeds=[],slurm_job_id=os.environ['SLURM_JOB_ID']))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','cell','freeze'));parser.add_argument('--root',type=Path,default=RUN);parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args();slurm();torch.set_num_threads(2)
    if args.command=='register':register(args.root)
    elif args.command=='cell':run_cell(args.root,args.index)
    else:freeze(args.root)
if __name__=='__main__':main()
