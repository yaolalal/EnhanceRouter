"""Immutable search stages and global winner freeze. No test evaluator import."""
import os
from . import protocol as p
from .engine import Context,train
from eargtc.joint_explore.gnn_downstream.io import read,write,load,sha,verify,require,DATASETS,VARIANTS,ROOT
import torch

def initialize():
    require();p.open_training()
    assert read(p.SOURCE/'FINAL_AUDIT.json')['all_pass']
    verify(read(p.SOURCE/'INPUT_AUDIT.json')['files'])
    contract=read(p.RUN/'TEST_GATE.json');assert contract['all_pass'];verify(contract['code'])
    pilot=read(p.RUN/'PILOT_GATE.json');assert pilot['all_pass'];verify(pilot['code']);verify(pilot['files'])
    write(p.RUN/'CONFIG.json',p.SPEC)
    manifests={str(f):sha(f) for f in (p.RUN/'CONFIG.json',p.RUN/'TEST_GATE.json',p.RUN/'PILOT_GATE.json',p.PLAN,
                p.SOURCE/'INPUT_AUDIT.json',p.SOURCE/'PREREGISTERED.json',p.SOURCE/'FROZEN.json',p.SOURCE/'FINAL_AUDIT.json',p.SOURCE/'DELIVERY.json')}
    for ds in DATASETS:
        graph=load(p.SOURCE/f'inputs/{ds}/graph.pt')
        assert set(graph)=={'encoder_config','input_dim','x','edge_index','embeddings','encoder_state'}
        for seed in range(42,47):p.check_item(load(p.SOURCE/f'inputs/{ds}/seed{seed}.pt'),len(graph['x']),seed)
    write(p.RUN/'PREREGISTERED.json',{'code':p.code_hashes(),'manifests':manifests,
          'source_input_files':read(p.SOURCE/'INPUT_AUDIT.json')['files'],'slurm_job_id':os.environ['SLURM_JOB_ID'],
          'query_truth_used':False})
    print('PREREGISTERED',flush=True)

def legacy(ctx,arm,fine,dest):
    path=p.SOURCE/f'cells/seed{ctx.seed}/{ctx.item["dataset"]}/{arm}_{"finetune" if fine else "frozen"}'
    done=p.bind_legacy(path,read(p.SOURCE/'FROZEN.json')['files'])
    verify({str(path/f):done['files'][str(path/f)] for f in ('checkpoint.pt','metadata.json','history.json')})
    old=read(path/'metadata.json');assert old['model_seed']==ctx.seed and not old['query_targets_accessed']
    c=p.base(ctx.item,arm);c['mode']='all' if fine else 'frozen'
    validation={**old['selected_validation'],'correct':round(old['selected_validation']['accuracy']*len(ctx.valid)),'count':len(ctx.valid)}
    meta={'id':'legacy','config':c,'validation':validation,'selected_epoch':old['selected_epoch'],
          'seed':ctx.seed,'dataset':ctx.item['dataset'],'arm':arm,'legacy':True,'query_targets_accessed':False,
          'checkpoint':str(path/'checkpoint.pt'),'source_metadata':str(path/'metadata.json'),
          'source_files':{str(path/f):done['files'][str(path/f)] for f in ('checkpoint.pt','metadata.json','history.json')}}
    write(dest/'legacy.json',meta);return meta

def candidate(ctx,c,arm,dest):
    trial=dest/'trials'/p.config_id(c);meta=train(ctx,c,arm,trial)
    return {**meta,'checkpoint':str(trial/'checkpoint.pt'),'complete':str(trial/'COMPLETE.json')}

def search_stage(ctx,arm,fine,frozen=None):
    dest=p.RUN/f'cells/seed{ctx.seed}/{ctx.item["dataset"]}/{arm}_{"finetune" if fine else "frozen"}'
    if (dest/'SELECTION.json').exists():
        selected=read(dest/'SELECTION.json');verify(selected['files']);return selected['winner']
    old=legacy(ctx,arm,fine,dest)
    configs=p.primary_fine(frozen) if fine else p.primary_frozen(ctx.item,arm)
    rows=[old]+[candidate(ctx,c,arm,dest) for c in configs]
    primary=p.choose(rows);primary_ids=[r['id'] for r in rows]
    write(dest/'PRIMARY.json',{'winner':primary,'candidate_ids':primary_ids})
    known={p.config_id(c) for c in configs}
    refined=[]
    for c in p.refinement(primary,arm,fine):
        cid=p.config_id(c)
        if cid in known:continue
        known.add(cid);refined.append(cid);rows.append(candidate(ctx,c,arm,dest))
    winner=p.choose(rows);files={str(dest/f):sha(dest/f) for f in ('legacy.json','PRIMARY.json')}
    for r in rows:
        if r['legacy']:files.update(r['source_files'])
        else:
            files.update(read(r['complete'])['files']);files[r['complete']]=sha(r['complete'])
    write(dest/'SELECTION.json',{'winner':winner,'candidate_count':len(rows),'trained_count':len(rows)-1,
          'primary_ids':primary_ids,'refinement_ids':refined,'candidates':rows,'files':files,
          'preregistration_sha256':sha(p.RUN/'PREREGISTERED.json'),'query_truth_used':False})
    print('SELECTED',ctx.item['dataset'],ctx.seed,arm,fine,winner['id'],winner['validation'],flush=True)
    return winner

def search(index):
    p.open_training();pre=p.gate();torch.set_num_threads(2)
    ds=DATASETS[index//5];seed=42+index%5
    completed=p.RUN/f'cells/seed{seed}/{ds}/COMPLETE.json'
    if completed.exists():
        verify(read(completed)['files'])
        for variant in VARIANTS:verify(read(completed.parent/variant/'SELECTION.json')['files'])
        print('Already complete',ds,seed,flush=True);return
    gp=p.SOURCE/f'inputs/{ds}/graph.pt';ip=p.SOURCE/f'inputs/{ds}/seed{seed}.pt'
    verify({str(f):pre['source_input_files'][str(f)] for f in (gp,ip)})
    ctx=Context(load(gp),{**load(ip),'dataset':ds},seed)
    for arm in ('router','gold'):
        frozen=search_stage(ctx,arm,False)
        search_stage(ctx,arm,True,frozen)
    p.gate();dest=p.RUN/f'cells/seed{seed}/{ds}'
    files={str(dest/v/'SELECTION.json'):sha(dest/v/'SELECTION.json') for v in VARIANTS}
    write(dest/'COMPLETE.json',{'files':files,'seed':seed,'dataset':ds,'slurm_job_id':os.environ['SLURM_JOB_ID']})
    print('CELL COMPLETE',ds,seed,flush=True)

def freeze():
    p.open_training();pre=p.gate();verify(pre['source_input_files'])
    files={};winners=[];trials=0
    for ds in DATASETS:
        for seed in range(42,47):
            cell=p.RUN/f'cells/seed{seed}/{ds}';complete=read(cell/'COMPLETE.json');verify(complete['files'])
            files.update(complete['files']);files[str(cell/'COMPLETE.json')]=sha(cell/'COMPLETE.json')
            item=load(p.SOURCE/f'inputs/{ds}/seed{seed}.pt');arm_frozen={}
            for variant in VARIANTS:
                arm,mode=variant.split('_');fine=mode=='finetune';sel=read(cell/variant/'SELECTION.json')
                assert not sel['query_truth_used'] and sel['preregistration_sha256']==sha(p.RUN/'PREREGISTERED.json')
                verify(sel['files']);files.update(sel['files']);rows=sel['candidates']
                assert p.choose(rows)==sel['winner']
                configs=p.primary_fine(arm_frozen[arm]) if fine else p.primary_frozen(item,arm)
                assert sel['primary_ids']==['legacy']+[p.config_id(c) for c in configs]
                primary=p.choose([r for r in rows if r['id'] in sel['primary_ids']])
                assert read(cell/variant/'PRIMARY.json')['winner']==primary
                expected=[];known={p.config_id(c) for c in configs}
                for c in p.refinement(primary,arm,fine):
                    cid=p.config_id(c)
                    if cid not in known:expected.append(cid);known.add(cid)
                assert expected==sel['refinement_ids']
                assert [r['id'] for r in rows]==sel['primary_ids']+expected
                for r in rows:
                    assert r['seed']==seed and r['arm']==arm and not r['query_targets_accessed']
                    if r['legacy']:continue
                    history=read(Path(r['complete']).parent/'history.json')
                    candidates=[{'validation':r['initial_validation'],'selected_epoch':0,'id':r['id']}]+[{'validation':h['validation'],'selected_epoch':h['epoch'],'id':r['id']} for h in history]
                    choice=p.choose(candidates)
                    assert choice['validation']==r['validation'] and choice['selected_epoch']==r['selected_epoch']
                    assert r['id']==p.config_id(r['config'])
                    if arm=='gold':assert r['accepted_training_count']==0 and r['config']['lambda']==0
                if not fine:arm_frozen[arm]=sel['winner']
                winners.append({'dataset':ds,'seed':seed,'variant':variant,**sel['winner']})
                trials+=sel['trained_count']
    write(p.RUN/'FROZEN.json',{'winners':winners,'winner_count':len(winners),'trained_trials':trials,'files':files,
          'preregistration_sha256':sha(p.RUN/'PREREGISTERED.json'),'query_truth_used':False,'slurm_job_id':os.environ['SLURM_JOB_ID']})
    print('FROZEN',len(winners),'winners;',trials,'new trials',flush=True)

from pathlib import Path
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['initialize','search','freeze']);parser.add_argument('--index',type=int)
    a=parser.parse_args()
    if a.action=='search':search(a.index)
    else:globals()[a.action]()
