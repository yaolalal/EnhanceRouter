"""Freeze first; evaluate and independently audit saved predictions afterwards."""
import csv
import itertools
import json
import os
import statistics
import numpy as np
import torch
from .protocol import RUN,REFERENCE,DATASETS,SEEDS,read,write,save,sha,check,cell,load,slurm,training_open,verify_code
from .search import checked_stage,key,cid,candidate_path,rank
from .training import score,make_model,predict

METRICS=('fixed_accuracy','fixed_macro_f1','new_accuracy','new_macro_f1','delta_accuracy','delta_macro_f1',
         'no_b_accuracy','no_b_macro_f1','pseudo_accuracy','pseudo_coverage','historical_accuracy','historical_macro_f1')

def freeze():
    training_open();reg=verify_code();check(reg['dependencies']);check(reg['protected'])
    stages=[checked_stage(i) for i in range(1,5)];selected=read(RUN/'SELECTION.json')
    files={str(RUN/'REGISTRATION.json'):sha(RUN/'REGISTRATION.json'),str(RUN/'SELECTION.json'):sha(RUN/'SELECTION.json')}
    for stage in range(1,5):
        files[str(RUN/f'stage{stage}.json')]=sha(RUN/f'stage{stage}.json')
        for p in (RUN/f'stage{stage}_complete').glob('*.json'):files[str(p)]=sha(p)
    for seed,ds in itertools.product(SEEDS,DATASETS):
        k=key(seed,ds);configs=[c for stage in stages for c in stage['candidates'][k]]
        ranking=rank(seed,ds,configs)
        if selected['cells'][k]['ranking']!=ranking:raise ValueError('Validation selection reconstruction failed')
        f,s,p,h,split,router=load(seed,ds)
        for row in ranking:
            path=candidate_path(seed,ds,row['config'])
            member=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
            if not torch.equal(member['validation_ids'],s['selection_ids']):raise ValueError('Validation IDs differ')
            observed=score(member['validation_probability'],s['selection_targets'])
            for metric in ('accuracy','nll'):
                if abs(observed[metric]-row[metric])>1e-12:raise ValueError('Candidate validation metric cannot be reconstructed')
            files[str(path/'result.json')]=sha(path/'result.json')
            files[str(path/'checkpoint.pt')]=sha(path/'checkpoint.pt')
        for folder in ('fixed','final'):
            receipt=cell(seed,ds)/folder/'complete.json';r=read(receipt);check(r['files'])
            files.update(r['files']);files[str(receipt)]=sha(receipt)
        from .fixed import verify_saved
        verify_saved(seed,ds,f,s,h)
        final=read(cell(seed,ds)/'final/selection.json')
        if final['candidate_id']!=ranking[0]['id'] or final['training_seed']!=seed:raise ValueError('Final selection mismatch')
        pred=torch.load(cell(seed,ds)/'final/predictions.pt',weights_only=True,map_location='cpu')
        for name in ('node_ids',):
            if not torch.equal(pred[name],f[name]):raise ValueError('Final node order differs')
        torch.testing.assert_close(pred['probability'].sum(1),torch.ones(len(f['node_ids'])),atol=2e-6,rtol=0)
        observed=score(pred['probability'][s['selection_ids']],s['selection_targets'])
        if abs(observed['accuracy']-ranking[0]['accuracy'])>1e-12 or abs(observed['nll']-ranking[0]['nll'])>2e-6:raise ValueError('Export validation parity failed')
        print('FREEZE VERIFIED',seed,ds,len(ranking),'candidates',flush=True)
    write(RUN/'FROZEN.json',dict(files=files,all_checks_pass=True,cell_count=len(SEEDS)*len(DATASETS),seeds=list(SEEDS),inner_seeds=[],
          test_truth_loaded=False,cross_split_selection=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('GLOBAL FREEZE COMPLETE; all training closed',flush=True)

def verify_frozen():
    slurm();reg=verify_code();frozen=read(RUN/'FROZEN.json')
    if not frozen['all_checks_pass'] or frozen['cell_count']!=len(SEEDS)*len(DATASETS) or frozen['inner_seeds']!=[] or frozen['cross_split_selection'] is not False:raise ValueError('Invalid global freeze')
    check(frozen['files']);check(reg['dependencies']);check(reg['protected']);return reg,frozen

