"""Slurm-only reproduction: authenticate -> fit 15 cells -> freeze -> evaluate."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from .io import (ROOT, DATASETS, Inputs, assert_imports_isolated, command, digest,
                 environment, guard_writes, local, read, require_slurm, save, source_state, write)


def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if hasattr(x,'tolist'):return x.tolist()
    return x


def router_recipe(bundle):
    recipe={k:v for k,v in bundle.items() if k in ('dataset','branch','tuning_schema','version','shots',
             'split_seed','affine','boundary','calibration','route_recipe','policy','unavailable')}
    recipe['heads']={h:{k:v for k,v in w.items() if k in ('engine','gold_normalization')}
          | {'config':w['model']['config']} for h,w in bundle.get('models',{}).items()}
    return jsonable(recipe)


def prepare(run, shot):
    import torch
    torch.set_num_threads(2)
    index=read(ROOT/'configs/inputs.json')
    keys=[f'{shot}/{s}/{d}' for s in (42,43,44) for d in DATASETS]
    needed=set(index['metadata_sources'])
    expected={}
    for key in keys:
        c=index['cells'][key]
        needed.update(c['inputs'].values())
        needed.update([c['checkpoint'],c['predictions'],c['selection'],c['graph']])
        needed.add(index['cells'][f"10/{key.split('/')[1]}/{c['dataset']}"]['inputs']['split'])
        for b in c['branches'].values():
            needed.update(b.values())
        expected.update(c['expected_sha256'])
    files={}
    for number,p in enumerate(sorted(needed),1):
        path=Path(p); before=path.stat();md5=digest(path,'md5');sha=digest(path)
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),p
        if p in expected and expected[p]!=sha:raise ValueError(f'Historical SHA256 mismatch: {p}')
        files[p]=dict(path=p,resolved_path=str(path.resolve()),md5=md5,sha256=sha,
                      size=after.st_size,mtime_ns=after.st_mtime_ns)
        print('HASHED',number,len(needed),path.name,flush=True)
    write(run/'frozen-inputs.json',dict(files=files,file_count=len(files),md5_algorithm='md5',
         created_at=datetime.now(timezone.utc).isoformat(),source_root=index['source_root']))
    cfg=read(ROOT/'configs/experiment.json')
    cfg.update(shot=shot,cell_keys=keys,source=read(ROOT/'provenance/source.json'),code=source_state(),
               complete_cell_configs={k:index['cells'][k]['config'] for k in keys},
               execution=dict(mode='refit_router_and_joint',fixed_historical_calibration_and_acceptance=True,
                 repeat_hyperparameter_search=False,refit_frozen_experts=False),
               enhance_router_commit=command(['git','rev-parse','HEAD'])['stdout'])
    import joblib
    cfg['complete_router_configs']={key:{branch:router_recipe(joblib.load(inputs_path['bundle']))
            for branch,inputs_path in index['cells'][key]['branches'].items()} for key in keys}
    write(run/'config.json',cfg)
    write(run/'environment.json',environment())
    with (run/'frozen-files.md5').open('x') as f:
        for p,r in sorted(files.items()):f.write(f"{r['md5']}  {p}\n")
    inputs=Inputs(run)
    for key in keys:inputs.split(key)
    # Resolve numerical imports now so missing isolation dependencies fail before the array.
    from . import router, adapter
    from eargtc.joint_explore.joint_purf_v10_v1 import training
    for name in router.ENGINES:router.engine(name)
    assert_imports_isolated()
    write(run/'PREPARED.json',dict(all_checks_pass=True,cell_count=len(keys),
          source_code=source_state(),frozen_inputs_sha256=digest(run/'frozen-inputs.json'),
          ealr_gtc_commit_id=cfg['source']['commit_id']))
    print('PREPARED',run,flush=True)


def fit_cell(run,index):
    import random
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from . import adapter, router
    from eargtc.joint_explore.joint_purf_v10_v1 import training
    from eargtc.joint_cross_repair.artifacts import validate_hidden
    prepared=read(run/'PREPARED.json');cfg=read(run/'config.json')
    assert prepared['source_code']==source_state(),'Code changed since submission'
    key=cfg['cell_keys'][index];shot,seed,ds=key.split('/');seed=int(seed)
    out=run/'cells'/f'shot{shot}/seed{seed}/{ds}'
    if out.exists():raise FileExistsError(f'Use a new run; existing cell: {out}')
    out.mkdir(parents=True)
    inputs=Inputs(run);cell=inputs.index['cells'][key];split=inputs.split(key)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    started=time.monotonic()
    up=inputs.tensor(cell['inputs']['upstream'])
    with threadpool_limits(limits=1):
        rebuilt,models,diagnostics=router.rebuild(inputs,key,up,refit=True)
    torch.set_num_threads(2)
    f,s,pools=adapter.prepare_arrays(split,up,inputs.tensor(cell['inputs']['embeddings']),
                                    inputs.tensor(cell['inputs']['supervision']),rebuilt,ds)
    hidden=inputs.tensor(cell['inputs']['hidden']);validate_hidden(hidden,split['num_nodes'])
    assert hidden['metadata']['dataset']==ds;hidden=hidden['hidden']
    write(out/'config.json',dict(cell=key,training_seed=seed,joint_config=cell['config'],
          router_bundles=cell['branches'],router_config=cfg['complete_router_configs'][key],
          standard_eval_ids=inputs.index['standard_eval_ids'][f'{seed}/{ds}'],
          source=cfg['source'],code=cfg['code'],run_config_sha256=digest(run/'config.json'),
          environment_sha256=digest(run/'environment.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    reference=inputs.tensor(cell['checkpoint'])
    assert reference['config']==cell['config'] and reference['training_seed']==seed
    member=training.train(f,s,pools,hidden,cell['config'],seed)
    assert member['checkpoints']['accuracy']==reference['checkpoints']['accuracy'],'Joint validation selection differs'
    for k,v in member['states']['accuracy'].items():
        torch.testing.assert_close(v,reference['states']['accuracy'][k],atol=0,rtol=0)
    model=training.make_model(f,hidden,cell['config']);model.load_state_dict(member['states']['accuracy'])
    validation=training.predict(model,f,hidden,s['selection_ids'])
    torch.testing.assert_close(validation,reference['validation_probability'],atol=0,rtol=0)
    probability=training.predict(model,f,hidden,f['node_ids'])
    reference_predictions=inputs.tensor(cell['predictions'])
    torch.testing.assert_close(probability,reference_predictions['probability'],atol=0,rtol=0)
    assert_imports_isolated()
    save(out/'checkpoint.pt',member)
    save(out/'predictions.pt',dict(node_ids=f['node_ids'],probability=probability))
    save(out/'router.pt',rebuilt)
    import joblib
    with (out/'router_models.joblib').open('xb') as stream:joblib.dump(models,stream,compress=3)
    used={p:inputs.lock['files'][p] for p in sorted(inputs.checked)}
    write(out/'frozen-inputs.json',dict(files=used))
    write(out/'diagnostics.json',dict(cell=key,training_seed=seed,pools={k:len(v) for k,v in pools.items()},
          selected_checkpoint=member['checkpoints']['accuracy'],router=diagnostics,
          exact_joint_weights=True,exact_validation=True,exact_full_node_predictions=True,
          exact_router_decisions=True,router_score_atol=1e-10,query_truth_loaded=False,
          elapsed_seconds=time.monotonic()-started,slurm_job_id=os.environ['SLURM_JOB_ID'],
          imported_project_modules={k:str(Path(v.__file__).resolve()) for k,v in sys.modules.copy().items()
              if getattr(v,'__file__',None) and Path(v.__file__).resolve().is_relative_to(ROOT)}))
    write(out/'FROZEN.json',dict(cell=key,all_checks_pass=True,query_truth_loaded=False,
          files={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print('CELL PASSED',key,'seconds',time.monotonic()-started,flush=True)


def metrics(prob,y):
    import numpy as np
    pred=prob.argmax(1).numpy();target=y.numpy();classes=range(prob.shape[1])
    f1=[]
    for c in classes:
        tp=np.count_nonzero((pred==c)&(target==c));den=np.count_nonzero(pred==c)+np.count_nonzero(target==c)
        f1.append(2*tp/den if den else 0.)
    return dict(accuracy=float(np.mean(pred==target)),macro_f1=float(np.mean(f1)))


def evaluate(run):
    import torch
    torch.set_num_threads(2)
    cfg=read(run/'config.json');inputs=Inputs(run);out=run/'evaluation';out.mkdir()
    assert cfg['code']==source_state(),'Code changed since training'
    frozen={}
    for key in cfg['cell_keys']:
        shot,seed,ds=key.split('/');folder=run/'cells'/f'shot{shot}/seed{seed}/{ds}'
        receipt=read(folder/'FROZEN.json');assert receipt['all_checks_pass'] and not receipt['query_truth_loaded']
        for name,h in receipt['files'].items():assert digest(folder/name)==h
        frozen[key]=dict(path=str(folder/'FROZEN.json'),sha256=digest(folder/'FROZEN.json'))
    # This seal precedes every graph-label load in the entire run.
    write(run/'ALL_MODELS_FROZEN.json',dict(cells=frozen,all_checks_pass=True,query_truth_loaded=False,
          model_selection_closed=True,slurm_job_id=os.environ['SLURM_JOB_ID']))
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    source_rows={}
    with inputs.path(inputs.index['reference_metrics']).open() as f:
        for row in csv.DictReader(f):
            if row['variant']=='full' and int(row['shot'])==cfg['shot']:
                source_rows[f"{row['shot']}/{row['seed']}/{row['dataset']}"]=row
    labels={};rows=[]
    for key in cfg['cell_keys']:
        shot,seed,ds=key.split('/');cell=inputs.index['cells'][key]
        if ds not in labels:
            graph=torch.load(inputs.path(cell['graph']),map_location='cpu',weights_only=False)
            labels[ds]=graph.y.reshape(-1).long();del graph
        ids=torch.tensor(inputs.index['standard_eval_ids'][f'{seed}/{ds}'],dtype=torch.long)
        folder=run/'cells'/f'shot{shot}/seed{seed}/{ds}'
        probability=torch.load(folder/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)['probability']
        historical=inputs.tensor(cell['predictions'])['probability']
        torch.testing.assert_close(probability, historical,atol=0,rtol=0)
        result=metrics(probability[ids],labels[ds][ids]);reference=metrics(historical[ids],labels[ds][ids])
        assert result==reference
        reported=source_rows[key]
        for m in ('accuracy','macro_f1'):assert abs(result[m]-float(reported[m]))<1e-12,(key,m,result,reported)
        rows.append(dict(dataset=ds,shot=int(shot),seed=int(seed),population='standard_10shot_eval1000',
             eval_count=1000,**result,reference_accuracy=float(reported['accuracy']),
             reference_macro_f1=float(reported['macro_f1']),delta_accuracy=result['accuracy']-float(reported['accuracy']),
             delta_macro_f1=result['macro_f1']-float(reported['macro_f1']),predictions_exact=True))
    summary=[]
    for ds in DATASETS:
        group=[r for r in rows if r['dataset']==ds];assert sorted(r['seed'] for r in group)==[42,43,44]
        summary.append(dict(dataset=ds,shot=cfg['shot'],
            **{f'{m}_{s}':fn([r[m] for r in group]) for m in ('accuracy','macro_f1')
                for s,fn in [('mean',statistics.mean),('std_ddof0',statistics.pstdev)]}))
    for name,data in [('per_seed.csv',rows),('summary.csv',summary)]:
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Rehash all files used by this run to prove its frozen sources remain unchanged.
    changed=[]
    for path,record in inputs.lock['files'].items():
        if digest(path,'md5')!=record['md5']:changed.append(path)
    assert not changed,changed
    migrated=read(ROOT/'provenance/copied-sources.json')
    assert all(digest(p,'md5')==r['source_md5'] for p,r in migrated['files'].items())
    assert_imports_isolated()
    report=dict(all_checks_pass=True,cells=len(rows),shot=cfg['shot'],seeds=[42,43,44],
                all_predictions_exact=True,all_metrics_match_historical=True,source_dependencies_unchanged=True,
                source_ealr_gtc_commit_id=cfg['source']['commit_id'],enhancerouter_commit=cfg['enhance_router_commit'],
                evaluation_protocol='standard_10shot_eval1000',summary=summary,per_seed=rows,
                slurm_job_id=os.environ['SLURM_JOB_ID'],frozen_manifest_sha256=digest(run/'frozen-inputs.json'))
    write(out/'ACCEPTANCE.json',report)
    text=['EnhanceRouter acceptance: PASS','',f"5 datasets x 3 seeds; {cfg['shot']}-shot; standard_10shot_eval1000.",
          'Router heads refitted; Joint/PURF V1 trained from scratch with selected historical configs.',
          'Frozen expert caches, calibration and acceptance settings reused read-only.',
          'All Joint weights, validation and all-node probabilities match exactly.',
          'Accuracy and Macro-F1 match historical per-seed results.',
          'Frozen dependencies and copied source MD5 unchanged.','',
          'Dataset | Accuracy mean +/- population std (%) | Macro-F1 mean +/- population std (%)']
    for r in summary:
        text.append(f"{r['dataset']} | {100*r['accuracy_mean']:.2f} +/- {100*r['accuracy_std_ddof0']:.2f} | "
                    f"{100*r['macro_f1_mean']:.2f} +/- {100*r['macro_f1_std_ddof0']:.2f}")
    (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','cell','evaluate'))
    p.add_argument('--run',required=True);p.add_argument('--shot',type=int,choices=(3,5,10),default=5)
    p.add_argument('--index',type=int);a=p.parse_args()
    require_slurm();run=local(a.run);guard_writes()
    try:
        if a.stage=='prepare':prepare(run,a.shot)
        elif a.stage=='cell':fit_cell(run,a.index)
        else:evaluate(run)
    except BaseException as exc:
        path=run/f"FAILURE-{a.stage}-{os.environ['SLURM_JOB_ID']}.json"
        if not path.exists():write(path,dict(error_type=type(exc).__name__,error=str(exc),stage=a.stage,index=a.index))
        raise


if __name__=='__main__':main()
