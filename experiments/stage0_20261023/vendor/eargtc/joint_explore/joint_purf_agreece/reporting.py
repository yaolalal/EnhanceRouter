"""Separate truth process after all 45 selected models are globally frozen."""
import csv
import fcntl
import itertools
import os
from pathlib import Path
import statistics
import tempfile
import numpy as np
import torch
from .protocol import RUN, REPO, CELLS, DATASETS, SHOTS, SEEDS, key, cell, read, write, sha, check, verify_code, load
from .training import make_model, predict
from eargtc.joint_explore.joint_purf_v10_v2.reporting import classification

METRICS = ('accuracy','macro_f1','baseline_accuracy','baseline_macro_f1','control_accuracy','control_macro_f1',
           'delta_accuracy','delta_macro_f1','retune_delta_accuracy','retune_delta_macro_f1','pseudo_accuracy','pseudo_coverage')


def aggregate(rows):
    groups={}
    for r in rows:groups.setdefault((r['dataset'],r['shot'],r['population']),[]).append(r)
    out=[]
    for (ds,shot,pop),group in sorted(groups.items()):
        if sorted(r['seed'] for r in group) != list(SEEDS):raise ValueError('Exactly seeds 42,43,44 required')
        row=dict(dataset=ds,shot=shot,population=pop,seed_count=3)
        for m in METRICS:
            v=[r[m] for r in group]
            if any(x is None for x in v):
                row.update({m+s:None for s in ('_mean','_std_ddof0','_std_ddof1')});continue
            for suffix,fn,npfn in (('_mean',statistics.mean,np.mean),('_std_ddof0',statistics.pstdev,lambda x:np.std(x,ddof=0)),('_std_ddof1',statistics.stdev,lambda x:np.std(x,ddof=1))):
                row[m+suffix]=fn(v)
                if abs(row[m+suffix]-float(npfn(v)))>1e-12:raise ValueError('Independent aggregation failed')
        out.append(row)
    return out


def checked_classification(prob,truth):
    from sklearn.metrics import accuracy_score,f1_score
    value=classification(prob,truth);pred=prob.argmax(1)
    if abs(value['accuracy']-accuracy_score(truth,pred))>1e-12 or abs(value['macro_f1']-f1_score(truth,pred,labels=np.arange(prob.shape[1]),average='macro',zero_division=0))>1e-12:
        raise ValueError('Independent sklearn metrics failed')
    return value


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_report(path):
    audit=read(path/'AUDIT.json')
    if not audit['all_checks_pass'] or audit['rows']!=90 or audit['groups']!=30 or audit['cells']!=45:
        raise ValueError('Report dimension audit failed')
    if audit['frozen_sha256']!=sha(RUN/'FROZEN.json'):raise ValueError('Report freeze changed')
    check({str(path/name):h for name,h in audit['files'].items()})
    return audit


def evaluate():
    reg=verify_code()
    with (RUN/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        frozen=read(RUN/'FROZEN.json')
        if not frozen['all_checks_pass'] or frozen['cell_count']!=45 or frozen['test_truth_loaded'] is not False:
            raise ValueError('All45 freeze required before any query truth')
        check(frozen['files']);check(reg['protected']);check(reg['inputs_hashes'])
        write(RUN/'EVALUATION_STARTED.json',dict(frozen_sha256=sha(RUN/'FROZEN.json')))
        final=RUN/'classification_report'
        if final.exists():
            verify_report(final)
        else:
            out=Path(tempfile.mkdtemp(prefix='.report_',dir=RUN))
            _attempt(out,reg,frozen)
            verify_report(out);out.rename(final)
        complete=dict(all_checks_pass=True,cell_count=45,seeds=list(SEEDS),shots=list(SHOTS),
                      report_audit_sha256=sha(final/'AUDIT.json'),frozen_sha256=sha(RUN/'FROZEN.json'),
                      files={str(p.relative_to(RUN)):sha(p) for p in final.iterdir() if p.is_file()})
        write(RUN/'COMPLETE.json',complete)
        print('FINAL REPORT COMPLETE',final,flush=True)


def _attempt(out,reg,frozen):
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    torch.set_num_threads(2)
    rows=[];configs=[];coverage=[];stages=[];graph_hashes={};populations=[]
    for ds in DATASETS:
        first=read(reg['inputs'][key(3,42,ds)]['paths']['split'])
        check({first['graph_path']:first['graph_sha256']})
        graph=torch.load(first['graph_path'],weights_only=False,map_location='cpu')
        truth=graph.y.reshape(-1).numpy().astype(np.int64);del graph
        graph_hashes[first['graph_path']]=first['graph_sha256']
        for shot,seed in itertools.product(SHOTS,SEEDS):
            f,s,p,h,split,router=load(shot,seed,ds)
            if split['graph_sha256']!=first['graph_sha256']:raise ValueError('Different truth identity')
            dest=cell(shot,seed,ds);selection=read(dest/'final/selection.json')
            pred=torch.load(dest/'final/predictions.pt',weights_only=True,map_location='cpu')
            old=torch.load(reg['inputs'][key(shot,seed,ds)]['baseline']['predictions.pt'],weights_only=True,map_location='cpu')
            if old['training_seed']!=seed or old['inner_ensemble'] is not False:raise ValueError('Baseline seed mismatch')
            cp=torch.load(dest/'candidates'/selection['control_candidate_id']/'checkpoint.pt',weights_only=True,map_location='cpu')
            model=make_model(f,h,cp['config']);model.load_state_dict(cp['states']['accuracy'])
            control=predict(model,f,h,f['node_ids']).numpy()
            probability=pred['probability'].numpy();baseline=old['probability'].numpy()
            if probability.shape!=baseline.shape or probability.shape!=control.shape:raise ValueError('Comparison populations differ')
            configs.append(dict(dataset=ds,shot=shot,seed=seed,winner=selection['winner'],control_config=cp['config'],
                                class_weights=selection['class_weights'],training_class_counts=selection['training_class_counts']))
            for when,pools in (('selected',selection['selected_coverage']),('training_end',selection['winner']['coverage'])):
                for name,v in pools.items():coverage.append(dict(dataset=ds,shot=shot,seed=seed,when=when,pool=name,**v))
            for stage in range(1,10):
                st=read(dest/f'stage{stage}_complete.json');stages.append(dict(dataset=ds,shot=shot,seed=seed,**st))
            five=read(reg['inputs'][key(5,seed,ds)]['paths']['split'])
            populations.append(dict(dataset=ds,shot=shot,seed=seed,fixed_ids_equal_to_5shot=split['evaluation_1000_ids']==five['evaluation_1000_ids'],
                                    fixed_intersection_with_5shot=len(set(split['evaluation_1000_ids'])&set(five['evaluation_1000_ids']))))
            for pop,idkey in (('full_query','evaluation_full_ids'),('fixed_evaluation','evaluation_1000_ids')):
                ids=np.asarray(split[idkey],dtype=np.int64)
                values=checked_classification(probability[ids],truth[ids]);oldvalues=checked_classification(baseline[ids],truth[ids]);cv=checked_classification(control[ids],truth[ids])
                accepted=ids[router['accepted'].numpy()[ids]]
                row=dict(dataset=ds,shot=shot,seed=seed,population=pop,**values,
                         baseline_accuracy=oldvalues['accuracy'],baseline_macro_f1=oldvalues['macro_f1'],control_accuracy=cv['accuracy'],control_macro_f1=cv['macro_f1'],
                         delta_accuracy=values['accuracy']-oldvalues['accuracy'],delta_macro_f1=values['macro_f1']-oldvalues['macro_f1'],
                         retune_delta_accuracy=values['accuracy']-cv['accuracy'],retune_delta_macro_f1=values['macro_f1']-cv['macro_f1'],
                         query_count=len(ids),accepted_count=len(accepted),
                         pseudo_accuracy=float(np.mean(router['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None,
                         pseudo_coverage=len(accepted)/len(ids),candidate_id=selection['winner']['id'],selected_step=selection['winner']['step'])
                rows.append(row)
            print('EVALUATED',shot,seed,ds,flush=True)
    expected=set(itertools.product(DATASETS,SHOTS,SEEDS,('full_query','fixed_evaluation')))
    if len(rows)!=90 or {(r['dataset'],r['shot'],r['seed'],r['population']) for r in rows}!=expected:raise ValueError('Result cells differ')
    summary=aggregate(rows)
    write(out/'RESULTS.json',dict(per_split=rows,mean_std=summary,graph_hashes=graph_hashes,population_comparisons=populations))
    write(out/'BEST_CONFIGS.json',configs);write(out/'SEARCH_STAGES.json',stages)
    csv_write(out/'per_seed.csv',rows);csv_write(out/'mean_std.csv',summary);csv_write(out/'coverage.csv',coverage)
    lines=['# Joint-PURF-AgreeCE + Router-v10-MS-Original','',
           'LLaMA3-8B；五个历史数据集，3/5/10-shot，seed 42–44。所有45个Joint均重新训练和验证调优。',
           '唯一监督修改：agreement KL删除，接受的一致节点参与硬标签CE及可选repair伪标签辅助CE。',
           '所有历史划分、Router决定、专家概率与表示不变；各池有放回采样64次，不要求全覆盖。',
           '选模仅使用各自历史验证calibration子集，Accuracy优先、NLL次之；同一训练平分取较早step。所有模型冻结后才评估query。',
           '均值±总体标准差(ddof=0)，CSV另存样本标准差(ddof=1)。Macro-F1包含全部类别，缺失类别F1计0。',
           '历史5-shot Router曾参考query结果；跨shot评估节点可能不同，因此本表不是相同盲测协议下shot数的净效应。',
           '原始数据、专家与hidden仍为外部哈希绑定依赖，此结果目录不是独立数据包。','']
    def ms(r,m):return f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_std_ddof0']:.2f}"
    for pop in ('full_query','fixed_evaluation'):
        lines += [f'## {pop}','', '| Dataset | 3-shot Acc | 5-shot Acc | 10-shot Acc | 3-shot Macro-F1 | 5-shot Macro-F1 | 10-shot Macro-F1 |','|---|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            rr={r['shot']:r for r in summary if r['dataset']==ds and r['population']==pop}
            lines.append('| '+ds+' | '+' | '.join(ms(rr[shot],m) for m in ('accuracy','macro_f1') for shot in SHOTS)+' |')
        lines += ['', '| Dataset | Shot | Acc Δ vs original (pp) | F1 Δ vs original (pp) | Acc Δ vs same-config CE (pp) |','|---|---:|---:|---:|---:|']
        for ds,shot in itertools.product(DATASETS,SHOTS):
            r=next(r for r in summary if (r['dataset'],r['shot'],r['population'])==(ds,shot,pop))
            lines.append(f"| {ds} | {shot} | {ms(r,'delta_accuracy')} | {ms(r,'delta_macro_f1')} | {ms(r,'retune_delta_accuracy')} |")
        lines.append('')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    search_lines=['# Search and sampling report','',
                  'One model family, nine finite stages: core81, fusion/aux strategies, local refinement, removals, schedule, fusion horizon, independent auxiliaries, class weighting/removals, final coordinate refinement.',
                  'All candidates use positive agreement CE and no KL. Each fit starts from the split seed; no checkpoint transfer or inner ensembles.',
                  'The same-old-configuration CE control is trained fresh and is also an eligible candidate. Test deltas are descriptive only.',
                  'Stage selections and negative/non-improving trials are retained. Sampling coverage is recorded at the selected checkpoint and training end.','',
                  '| Dataset | Shot | Seed | Candidates | Selected step | Agreement coverage at selection |','|---|---:|---:|---:|---:|---:|']
    for shot,seed,ds in CELLS:
        sel=read(cell(shot,seed,ds)/'final/selection.json');v=sel['selected_coverage']['agreement']['fraction']
        frac='NA' if v is None else f'{100*v:.2f}%'
        search_lines.append(f"| {ds} | {shot} | {seed} | {len(sel['ranking'])} | {sel['winner']['step']} | {frac} |")
    (out/'SEARCH_REPORT.md').write_text('\n'.join(search_lines)+'\n')
    check(reg['protected']);check(reg['inputs_hashes']);check(frozen['files'])
    write(out/'AUDIT.json',dict(all_checks_pass=True,rows=90,groups=30,cells=45,seeds=list(SEEDS),shots=list(SHOTS),
          independent_sklearn_numpy_verified=True,all45_frozen_before_truth=True,historical_input_hashes_preserved=True,
          frozen_sha256=sha(RUN/'FROZEN.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},slurm_job_id=os.environ['SLURM_JOB_ID']))
