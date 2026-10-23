"""Separate post-freeze truth evaluation and exact seed42–44 shot summary."""
import argparse
import csv
import fcntl
import itertools
import os
from pathlib import Path
import statistics
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from eargtc.joint_explore.joint_purf_v10_v1.protocol import read,write,sha,check,slurm
from eargtc.joint_explore.joint_purf_v10_v2.reporting import classification

REPO=Path(__file__).resolve().parents[2]
RUN=REPO/'runs/router_v10_shots_v1_20260914'
HISTORICAL=REPO/'runs/joint_purf_v10_v2_20260914'
SEEDS=(42,43,44)
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
POPS=(('full_query','evaluation_full_ids'),('fixed_evaluation','evaluation_test_ids'))
METRICS=('accuracy','macro_f1','pseudo_accuracy','pseudo_coverage')

def aggregate(rows):
    groups={}
    for row in rows:groups.setdefault((row['dataset'],row['shot'],row['population']),[]).append(row)
    out=[]
    for (ds,shot,pop),group in sorted(groups.items()):
        if sorted(r['seed'] for r in group)!=list(SEEDS):raise ValueError('Exactly seeds42,43,44 required')
        item=dict(dataset=ds,shot=shot,population=pop,seed_count=3)
        for metric in METRICS:
            values=[r[metric] for r in group if r[metric] is not None];item[metric+'_defined_seeds']=len(values)
            for suffix,fn in (('mean',statistics.mean),('std_ddof0',statistics.pstdev),('std_ddof1',statistics.stdev)):
                item[metric+'_'+suffix]=fn(values) if len(values)==3 else None
        out.append(item)
    return out

def historical_rows():
    complete=read(HISTORICAL/'COMPLETE.json')
    if not complete['all_checks_pass'] or complete['cell_count']!=25:raise ValueError('Historical experiment incomplete')
    hashes={str(HISTORICAL/name):complete['files'][name] for name in ('RESULTS.json','FINAL_AUDIT.json','REGISTRATION.json')};check(hashes)
    if not read(HISTORICAL/'FINAL_AUDIT.json')['all_checks_pass']:raise ValueError('Historical audit failed')
    hashes[str(HISTORICAL/'COMPLETE.json')]=sha(HISTORICAL/'COMPLETE.json');rows=[]
    for row in read(HISTORICAL/'RESULTS.json')['per_split']:
        if row['seed'] not in SEEDS:continue
        if row['training_seed']!=row['seed'] or row['inner_ensemble'] is not False:raise ValueError('Historical seed mismatch')
        rows.append(dict(dataset=row['dataset'],shot=5,population='full_query' if row['population']=='full_query' else 'fixed_evaluation',seed=row['seed'],
            accuracy=row['v2_accuracy'],macro_f1=row['v2_macro_f1'],pseudo_accuracy=row['pseudo_accuracy'],pseudo_coverage=row['pseudo_coverage'],
            query_count=row['query_count'],accepted_count=row['accepted_count'],source='historical_5shot',
            protocol='historical_query_observed_router; own_validation_joint_v1_then_v2'))
    if len(rows)!=30 or {(r['dataset'],r['seed'],r['population']) for r in rows}!={(d,s,p) for d,s,p in itertools.product(DATASETS,SEEDS,('full_query','fixed_evaluation'))}:raise ValueError('Historical dimensions differ')
    return rows,hashes

def verify_report(out, freeze_digests, final=None):
    """Verify a complete report, mapping staged hashes to their final paths."""
    final=out if final is None else final
    audit=read(out/'AUDIT.json')
    if (audit.get('all_checks_pass') is not True or audit.get('rows')!=90
            or audit.get('groups')!=30 or audit.get('seeds')!=list(SEEDS)):
        raise ValueError('Invalid completed report audit')
    if any(audit.get(k)!=v for k,v in freeze_digests.items()):raise ValueError('Completed report freeze differs')
    required={'RESULTS.json','REPORT.md','per_split.csv','mean_std.csv'}
    expected={str(final/name) for name in required}
    if set(audit.get('files',{}))!=expected:raise ValueError('Report artifact manifest differs')
    if {p.name for p in out.iterdir()}!=required|{'AUDIT.json'}:raise ValueError('Unexpected report artifacts')
    check({str(out/Path(path).name):digest for path,digest in audit['files'].items()})
    check(audit['historical_hashes'])
    result=read(out/'RESULTS.json')
    if len(result['per_split'])!=90 or len(result['mean_std'])!=30:raise ValueError('Report dimensions differ')
    return audit


def evaluate(root=RUN):
    slurm();torch.set_num_threads(2);root=Path(root).resolve()
    # The lock serializes publication and protects an existing completed report.
    with (root/'.classification_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        frozen=read(root/'DOWNSTREAM_FROZEN.json');router=read(root/'ROUTER_FROZEN.json')
        if any(x.get('all_checks_pass') is not True or x.get('cell_count')!=30 for x in (frozen,router)):raise ValueError('Both all30 freezes required')
        check(frozen['files']);check(router['files']);reg=read(root/'DOWNSTREAM_REGISTRATION.json');check(reg['code_hashes'])
        digests=dict(downstream_frozen_sha256=sha(root/'DOWNSTREAM_FROZEN.json'),router_frozen_sha256=sha(root/'ROUTER_FROZEN.json'))
        started=root/'EVALUATION_STARTED.json'
        if started.exists():
            receipt=read(started)
            if any(receipt.get(k)!=v for k,v in digests.items()):raise ValueError('Evaluation start freeze differs')
        else:write(started,dict(digests,slurm_job_id=os.environ['SLURM_JOB_ID']))
        final=root/'classification_report'
        if final.exists():
            verify_report(final,digests)
            print('CLASSIFICATION REPORT VERIFIED',final,flush=True)
            return
        attempts=root/'classification_report_attempts';attempts.mkdir(exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix=f"job{os.environ['SLURM_JOB_ID']}_",dir=attempts))
        try:
            _evaluate_attempt(root,out,frozen,router,reg)
            verify_report(out,digests,final)
            # Same-filesystem rename exposes the complete report at once.
            out.rename(final)
        except BaseException as error:
            # Keep partial artifacts and failure evidence for an independent retry.
            write(out/'FAILURE.json',dict(error_type=type(error).__name__,error=str(error),
                slurm_job_id=os.environ['SLURM_JOB_ID'],**digests))
            raise
        print('CLASSIFICATION REPORT COMPLETE',final,flush=True)


def _evaluate_attempt(root,out,frozen,router,reg):
    report_source_sha256=sha(Path(__file__).resolve())
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    rows,hashes=historical_rows();historical_reg=read(HISTORICAL/'REGISTRATION.json');graph_hashes={};fixed_population_checks=[]
    for ds in DATASETS:
        first=read(reg['inputs'][f'3/42/{ds}']['paths']['split']);check({first['graph_path']:first['graph_sha256']})
        graph=torch.load(first['graph_path'],weights_only=False,map_location='cpu');truth=graph.y.reshape(-1).numpy().astype(np.int64)
        graph_hashes[first['graph_path']]=first['graph_sha256'];del graph
        for shot,seed in itertools.product((3,10),SEEDS):
            record=reg['inputs'][f'{shot}/{seed}/{ds}'];check(record['hashes']);split=read(record['paths']['split'])
            if split['graph_sha256']!=first['graph_sha256']:raise ValueError('Graph differs across shots/seeds')
            old=read(historical_reg['inputs'][f'{seed}/{ds}']['paths']['split'])
            fixed_population_checks.append(dict(dataset=ds,shot=shot,seed=seed,
                identical_historical_fixed_ids=split["evaluation_test_ids"]==old["evaluation_1000_ids"],
                own_count=len(split["evaluation_test_ids"]),historical_count=len(old["evaluation_1000_ids"]),
                intersection_count=len(set(split["evaluation_test_ids"]) & set(old["evaluation_1000_ids"]))))
            dest=root/f'shot{shot}/seed{seed}/downstream/{ds}/final'
            pred=torch.load(dest/'predictions.pt',weights_only=True,map_location='cpu')
            if pred['training_seed']!=seed or pred['shots']!=shot or pred['inner_ensemble'] is not False:raise ValueError('Classifier prediction identity differs')
            probability=pred['probability'].numpy();routing=torch.load(record['paths']['router'],weights_only=True,map_location='cpu')
            for pop,key in POPS:
                ids=np.asarray(split[key],dtype=np.int64);accepted=ids[routing['accepted'].numpy()[ids]]
                values=classification(probability[ids],truth[ids])
                # Independent reference includes all classes, including absent prediction/truth classes.
                from sklearn.metrics import accuracy_score,f1_score
                observed=probability[ids].argmax(1)
                if abs(values['accuracy']-accuracy_score(truth[ids],observed))>1e-12 or abs(values['macro_f1']-f1_score(truth[ids],observed,labels=np.arange(probability.shape[1]),average='macro',zero_division=0))>1e-12:raise ValueError('Independent metrics disagree')
                rows.append(dict(dataset=ds,shot=shot,population=pop,seed=seed,**values,
                    pseudo_accuracy=float(np.mean(routing['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None,
                    pseudo_coverage=len(accepted)/len(ids),query_count=len(ids),accepted_count=len(accepted),source='fresh_3_10shot',
                    protocol='validation_only_router; fresh_own_validation_joint_v1_then_v2'))
            print('POSTFREEZE EVALUATED',shot,seed,ds,flush=True)
    expected={(ds,shot,pop,seed) for ds,shot,pop,seed in itertools.product(DATASETS,(3,5,10),('full_query','fixed_evaluation'),SEEDS)}
    if len(rows)!=90 or {(r['dataset'],r['shot'],r['population'],r['seed']) for r in rows}!=expected:raise ValueError('Expected exact90 result rows')
    summary=aggregate(rows)
    for item in summary:
        group=[r for r in rows if (r['dataset'],r['shot'],r['population'])==(item['dataset'],item['shot'],item['population'])]
        for m in METRICS:
            vals=[r[m] for r in group]
            if all(v is not None for v in vals):
                for suffix,fn in (('mean',np.mean),('std_ddof0',lambda x:np.std(x,ddof=0)),('std_ddof1',lambda x:np.std(x,ddof=1))):
                    if abs(item[m+'_'+suffix]-float(fn(vals)))>1e-12:raise ValueError('Independent aggregates disagree')
    write(out/'RESULTS.json',dict(per_split=rows,mean_std=summary,historical_hashes=hashes,graph_hashes=graph_hashes,
                                fixed_population_checks=fixed_population_checks,seeds=list(SEEDS),classifier='Joint-PURF-v2'))
    for name,data in (('per_split.csv',rows),('mean_std.csv',summary)):
        with (out/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['# Router-v10-MS-Original + Joint-PURF-v2：3/5/10-shot','',
           'seed42、43、44；百分数均值 ± 总体标准差(ddof=0)，CSV另存样本标准差(ddof=1)。',
           '3/10-shot：冻结GNN-as-Judge划分，专家和Router重新拟合；Router仅验证集选型；分类器重新执行V1与V2候选搜索。',
           '5-shot：单独归因于既有封存结果，历史Router曾观察query；本表不能解释为相同盲测协议下shot数的净效应。',
           '分类训练使用同shot同seed support和被Router接受的伪标签；验证选择保留历史确定性分半，不跨seed集成。',
           'fixed_evaluation按各shot自身冻结的评估节点计算；5-shot沿用历史eval1000，跨shot可能不是同一节点集合。full_query亦随shot改变，差异见RESULTS.json。',
           '原始数据、专家缓存和LLM hidden为外部哈希绑定依赖；该目录不是独立数据包。','']
    def ms(r,m):return 'NA' if r[m+'_mean'] is None else f"{100*r[m+'_mean']:.2f} ± {100*r[m+'_std_ddof0']:.2f}"
    for pop,_ in POPS:
        lines += [f'## {pop}','', '| Dataset | 3-shot Accuracy | 5-shot Accuracy | 10-shot Accuracy | 3-shot Macro-F1 | 5-shot Macro-F1 | 10-shot Macro-F1 |','|---|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            rr={r['shot']:r for r in summary if r['dataset']==ds and r['population']==pop}
            lines.append('| '+ds+' | '+' | '.join(ms(rr[shot],m) for m in ('accuracy','macro_f1') for shot in (3,5,10))+' |')
        lines += ['', '| Dataset | Shot | Router pseudo precision | Router coverage |','|---|---:|---:|---:|']
        for ds,shot in itertools.product(DATASETS,(3,5,10)):
            r=next(r for r in summary if r['dataset']==ds and r['shot']==shot and r['population']==pop)
            lines.append(f"| {ds} | {shot} | {ms(r,'pseudo_accuracy')} | {ms(r,'pseudo_coverage')} |")
        lines.append('')
    with (out/'REPORT.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
    check(hashes);check(frozen['files']);check(router['files'])
    write(out/'AUDIT.json',dict(all_checks_pass=True,rows=90,groups=30,seeds=list(SEEDS),independent_sklearn_numpy_verified=True,
        prescribed_fixed_populations_preserved=True, historical_population_differences_recorded=True,all30_frozen_before_truth=True,slurm_job_id=os.environ['SLURM_JOB_ID'],
        report_source_sha256=report_source_sha256,historical_hashes=hashes,files={str(root/'classification_report'/p.name):sha(p) for p in out.iterdir() if p.is_file()},
        downstream_frozen_sha256=sha(root/'DOWNSTREAM_FROZEN.json'),router_frozen_sha256=sha(root/'ROUTER_FROZEN.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RUN);evaluate(p.parse_args().root)
