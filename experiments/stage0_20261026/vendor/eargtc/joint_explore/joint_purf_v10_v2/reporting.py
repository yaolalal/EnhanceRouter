"""Post-freeze paired metrics, descriptive search effects, and independent audit."""
import csv,itertools,os,shutil,statistics,sys
from pathlib import Path
import numpy as np
import torch
from .protocol import RUN,REPO,PREVIOUS,SEEDS,DATASETS,CELLS,cell,read,write,sha,check,verify_all,merge_records,load,slurm
from .search import cid,candidate_path,rank,prior_configs,checked_stage
from .training import make_model,predict
MODELS=('fixed','v1','v2','historical')
METRICS=tuple(f'{m}_{k}' for m in MODELS for k in ('accuracy','macro_f1'))+tuple(f'delta_{b}_{k}' for b in ('v1','fixed','historical') for k in ('accuracy','macro_f1'))+('pseudo_accuracy','pseudo_coverage')
POPULATIONS=(('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids'))


def verify_frozen():
    slurm();f=read(RUN/'FROZEN.json')
    if not f['all_checks_pass'] or f['cell_count']!=25 or f['inner_seeds']!=[] or f['cross_split_selection'] is not False or f['test_truth_loaded'] is not False:raise ValueError('Invalid freeze')
    reg=verify_all();check(f['files']);return reg,f

def classification(prob,truth):
    pred=prob.argmax(1);c=prob.shape[1]
    mat=np.bincount(truth*c+pred,minlength=c*c).reshape(c,c);tp=mat.diagonal().astype(float);den=mat.sum(0)+mat.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(c),where=den>0)
    return dict(accuracy=float(np.mean(pred==truth)),macro_f1=float(f1.mean()))

def add_deltas(row):
    for baseline in ("v1","fixed","historical"):
        for metric in ("accuracy","macro_f1"):row[f"delta_{baseline}_{metric}"]=row[f"v2_{metric}"]-row[f"{baseline}_{metric}"]
    return row

def evaluation_artifacts():
    summaries={"RESULTS.json","BEST_CONFIGS.json","CLASS_DIAGNOSTICS.json"}
    return [p for p in RUN.iterdir() if p.is_file() and (p.suffix in (".csv",".md") or p.name in summaries)]

def aggregate(rows,metrics=METRICS):
    grouped={}
    for r in rows:grouped.setdefault((r['dataset'],r['population']),[]).append(r)
    result=[]
    for (ds,pop),g in grouped.items():
        if sorted(r['seed'] for r in g)!=list(SEEDS):raise ValueError('Exactly five distinct historical seeds required')
        row=dict(dataset=ds,population=pop,seed_count=5)
        for m in metrics:
            vals=[r[m] for r in g if r[m] is not None];row[m+'_defined_seeds']=len(vals)
            for suffix,fn in (('_mean',statistics.mean),('_std_ddof0',statistics.pstdev),('_std_ddof1',statistics.stdev)):
                row[m+suffix]=fn(vals) if len(vals)==5 else None
        result.append(row)
    return result

def csv_write(name,rows):
    if not rows:raise ValueError('Refusing empty report '+name)
    with (RUN/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pct(v):return 'NA' if v is None else f'{100*v:.2f}'
def ms(r,k):return 'NA' if r[k+'_mean'] is None else f"{pct(r[k+'_mean'])} ± {pct(r[k+'_std_ddof0'])}"

def probabilities(reg,seed,ds):
    record=reg['inputs'][f'{seed}/{ds}']
    paths=dict(fixed=record['fixed_predictions'],v1=record['baseline_predictions'],v2=str(cell(seed,ds)/'final/predictions.pt'),
               historical=str(REPO/f'runs/joint_unified_v1_20260913/seed{seed}/{ds}/fixed/predictions.pt'))
    out={}
    for name,path in paths.items():
        obj=torch.load(path,weights_only=True,map_location='cpu')
        if obj['training_seed']!=seed or obj['inner_ensemble'] is not False:raise ValueError('Prediction seed protocol differs')
        out[name]=obj['fixed' if name in ('fixed','historical') else 'probability']
    return out

def graph_truth(reg,ds):
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    split=read(reg['inputs'][f'42/{ds}']['paths']['split']);check({split['graph_path']:split['graph_sha256']})
    graph=torch.load(split['graph_path'],map_location='cpu',weights_only=False)
    return graph.y.reshape(-1).numpy().astype(np.int64),split['graph_sha256']

def evaluate():
    reg,frozen=verify_frozen();torch.set_num_threads(2)
    write(RUN/'EVALUATION_STARTED.json',dict(frozen_sha256=sha(RUN/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    oldrows=read(PREVIOUS/'RESULTS.json')['per_split']
    oldindex={(r['dataset'],r['seed'],r['population']):r for r in oldrows if r['variant']=='original'};rows=[]
    for ds in DATASETS:
        truth,graphhash=graph_truth(reg,ds)
        for seed in SEEDS:
            record=reg['inputs'][f'{seed}/{ds}'];split=read(record['paths']['split'])
            if split['graph_sha256']!=graphhash:raise ValueError('Graph identity differs across splits')
            probs=probabilities(reg,seed,ds);router=torch.load(record['paths']['router'],weights_only=True,map_location='cpu')
            for pop,idkey in POPULATIONS:
                ids=np.asarray(split[idkey],dtype=np.int64);accepted=ids[router['accepted'].numpy()[ids]]
                row=dict(dataset=ds,population=pop,seed=seed,training_seed=seed,inner_ensemble=False,query_count=len(ids),accepted_count=len(accepted))
                for m,prob in probs.items():row.update({m+'_'+k:v for k,v in classification(prob.numpy()[ids],truth[ids]).items()})
                add_deltas(row)
                row['pseudo_accuracy']=float(np.mean(router['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None
                row['pseudo_coverage']=len(accepted)/len(ids);old=oldindex[ds,seed,pop]
                for new,prev in [('v1','purf'),('fixed','fixed_matched'),('historical','fixed_reference')]:
                    for k in ('accuracy','macro_f1'):
                        if abs(row[new+'_'+k]-old[prev+'_'+k])>1e-12:raise ValueError('Sealed baseline metrics changed')
                if any(row[k]!=old[k] for k in ('pseudo_accuracy','pseudo_coverage','accepted_count','query_count')):raise ValueError('Frozen pseudo quality changed')
                rows.append(row)
        print('EVALUATED',ds,flush=True)
    summary=aggregate(rows)
    write(RUN/'RESULTS.json',dict(per_split=rows,mean_std=summary,frozen_sha256=sha(RUN/'FROZEN.json')))
    csv_write('per_split.csv',rows);csv_write('mean_std.csv',summary)
    for pop,_ in POPULATIONS:
        csv_write(pop+'_per_split.csv',[r for r in rows if r['population']==pop]);csv_write(pop+'_mean_std.csv',[r for r in summary if r['population']==pop])
    report(rows,summary);search_report();exploration_report(rows,summary)
    artifacts=evaluation_artifacts()
    write(RUN/'EVALUATION_COMPLETE.json',dict(all_checks_pass=True,rows=50,groups=10,files={str(p):sha(p) for p in artifacts},
        frozen_sha256=sha(RUN/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('EVALUATION AND REPORTS COMPLETE',flush=True)

def report(rows,summary):
    lines=['# Router-v10-MS-Original + Joint-PURF V2 五数据集调优','',
        '所有指标为百分数；均值±总体标准差(ddof=0)，CSV另存样本标准差(ddof=1)。增益单位为百分点。',
        '固定上一轮Original伪标签内容、接受集合、路由与专家输出；seed42–46等于各自历史split seed，inner_seeds=[]，不集成。',
        '主表Fixed为同一个Router-v10下上一轮已封存的Joint-Fixed；V1为上一轮选中的PURF，V2为本轮验证集选中的模型。三者使用相同Router伪标签。历史Router-v9-new Fixed另列附表。',
        '每个seed/dataset独立四阶段搜索；上一轮V1 checkpoint始终保留为候选。以本划分validation Accuracy优先、NLL次之，完全相同则优先继承V1。没有跨seed或query选型。',
        '全部25组模型与配置全局冻结后才运行本轮query评估。上游Router及既有实验存在历史query观测，本轮属于已观察历史基准上的进一步探索，不能声称未观察测试集盲测。',
        '模型仍为独立降维后拼接、一个repair、独立正值alpha/beta融合、无pB；固定gate仅为对照候选。新增损失权重不增删或改写伪标签。',
        'V2搜索预算大于基线，均值变化不等同统计显著，不把多个策略交互归因于单项改动。原始数据/专家/hidden为外部哈希绑定依赖。','']
    for pop,_ in POPULATIONS:
        lines += [f'## {pop}：五种子主表','',
            '| Dataset | Fixed Acc | PURF V1 Acc | PURF V2 Acc | ΔV2−V1 Acc | Fixed F1 | PURF V1 F1 | PURF V2 F1 | ΔV2−V1 F1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            r=next(r for r in summary if r['dataset']==ds and r['population']==pop)
            cols=[ms(r,k) for k in ('fixed_accuracy','v1_accuracy','v2_accuracy','delta_v1_accuracy','fixed_macro_f1','v1_macro_f1','v2_macro_f1','delta_v1_macro_f1')]
            lines.append('| '+ds+' | '+' | '.join(cols)+' |')
        lines += ['',f'## {pop}：固定伪标签质量（三个模型完全相同）','',
            '| Dataset | Pseudo Accuracy | Coverage |','|---|---:|---:|']
        for ds in DATASETS:
            r=next(r for r in summary if r['dataset']==ds and r['population']==pop)
            lines.append(f"| {ds} | {ms(r,'pseudo_accuracy')} | {ms(r,'pseudo_coverage')} |")
        lines += ['',f'## {pop}：逐 seed 配对结果','',
            '| Dataset | Seed | Fixed Acc | V1 Acc | V2 Acc | ΔV1 Acc | ΔFixed Acc | Fixed F1 | V1 F1 | V2 F1 | ΔV1 F1 | ΔFixed F1 | Pseudo Acc | Coverage |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['population']!=pop:continue
            cols=[pct(r[k]) for k in ('fixed_accuracy','v1_accuracy','v2_accuracy','delta_v1_accuracy','delta_fixed_accuracy','fixed_macro_f1','v1_macro_f1','v2_macro_f1','delta_v1_macro_f1','delta_fixed_macro_f1','pseudo_accuracy','pseudo_coverage')]
            lines.append(f"| {r['dataset']} | {r['seed']} | "+' | '.join(cols)+' |')
        lines += ['',f'## {pop}：历史Router-v9-new Fixed参考','',
            '| Dataset | Historical Fixed Acc | V2 Acc | ΔAcc | Historical Fixed F1 | V2 F1 | ΔF1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            r=next(r for r in summary if r['dataset']==ds and r['population']==pop)
            lines.append('| '+ds+' | '+' | '.join(ms(r,k) for k in ('historical_accuracy','v2_accuracy','delta_historical_accuracy','historical_macro_f1','v2_macro_f1','delta_historical_macro_f1'))+' |')
        lines.append('')
    lines += ['选中配置与所有候选见 [SEARCH_REPORT.md](SEARCH_REPORT.md)；阶段收益、单项与交互改动见 [EXPLORATION_GAINS.md](EXPLORATION_GAINS.md)；类别诊断见 [CLASS_DIAGNOSTICS.md](CLASS_DIAGNOSTICS.md)。',
        '复现入口见README.md；FINAL_AUDIT.json和COMPLETE.json记录最终独立复核及封存状态。']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')

def search_report():
    stages=[checked_stage(i) for i in range(1,5)];selected=read(RUN/'SELECTION.json');trials=[];effects=[];stage_rows=[];configs=[];diagnostics=[]
    for seed,ds in CELLS:
        base=read(cell(seed,ds)/'BASELINE.json')['config'];br=read(candidate_path(seed,ds,base)/'result.json');prior=br['selected']['accuracy']
        for row in selected['cells'][f'{seed}/{ds}']['ranking']:
            r=read(candidate_path(seed,ds,row['config'])/'result.json')
            trials.append(dict(dataset=ds,seed=seed,candidate_id=row['id'],origin=r['origin'],validation_accuracy=row['accuracy'],validation_nll=row['nll'],step=row['step'],train_seconds=r['train_seconds'],config=json.dumps(row['config'],sort_keys=True)))
        for st in stages:
            anchor=st['selection_evidence'][f'{seed}/{ds}'][0]
            cumulative=rank(seed,ds,prior_configs(seed,ds,stages[:st['stage']]))[0]
            stage_rows.append(dict(dataset=ds,seed=seed,stage=st['stage'],accuracy=cumulative['accuracy'],delta_previous=cumulative['accuracy']-prior,delta_v1=cumulative['accuracy']-br['selected']['accuracy']))
            prior=cumulative['accuracy']
            for cfg in st['candidates'][f'{seed}/{ds}']:
                rr=read(candidate_path(seed,ds,cfg)/'result.json');changed={k:cfg[k] for k in cfg if cfg[k]!=anchor['config'][k]}
                effects.append(dict(dataset=ds,seed=seed,stage=st['stage'],anchor_id=anchor['id'],candidate_id=cid(cfg),changed_count=len(changed),changes=json.dumps(changed,sort_keys=True),delta_accuracy=rr['selected']['accuracy']-anchor['accuracy'],delta_nll=rr['selected']['nll']-anchor['nll']))
        final=read(cell(seed,ds)/'final/selection.json');configs.append(dict(dataset=ds,seed=seed,**final))
        diagnostics.append(dict(dataset=ds,seed=seed,**read(cell(seed,ds)/'DIAGNOSTICS.json')))
    csv_write('search_trials.csv',trials);csv_write('strategy_effects.csv',effects);csv_write('stage_gains.csv',stage_rows)
    write(RUN/'BEST_CONFIGS.json',dict(cells=configs,criterion='own_validation_accuracy_then_nll_then_inherited_then_id'))
    write(RUN/'CLASS_DIAGNOSTICS.json',dict(cells=diagnostics,query_truth_used=False))
    trained=[t for t in trials if t['origin']=='trained'];inherited=[t for t in trials if t['origin']=='inherited_v1']
    lines=['# PURF V2 搜索与最优配置','',f'实际新训练 {len(trained)} 次；明确继承V1候选 {len(inherited)} 个；新训练累计每fit墙钟时间 {sum(t["train_seconds"] for t in trained)/3600:.2f} 小时（不是整个作业的实际历时，不含I/O与校验）。',
        '阶段1训练时长/学习率；阶段2 gate速度及固定对照；阶段3 gold/pseudo辅助CE拆分；阶段4条件类别加权与移除消融。每一格独立选anchor，相同配置仅拟合一次。',
        '所有新训练从本split seed初始化；继承基线使用封存checkpoint，无继续训练。常数学习率、每50步验证、patience5保持不变。',
        'class_power>0时，以gold及接受的冲突伪标签计数，add-one平滑后逆频次幂并截断到[0.25,4]，用于各CE的加权均值；不改变采样、标签、接受集合，agreement KL不加类别权重。','',
        '| Dataset | Seed | Origin | d | lr | max steps | selected step | gate mode | gate lr | gold aux | pseudo aux | class power | warmup | gate gold | KL | alpha | beta | G/L/R weights | val Acc |',
        '|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|']
    for row in configs:
        c=row['config'];v=row['validation'];weights='/'.join(f'{x:.3f}' for x in row['weights'])
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['origin']} | {c['dim']} | {c['lr']:.6g} | {c['max_steps']} | {v['step']} | {c['gate_mode']} | {c['gate_lr']} | {c['aux_gold']} | {c['aux_pseudo']} | {c['class_power']} | {c['warmup']} | {c['gate_gold']} | {c['keep']} | {row['alpha']:.4f} | {row['beta']:.4f} | {weights} | {pct(v['accuracy'])} |")
    lines += ['', '完整配置见BEST_CONFIGS.json，包括未在主表展开的pseudo系数、batch size等。search_trials.csv保留所有候选；strategy_effects.csv标明相对各阶段anchor的变化字段。多字段变化不能解释成某一单项技巧的净效应。',
        '固定gate候选以及checkpoint仍处warmup内的候选不能归因于alpha/beta学习；gate接近初始值也不能单独证明优化不足。']
    (RUN/'SEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
    lines=['# 训练标签频次与验证集诊断','',
        '每一行仅使用自身划分gold+accepted teach_g/teach_l训练标签计算频次；某类缺失或非零频次max/min≥2时进入温和类别加权候选。validation逐类recall仅作描述，不计算类别权重。完整类别向量见CLASS_DIAGNOSTICS.json。','',
        '| Dataset | Seed | Missing training classes | Positive count max/min | Class-weight eligible |','|---|---:|---:|---:|---|']
    for row in diagnostics:
        counts=row['training_class_counts'];positive=[n for n in counts if n]
        ratio=max(positive)/min(positive) if positive else 0
        lines.append(f"| {row['dataset']} | {row['seed']} | {sum(n==0 for n in counts)} | {ratio:.2f} | {row['class_weight_eligible']} |")
    (RUN/'CLASS_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')

def exploration_report(rows,summary):
    with (RUN/'stage_gains.csv').open() as f:stages=list(csv.DictReader(f))
    lines=['# 固定伪标签后的训练调优增益','',
        '阶段validation收益为描述性汇总，选型发生在各自划分内。query收益仅在全局冻结后计算，不用于重选。',
        '与V1相比，本轮输入、模型结构和伪标签保持固定；训练策略与搜索预算发生变化。继承V1候选确保验证集最优值不会退化，不能保证query也不退化。','',
        '| Dataset | Stage | Mean incremental val Acc pp | Mean total vs V1 pp |','|---|---:|---:|---:|']
    for ds,st in itertools.product(DATASETS,range(1,5)):
        group=[r for r in stages if r['dataset']==ds and int(r['stage'])==st]
        if sorted(int(r['seed']) for r in group)!=list(SEEDS):raise ValueError('Stage seed count differs')
        lines.append(f"| {ds} | {st} | {100*statistics.mean(float(r['delta_previous']) for r in group):+.3f} | {100*statistics.mean(float(r['delta_v1']) for r in group):+.3f} |")
    for pop,_ in POPULATIONS:
        lines += ['',f'## {pop} 实际预测收益','',
            '| Dataset | V1 Acc | V2 Acc | ΔAcc | Improved seeds Acc | V1 F1 | V2 F1 | ΔF1 | Improved seeds F1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ds in DATASETS:
            r=next(r for r in summary if r['dataset']==ds and r['population']==pop);g=[x for x in rows if x['dataset']==ds and x['population']==pop]
            lines.append(f"| {ds} | {ms(r,'v1_accuracy')} | {ms(r,'v2_accuracy')} | {ms(r,'delta_v1_accuracy')} | {sum(x['delta_v1_accuracy']>0 for x in g)}/5 | {ms(r,'v1_macro_f1')} | {ms(r,'v2_macro_f1')} | {ms(r,'delta_v1_macro_f1')} | {sum(x['delta_v1_macro_f1']>0 for x in g)}/5 |")
    lines += ['', 'strategy_effects.csv列出每个候选与本阶段anchor的配对验证差值；changed_count=1的行才是单字段对照。各seed的anchor可能不同，不能把不同背景的多项改动合并成单因素结论。']
    (RUN/'EXPLORATION_GAINS.md').write_text('\n'.join(lines)+'\n')

def audit():
    reg,frozen=verify_frozen();receipt=read(RUN/'EVALUATION_COMPLETE.json');check(receipt['files']);torch.set_num_threads(2)
    from sklearn.metrics import accuracy_score,f1_score
    import sklearn
    results=read(RUN/'RESULTS.json');rows=results['per_split'];summary=results['mean_std']
    index={(r['dataset'],r['seed'],r['population']):r for r in rows}
    if len(index)!=50 or len(rows)!=50 or len(summary)!=10:raise ValueError('Incorrect report dimensions')
    for ds in DATASETS:
        truth,graphhash=graph_truth(reg,ds)
        for seed in SEEDS:
            f,s,p,h,split,router=load(seed,ds);probs=probabilities(reg,seed,ds)
            obj=torch.load(cell(seed,ds)/'final/model.pt',weights_only=True,map_location='cpu')
            model=make_model(f,h,obj['config']);model.load_state_dict(obj['state'])
            torch.testing.assert_close(probs['v2'],predict(model,f,h,f['node_ids']),atol=0,rtol=0)
            for pop,idkey in POPULATIONS:
                ids=np.asarray(split[idkey],dtype=np.int64);row=index[ds,seed,pop]
                for name,prob in probs.items():
                    pred=prob.numpy()[ids].argmax(1)
                    values=dict(accuracy=float(accuracy_score(truth[ids],pred)),macro_f1=float(f1_score(truth[ids],pred,labels=np.arange(prob.shape[1]),average='macro',zero_division=0)))
                    if any(abs(row[name+'_'+k]-v)>1e-12 for k,v in values.items()):raise ValueError('Independent classification mismatch')
                accepted=ids[router['accepted'].numpy()[ids]]
                acc=float(np.mean(router['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None
                if row['pseudo_accuracy']!=acc or row['pseudo_coverage']!=len(accepted)/len(ids):raise ValueError('Pseudo metrics mismatch')
                for b in ('v1','fixed','historical'):
                    for k in ('accuracy','macro_f1'):
                        if abs(row[f'delta_{b}_{k}']-(row['v2_'+k]-row[b+'_'+k]))>1e-14:raise ValueError('Paired difference mismatch')
            print('AUDIT REPLAY',seed,ds,flush=True)
    for r in summary:
        group=[x for x in rows if x['dataset']==r['dataset'] and x['population']==r['population']]
        if sorted(x['seed'] for x in group)!=list(SEEDS):raise ValueError('Invalid aggregate seeds')
        for m in METRICS:
            values=[x[m] for x in group]
            for suffix,fn in [('_mean',np.mean),('_std_ddof0',lambda x:np.std(x,ddof=0)),('_std_ddof1',lambda x:np.std(x,ddof=1))]:
                observed=r[m+suffix]
                if any(v is None for v in values):
                    if observed is not None:raise ValueError('Undefined aggregate differs')
                elif abs(float(fn(values))-observed)>1e-12:raise ValueError('Independent aggregate mismatch')
    verify_frozen();check(receipt['files'])
    write(RUN/'FINAL_AUDIT.json',dict(all_checks_pass=True,rows=50,groups=10,distinct_model_replays=25,sklearn_metrics_verified=True,
        numpy_aggregates_verified=True,pseudo_quality_equals_v1=True,inner_seeds=[],frozen_sha256=sha(RUN/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    sources=merge_records(reg['code_hashes'],read(PREVIOUS/'SOURCE_SNAPSHOT.json')['source_files']);check(sources);copied={}
    for src,digest in sources.items():
        dst=RUN/'source'/Path(src).relative_to(REPO);dst.parent.mkdir(parents=True,exist_ok=True)
        with Path(src).open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        if sha(dst)!=digest:raise ValueError('Source snapshot differs')
        copied[str(dst.relative_to(RUN))]=digest
    write(RUN/'SOURCE_SNAPSHOT.json',dict(files=copied,source_files=sources,standalone_data_package=False))
    write(RUN/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,threads=torch.get_num_threads(),slurm_job_id=os.environ['SLURM_JOB_ID']))
    files={str(p.relative_to(RUN)):sha(p) for p in RUN.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
    write(RUN/'COMPLETE.json',dict(all_checks_pass=True,cell_count=25,rows=50,groups=10,inner_seeds=[],training_seed_equals_split_seed=True,
        pseudo_labels_unchanged=True,files=files,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('COMPLETE: all25 V2 cells independently audited and sealed',flush=True)
