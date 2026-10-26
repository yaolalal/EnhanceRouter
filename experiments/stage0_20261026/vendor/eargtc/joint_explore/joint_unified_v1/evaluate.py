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
    write(RUN/'FROZEN.json',dict(files=files,all_checks_pass=True,cell_count=25,seeds=list(SEEDS),inner_seeds=[],
          test_truth_loaded=False,cross_split_selection=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('GLOBAL FREEZE COMPLETE; all training closed',flush=True)

def verify_frozen():
    slurm();reg=verify_code();frozen=read(RUN/'FROZEN.json')
    if not frozen['all_checks_pass'] or frozen['cell_count']!=25 or frozen['inner_seeds']!=[] or frozen['cross_split_selection'] is not False:raise ValueError('Invalid global freeze')
    check(frozen['files']);check(reg['dependencies']);check(reg['protected']);return reg,frozen

def classification(prob,truth):
    pred=prob.argmax(1);c=prob.shape[1]
    mat=np.bincount(truth*c+pred,minlength=c*c).reshape(c,c)
    tp=mat.diagonal().astype(float);den=mat.sum(0)+mat.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(c),where=den>0)
    return dict(accuracy=float(np.mean(pred==truth)),macro_f1=float(f1.mean()))

def csv_write(path,rows):
    path=RUN/path
    if path.exists():raise ValueError('Refusing report overwrite')
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate(rows):
    output=[]
    for pop,ds in itertools.product(('full_query','eval1000'),DATASETS):
        group=[r for r in rows if r['population']==pop and r['dataset']==ds]
        if sorted(r['split_seed'] for r in group)!=list(SEEDS):raise ValueError('Five distinct seed rows required')
        row=dict(population=pop,dataset=ds,split_count=5)
        for m in METRICS:
            values=[r[m] for r in group if r[m] is not None]
            row[m+'_defined_splits']=len(values)
            for suffix,fn in (('_mean',statistics.mean),('_std_ddof0',statistics.pstdev),('_std_ddof1',statistics.stdev)):
                row[m+suffix]=fn(values) if len(values)==5 else None
        output.append(row)
    return output

def evaluate():
    reg,frozen=verify_frozen()
    marker=RUN/'EVALUATION_STARTED.json'
    if not marker.exists():write(marker,dict(frozen_sha256=sha(RUN/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs();torch.set_num_threads(2)
    rows=[]
    # The historical ensemble is a separate labeled reference, never an optimization input.
    with (REFERENCE/'joint/evaluation/per_split.csv').open(newline='') as f:
        historical={(int(r['split_seed']),r['dataset'],r['population']):r for r in csv.DictReader(f)}
    for ds in DATASETS:
        split0=read(reg['inputs'][key(42,ds)]['paths']['split'])
        if sha(split0['graph_path'])!=split0['graph_sha256']:raise ValueError('Evaluation graph changed')
        graph=torch.load(split0['graph_path'],weights_only=False,map_location='cpu');truth=graph.y.reshape(-1).numpy().astype(np.int64);del graph
        for seed in SEEDS:
            paths=reg['inputs'][key(seed,ds)]['paths'];split=read(paths['split'])
            if split['graph_sha256']!=split0['graph_sha256']:raise ValueError('Seed evaluation graph differs')
            router=torch.load(paths['router'],weights_only=True,map_location='cpu')
            fixed=torch.load(cell(seed,ds)/'fixed/predictions.pt',weights_only=True,map_location='cpu')
            new=torch.load(cell(seed,ds)/'final/predictions.pt',weights_only=True,map_location='cpu')
            if fixed['training_seed']!=seed or new['training_seed']!=seed or fixed['inner_ensemble'] or new['inner_ensemble']:raise ValueError('Seed/ensemble protocol differs')
            for pop,idskey in [('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids')]:
                ids=np.asarray(split[idskey],dtype=np.int64);accepted=ids[router['accepted'].numpy()[ids]]
                row=dict(population=pop,dataset=ds,split_seed=seed,training_seed=seed,inner_ensemble=False,query_count=len(ids),accepted_count=len(accepted))
                for name,prob in [('fixed',fixed['fixed']),('no_b',fixed['no_b']),('new',new['probability'])]:
                    row.update({name+'_'+k:v for k,v in classification(prob.numpy()[ids],truth[ids]).items()})
                row['delta_accuracy']=row['new_accuracy']-row['fixed_accuracy'];row['delta_macro_f1']=row['new_macro_f1']-row['fixed_macro_f1']
                row['pseudo_accuracy']=float(np.mean(router['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None
                row['pseudo_coverage']=len(accepted)/len(ids)
                old=historical[seed,ds,pop]
                row['historical_accuracy']=float(old['accuracy']);row['historical_macro_f1']=float(old['macro_f1'])
                rows.append(row)
    summary=aggregate(rows)
    write(RUN/'RESULTS.json',dict(per_split=rows,mean_std=summary,frozen_sha256=sha(RUN/'FROZEN.json')))
    csv_write('per_split.csv',rows);csv_write('mean_std.csv',summary)
    for pop in ('full_query','eval1000'):
        csv_write(pop+'_per_split.csv',[r for r in rows if r['population']==pop])
        csv_write(pop+'_mean_std.csv',[r for r in summary if r['population']==pop])
    report(rows,summary)
    artifacts=[p for p in RUN.iterdir() if p.name.endswith('.csv') or p.name in ('RESULTS.json','REPORT.md','SEARCH_REPORT.md','search_trials.csv','strategy_effects.csv')]
    write(RUN/'EVALUATION_COMPLETE.json',dict(all_checks_pass=True,metric_rows=50,aggregate_groups=10,
          files={str(p):sha(p) for p in artifacts},frozen_sha256=sha(RUN/'FROZEN.json')))
    print('EVALUATED 50 paired rows, 10 five-seed groups',flush=True)

def percent(x):return 'NA' if x is None else f'{100*x:.2f}'
def meanstd(row,metric):return 'NA' if row[metric+'_mean'] is None else f"{percent(row[metric+'_mean'])} ± {percent(row[metric+'_std_ddof0'])}"

def report(rows,summary):
    lines=['# Router-v9-new：Joint-Fixed 与新单 repair 模型','',
        '主对照：五个历史划分42–46，每次训练 seed 等于划分 seed，两个模型均关闭内层集成。表中 Accuracy、Macro-F1 和覆盖率以百分数表示，增益为百分点；均值±总体标准差（ddof=0），CSV另存ddof=1。','',
        '每个划分独立完成相同的四阶段 validation 搜索；仅使用本划分历史 calibration-half validation 选 checkpoint 和配置，未汇总其他划分 validation 来指导选择。全25格模型和预测冻结后才加载 query truth。','',
        '新模型：GNN/LLM 各自 LayerNorm→可训练降维→GELU 后拼接，一个64维GELU repair分类器。p=(alpha*pG+beta*pL+pRepair)/(alpha+beta+1)，alpha/beta为两个独立softplus参数，repair系数固定1。无pB预测路径。','',
        'Joint-Fixed使用原始数值训练、九配方、checkpoint/残差缩放/融合候选搜索；唯一实验协议变化为单seed训练且与划分一致。历史三成员集成另列参考，不能解释为公平单seed主对照。新模型搜索预算更大，不能把全部差异归因于结构。','',
        'Router完全冻结，接受集合、伪标签、原始分数及完整disagreement路由经过核验。两种新训练使用同一伪标签池；agreement不加CE，new可选agreement KL。伪标签准确率仅在已接受节点计算，覆盖率为接受数/报告集合节点数；两模型的这两项指标相同。','',
        'full-query与eval1000使用同一份冻结全节点预测。eval1000是query子集，仍可能通过无真值伪标签参与训练。本实验是transductive；Router-v9-new上游曾按posthoc query metrics选择，因此不能声称整体为未观察测试集评估。Arxiv沿用历史query口径，Products沿用历史子图。原始数据、专家和hidden缓存是外部哈希依赖。','']
    for pop in ('full_query','eval1000'):
        lines += [f'## {pop}：五种子均值与标准差','',
                  '| Dataset | Fixed Acc | New Acc | ΔAcc | Fixed F1 | New F1 | ΔF1 | Pseudo Acc | Coverage |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['population']==pop:lines.append('| '+r['dataset']+' | '+' | '.join(meanstd(r,m) for m in ('fixed_accuracy','new_accuracy','delta_accuracy','fixed_macro_f1','new_macro_f1','delta_macro_f1','pseudo_accuracy','pseudo_coverage'))+' |')
        lines += ['',f'## {pop}：逐 seed 并列对照','',
                  '| Dataset | Seed | Fixed Acc | New Acc | ΔAcc | Fixed F1 | New F1 | ΔF1 | Pseudo Acc | Coverage |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['population']==pop:lines.append('| '+r['dataset']+' | '+str(r['split_seed'])+' | '+' | '.join(percent(r[m]) for m in ('fixed_accuracy','new_accuracy','delta_accuracy','fixed_macro_f1','new_macro_f1','delta_macro_f1','pseudo_accuracy','pseudo_coverage'))+' |')
        lines += ['',f'## {pop}：无 pB 消融与历史集成参考','',
                  '| Dataset | Fixed single Acc | Fixed no-pB Acc | Historical ensemble Acc | Fixed single F1 | Fixed no-pB F1 | Historical ensemble F1 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['population']==pop:lines.append('| '+r['dataset']+' | '+' | '.join(meanstd(r,m) for m in ('fixed_accuracy','no_b_accuracy','historical_accuracy','fixed_macro_f1','no_b_macro_f1','historical_macro_f1'))+' |')
    lines += ['','具体配置、训练策略消融与所有试验见 [SEARCH_REPORT.md](SEARCH_REPORT.md)、[search_trials.csv](search_trials.csv) 和 [strategy_effects.csv](strategy_effects.csv)。所有报告均保留退化结果；“最优”仅指本次预注册预算内的 validation 选择。','']
    (RUN/'REPORT.md').write_text('\n'.join(lines))
    search_report()

def search_report():
    selection=read(RUN/'SELECTION.json');stages=[read(RUN/f'stage{i}.json') for i in range(1,5)];trials=[];effects=[]
    lines=['# Validation 搜索记录','',
        '四阶段：27个核心维度/学习率/伪标签权重组合；12个训练策略候选；局部超参数细化；逐项移除及辅助CE/KL组合。完全相同的配置在同一seed只训练一次。各格独立使用自己的validation，排名从不使用其他seed或query指标。','',
        '## 最终配置','',
        '| Dataset | Seed | d | lr | pseudo | gate lr ratio | warmup | gate gold only | aux | KL | alpha | beta | weights G/L/R | val Acc | step | trials |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for seed,ds in itertools.product(SEEDS,DATASETS):
        k=key(seed,ds);selected=read(cell(seed,ds)/'final/selection.json');cfg=selected['config'];ranking=selection['cells'][k]['ranking']
        lines.append('| '+ds+' | '+str(seed)+' | '+' | '.join(str(cfg[x]) for x in ('dim','lr','cross','gate_lr','warmup','gate_gold','aux','keep'))+
                     f" | {selected['alpha']:.4f} | {selected['beta']:.4f} | "+'/'.join(f'{w:.3f}' for w in selected['weights'])+
                     f" | {percent(selected['validation']['accuracy'])} | {selected['validation']['step']} | {len(ranking)} |")
        anchor=stages[1]['selection_evidence'][k][0]
        for row in ranking:
            r=read(candidate_path(seed,ds,row['config'])/'result.json')
            trials.append(dict(dataset=ds,seed=seed,candidate_id=row['id'],stages=','.join(str(i+1) for i,s in enumerate(stages) if row['id'] in {cid(c) for c in s['candidates'][k]}),
                selected=row['id']==selected['candidate_id'],**row['config'],validation_accuracy=row['accuracy'],validation_nll=row['nll'],
                checkpoint_step=row['step'],train_seconds=r['train_seconds'],parameter_count=r['parameter_count']))
        byid={r['id']:r for r in ranking}
        for cfg0 in stages[1]['candidates'][k]:
            r=byid[cid(cfg0)]
            effects.append(dict(dataset=ds,seed=seed,stage='strategy',candidate_id=r['id'],
                changed_fields=','.join(x for x in cfg0 if cfg0[x]!=anchor['config'][x]) or 'anchor',
                gate_lr=cfg0['gate_lr'],warmup=cfg0['warmup'],gate_gold=cfg0['gate_gold'],aux=cfg0['aux'],keep=cfg0['keep'],
                delta_validation_accuracy=r['accuracy']-anchor['accuracy'],delta_validation_nll=r['nll']-anchor['nll']))
    csv_write('search_trials.csv',trials);csv_write('strategy_effects.csv',effects)
    lines += ['','## 训练策略相对本格核心 anchor 的 validation 配对差值','',
              '下表为各数据集五seed的描述性汇总，不用于其他seed的候选生成或选择。交互配置单独列出。','',
              '| Dataset | gate lr | warmup | gate Gold | aux | KL | Δval Acc pp mean±std | improved seeds |','|---|---:|---:|---|---:|---:|---:|---:|']
    groups={}
    for r in effects:groups.setdefault(tuple(r[x] for x in ('dataset','gate_lr','warmup','gate_gold','aux','keep')),[]).append(r['delta_validation_accuracy'])
    for k,values in groups.items():
        lines.append('| '+' | '.join(map(str,k))+f' | {100*statistics.mean(values):+.3f} ± {100*statistics.pstdev(values):.3f} | {sum(v>0 for v in values)}/{len(values)} |')
    lines += ['',f'共 {len(trials)} 个新模型训练配置，训练累计 {sum(r["train_seconds"] for r in trials)/3600:.2f} CPU作业墙钟小时（并发任务求和，不等于实际经过时间），另有25×9=225个Fixed成员。',
              '最优仅限已探索候选；validation重复使用存在选择偏差。最终checkpoint保留accuracy/NLL两类完整状态，最终部署使用accuracy优先checkpoint。没有内层模型平均。','']
    (RUN/'SEARCH_REPORT.md').write_text('\n'.join(lines))

def audit():
    reg,frozen=verify_frozen();receipt=read(RUN/'EVALUATION_COMPLETE.json');check(receipt['files'])
    check(reg['dependencies']);check(reg['protected'])
    from sklearn.metrics import accuracy_score,f1_score
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs();torch.set_num_threads(2)
    rows=read(RUN/'RESULTS.json')['per_split'];indexed={(r['split_seed'],r['dataset'],r['population']):r for r in rows}
    for ds in DATASETS:
        sp=read(reg['inputs'][key(42,ds)]['paths']['split']);check({sp['graph_path']:sp['graph_sha256']})
        graph=torch.load(sp['graph_path'],weights_only=False,map_location='cpu');truth=graph.y.reshape(-1).numpy();del graph
        for seed in SEEDS:
            f,s,p,h,split,router=load(seed,ds)
            checkpoint=torch.load(cell(seed,ds)/'final/model.pt',weights_only=True,map_location='cpu')
            model=make_model(f,h,checkpoint['config']);model.load_state_dict(checkpoint['state'])
            saved=torch.load(cell(seed,ds)/'final/predictions.pt',weights_only=True,map_location='cpu')['probability']
            reproduced=predict(model,f,h,f['node_ids']);torch.testing.assert_close(saved,reproduced,atol=0,rtol=0)
            fixed=torch.load(cell(seed,ds)/'fixed/predictions.pt',weights_only=True,map_location='cpu')
            for pop,idskey in [('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids')]:
                ids=np.asarray(split[idskey]);r=indexed[seed,ds,pop]
                for name,probs in [('new',saved),('fixed',fixed['fixed']),('no_b',fixed['no_b'])]:
                    pred=probs.numpy()[ids].argmax(1);c=probs.shape[1]
                    expected={'accuracy':float(accuracy_score(truth[ids],pred)),
                              'macro_f1':float(f1_score(truth[ids],pred,labels=np.arange(c),average='macro',zero_division=0))}
                    for metric,value in expected.items():
                        if abs(value-r[name+'_'+metric])>1e-12:raise ValueError('Independent metric mismatch')
                accepted=ids[router['accepted'].numpy()[ids]]
                expected=float(np.mean(router['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None
                if r['pseudo_accuracy']!=expected or r['pseudo_coverage']!=len(accepted)/len(ids):raise ValueError('Pseudo report mismatch')
                for metric in ('accuracy','macro_f1'):
                    if abs(r['delta_'+metric]-(r['new_'+metric]-r['fixed_'+metric]))>1e-14:raise ValueError('Delta mismatch')
            print('INDEPENDENT AUDIT',seed,ds,flush=True)
    summary=read(RUN/'RESULTS.json')['mean_std']
    for r in summary:
        group=[x for x in rows if x['dataset']==r['dataset'] and x['population']==r['population']]
        if len(group)!=5:raise ValueError('Aggregate seed count mismatch')
        for metric in METRICS:
            values=[x[metric] for x in group]
            for suffix,fn in [('_mean',np.mean),('_std_ddof0',lambda x:np.std(x,ddof=0)),('_std_ddof1',lambda x:np.std(x,ddof=1))]:
                if any(v is None for v in values):
                    if r[metric+suffix] is not None:raise ValueError('Undefined aggregation mismatch')
                elif abs(float(fn(values))-r[metric+suffix])>1e-12:raise ValueError('Independent aggregate mismatch')
    check(reg['protected']);check(receipt['files'])
    write(RUN/'FINAL_AUDIT.json',dict(all_checks_pass=True,paired_rows=50,aggregate_groups=10,
          whole_model_predictions_reproduced=25,metrics_independently_recomputed=True,protected_files=len(reg['protected']),
          frozen_sha256=sha(RUN/'FROZEN.json'),report_sha256=sha(RUN/'REPORT.md'),
          slurm_job_id=os.environ['SLURM_JOB_ID'],inner_seeds=[],cross_split_selection=False))
    print('FINAL AUDIT PASSED',flush=True)
