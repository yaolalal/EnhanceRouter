"""Cross-variant truth gate, paired reports, and independent post-freeze audit."""
import csv,itertools,os,statistics,sys,shutil
from pathlib import Path
import numpy as np
import torch
from .protocol import ROOT,PREVIOUS,ALL_DATASETS,SEEDS,REPO,read,write,save,sha,check,slurm
from .training import DEFAULTS,make_model,predict,score
from .search import cid
VARIANTS=('original','min1')
LABELS={'original':'Router-v10-MS-Original','min1':'Router-v10-MS-CiteSeer-Min1'}

def root_for(variant,ds):return ROOT/('original' if ds!='citeseer' else variant)
def cell_for(variant,seed,ds):return root_for(variant,ds)/f'seed{seed}'/ds

def global_freeze():
    slurm();files={};counts={}
    if (ROOT/'EVALUATION_STARTED.json').exists():raise ValueError('Evaluation already started')
    for variant in VARIANTS:
        root=ROOT/variant;reg=read(root/'REGISTRATION.json');f=read(root/'FROZEN.json')
        expected=25 if variant=='original' else 5
        if not f['all_checks_pass'] or f['cell_count']!=expected or f['inner_seeds']!=[] or f['test_truth_loaded'] is not False:raise ValueError('Incomplete variant freeze')
        for records in (reg['code_hashes'],reg['dependencies'],reg['protected'],f['files']):check(records)
        files.update(f['files']);files[str(root/'FROZEN.json')]=sha(root/'FROZEN.json');counts[variant]=expected
    shared={f'{seed}/{ds}':str(cell_for('original',seed,ds)) for seed,ds in itertools.product(SEEDS,ALL_DATASETS) if ds!='citeseer'}
    # Min1 is defined as an exact source alias outside CiteSeer, without cross-cell score selection.
    write(ROOT/'FROZEN.json',dict(all_checks_pass=True,distinct_cells=30,logical_cells=50,variants=counts,files=files,
        shared_min1_cells=shared,inner_seeds=[],training_seed_equals_split_seed=True,test_truth_loaded=False,
        cross_split_selection=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('BOTH VARIANTS FROZEN; ALL FITTING CLOSED',flush=True)

def verify_global():
    slurm();f=read(ROOT/'FROZEN.json')
    if f['distinct_cells']!=30 or f['logical_cells']!=50 or not f['all_checks_pass'] or f['inner_seeds']!=[]:raise ValueError('Invalid global freeze')
    check(f['files'])
    for v in VARIANTS:
        r=read(ROOT/v/'REGISTRATION.json')
        for group in ('code_hashes','dependencies','protected'):check(r[group])
    return f

def classification(prob,truth):
    pred=prob.argmax(1);c=prob.shape[1]
    mat=np.bincount(truth*c+pred,minlength=c*c).reshape(c,c)
    tp=mat.diagonal().astype(float);den=mat.sum(0)+mat.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(c),where=den>0)
    return dict(accuracy=float(np.mean(pred==truth)),macro_f1=float(f1.mean()))

def csv_write(name,rows):
    with (ROOT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pct(x):return 'NA' if x is None else f'{100*x:.2f}'
def meanstd(r,k):return 'NA' if r[k+'_mean'] is None else f"{pct(r[k+'_mean'])} ± {pct(r[k+'_std_ddof0'])}"

def aggregate(rows,groupkeys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in groupkeys),[]).append(r)
    result=[]
    for key,g in groups.items():
        if sorted(r['seed'] for r in g)!=list(SEEDS):raise ValueError('Exactly five distinct seeds required')
        out=dict(zip(groupkeys,key));out['seed_count']=5
        for m in metrics:
            values=[r[m] for r in g if r[m] is not None];out[m+'_defined_seeds']=len(values)
            for suffix,fn in (('_mean',statistics.mean),('_std_ddof0',statistics.pstdev),('_std_ddof1',statistics.stdev)):
                out[m+suffix]=fn(values) if len(values)==5 else None
        result.append(out)
    return result

METRICS=('fixed_reference_accuracy','fixed_reference_macro_f1','purf_accuracy','purf_macro_f1',
    'fixed_matched_accuracy','fixed_matched_macro_f1','default_accuracy','default_macro_f1',
    'delta_reference_accuracy','delta_reference_macro_f1','delta_matched_accuracy','delta_matched_macro_f1',
    'delta_default_accuracy','delta_default_macro_f1','pseudo_accuracy','pseudo_coverage','reference_pseudo_accuracy','reference_pseudo_coverage')

def evaluate():
    verify_global();torch.set_num_threads(2)
    write(ROOT/'EVALUATION_STARTED.json',dict(frozen_sha256=sha(ROOT/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    regs={v:read(ROOT/v/'REGISTRATION.json') for v in VARIANTS};oldreg=read(PREVIOUS/'REGISTRATION.json');rows=[]
    for ds in ALL_DATASETS:
        s0=read(regs['original']['inputs'][f'42/{ds}']['paths']['split']);check({s0['graph_path']:s0['graph_sha256']})
        graph=torch.load(s0['graph_path'],weights_only=False,map_location='cpu');truth=graph.y.reshape(-1).numpy().astype(np.int64);del graph
        for seed in SEEDS:
            ref=torch.load(PREVIOUS/f'seed{seed}/{ds}/fixed/predictions.pt',weights_only=True,map_location='cpu')
            oldr=torch.load(oldreg['inputs'][f'{seed}/{ds}']['paths']['router'],weights_only=True,map_location='cpu')
            if ref['training_seed']!=seed or ref['inner_ensemble'] is not False:raise ValueError('Reference seed protocol differs')
            for variant in VARIANTS:
                actual='original' if ds!='citeseer' else variant;record=regs[actual]['inputs'][f'{seed}/{ds}'];split=read(record['paths']['split'])
                if split['graph_sha256']!=s0['graph_sha256']:raise ValueError('Graph mismatch')
                dest=cell_for(variant,seed,ds);r=torch.load(record['paths']['router'],weights_only=True,map_location='cpu')
                purf=torch.load(dest/'final/predictions.pt',weights_only=True,map_location='cpu')
                fixed=torch.load(dest/'fixed/predictions.pt',weights_only=True,map_location='cpu')
                default=torch.load(dest/'final/default.pt',weights_only=True,map_location='cpu')
                for obj in (purf,fixed,default):
                    if obj['training_seed']!=seed or obj['inner_ensemble'] is not False:raise ValueError('Seed/ensemble protocol differs')
                for pop,key in [('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids')]:
                    ids=np.asarray(split[key],dtype=np.int64);accepted=ids[r['accepted'].numpy()[ids]];oa=ids[oldr['accepted'].numpy()[ids]]
                    row=dict(variant=variant,router=LABELS[variant],dataset=ds,population=pop,seed=seed,training_seed=seed,inner_ensemble=False,
                        shared_with_original=variant=='min1' and ds!='citeseer',query_count=len(ids),accepted_count=len(accepted))
                    for name,prob in [('fixed_reference',ref['fixed']),('purf',purf['probability']),('fixed_matched',fixed['fixed']),('default',default['probability'])]:
                        row.update({name+'_'+k:v for k,v in classification(prob.numpy()[ids],truth[ids]).items()})
                    for baseline in ('reference','matched','default'):
                        prefix=baseline if baseline=='default' else 'fixed_'+baseline
                        for m in ('accuracy','macro_f1'):row['delta_'+baseline+'_'+m]=row['purf_'+m]-row[prefix+'_'+m]
                    row['pseudo_accuracy']=float(np.mean(r['selected_pred'].numpy()[accepted]==truth[accepted])) if len(accepted) else None
                    row['pseudo_coverage']=len(accepted)/len(ids)
                    row['reference_pseudo_accuracy']=float(np.mean(oldr['selected_pred'].numpy()[oa]==truth[oa])) if len(oa) else None
                    row['reference_pseudo_coverage']=len(oa)/len(ids);rows.append(row)
        print('EVALUATED',ds,flush=True)
    summary=aggregate(rows,('variant','population','dataset'),METRICS)
    write(ROOT/'RESULTS.json',dict(per_split=rows,mean_std=summary,frozen_sha256=sha(ROOT/'FROZEN.json')))
    csv_write('per_split.csv',rows);csv_write('mean_std.csv',summary)
    for pop in ('full_query','eval1000'):
        csv_write(pop+'_per_split.csv',[r for r in rows if r['population']==pop]);csv_write(pop+'_mean_std.csv',[r for r in summary if r['population']==pop])
    report(rows,summary);search_report();exploration_report(rows,summary)
    artifacts=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix=='.csv' or p.name in ('RESULTS.json','REPORT.md','SEARCH_REPORT.md','EXPLORATION_GAINS.md'))]
    write(ROOT/'EVALUATION_COMPLETE.json',dict(all_checks_pass=True,rows=100,groups=20,files={str(p):sha(p) for p in artifacts},
        frozen_sha256=sha(ROOT/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))

def report(rows,summary):
    index={(r['variant'],r['population'],r['dataset']):r for r in summary}
    lines=['# Joint-PURF：两种 Router-v10 五种子实验','',
        'Accuracy、Macro-F1、伪标签准确率和覆盖率均以百分数表示；均值±总体标准差（ddof=0），CSV同时保留样本标准差ddof=1。增益单位为百分点。','',
        '五个历史划分seed42–46，所有本轮训练seed等于split seed，inner_seeds=[]，不做内层集成。每个Router/seed/dataset单独使用本划分calibration-half validation进行四阶段搜索，以validation Accuracy优先、NLL次之选checkpoint和配置。跨seed不共享验证成绩进行选择。','',
        '主表Joint-Fixed为上一轮Router-v9-new的已封存单seed对照（runs/joint_unified_v1_20260913），用于承接历史比较；不是本轮Router-v10下的Fixed。下方另列两个Router-v10各自重新训练的同Router Joint-Fixed，区分Router与模型变化。','',
        'Original来源为runs/router_v10_observed_best_five_seeds_20260913_v1；Min1仅使用runs/router_v10_citeseer_min_accept_ablation_20260913_v1中的validation_min1替换CiteSeer。其余四数据集完全相同，明确共享Original搜索/模型，不重复训练或伪装成独立实验。','',
        'Joint-PURF结构保持上一版：GNN/LLM独立LN→Linear(d)→GELU后拼接，一个64维repair分类器；p=(alpha*pG+beta*pL+pRepair)/(alpha+beta+1)，alpha/beta独立softplus可学习、初始1，repair系数1，无pB。专家冻结。','',
        '两路冲突伪标签CE保留历史池与权重语义；agreement不额外加伪标签CE，可选KL。Router接受集合、伪标签、包括拒绝节点的路由及float64原始分数均保留；Fixed的validation路由由对应冻结头重放。','',
        '全部30个不同输入实验（逻辑上50格）冻结后统一加载query truth。full-query/eval1000来自同一份全节点预测，eval1000仅取历史子集不重新排序。伪标签准确率仅在接受节点计算，覆盖率=接受节点/相应报告集合；零接受准确率为NA。','',
        '上游Router Original含历史query观测后分支选择，Min1阈值选于validation也不消除整个上游的选择历史；本轮是transductive历史基准扩展，不能称未观察测试集盲测。PURF搜索预算大于Fixed，差异不能全归于结构；“最优”仅指预注册候选内的本划分validation最优。原始数据/专家/hidden仍为外部哈希绑定依赖。','']
    for pop in ('full_query','eval1000'):
        lines += [f'## {pop}：主表五种子均值±标准差','',
            '| Dataset | Joint-Fixed Acc | Original+PURF Acc | Min1+PURF Acc | Joint-Fixed F1 | Original+PURF F1 | Min1+PURF F1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for ds in ALL_DATASETS:
            a=index['original',pop,ds];b=index['min1',pop,ds]
            lines.append('| '+ds+' | '+' | '.join([meanstd(a,'fixed_reference_accuracy'),meanstd(a,'purf_accuracy'),meanstd(b,'purf_accuracy'),meanstd(a,'fixed_reference_macro_f1'),meanstd(a,'purf_macro_f1'),meanstd(b,'purf_macro_f1')])+' |')
        lines += ['',f'## {pop}：同 Router 的 Joint-Fixed 对照','',
            '| Dataset | Router | Fixed Acc | PURF Acc | ΔAcc | Fixed F1 | PURF F1 | ΔF1 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for ds,v in itertools.product(ALL_DATASETS,VARIANTS):
            r=index[v,pop,ds];lines.append('| '+ds+' | '+v+' | '+' | '.join(meanstd(r,k) for k in ('fixed_matched_accuracy','purf_accuracy','delta_matched_accuracy','fixed_matched_macro_f1','purf_macro_f1','delta_matched_macro_f1'))+' |')
        lines += ['',f'## {pop}：冻结 Router 伪标签质量','',
            '| Dataset | Fixed参考 Pseudo Acc | Fixed参考 Coverage | Original Pseudo Acc | Original Coverage | Min1 Pseudo Acc | Min1 Coverage |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for ds in ALL_DATASETS:
            a=index['original',pop,ds];b=index['min1',pop,ds]
            lines.append('| '+ds+' | '+' | '.join([meanstd(a,'reference_pseudo_accuracy'),meanstd(a,'reference_pseudo_coverage'),meanstd(a,'pseudo_accuracy'),meanstd(a,'pseudo_coverage'),meanstd(b,'pseudo_accuracy'),meanstd(b,'pseudo_coverage')])+' |')
        lines += ['',f'## {pop}：逐 seed 结果与相对主表 Fixed 的增益','',
            '| Dataset | Router | Seed | Fixed Acc | PURF Acc | ΔAcc | Fixed F1 | PURF F1 | ΔF1 | Pseudo Acc | Coverage |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['population']==pop:lines.append('| '+r['dataset']+' | '+r['variant']+' | '+str(r['seed'])+' | '+' | '.join(pct(r[k]) for k in ('fixed_reference_accuracy','purf_accuracy','delta_reference_accuracy','fixed_reference_macro_f1','purf_macro_f1','delta_reference_macro_f1','pseudo_accuracy','pseudo_coverage'))+' |')
    lines += ['','[SEARCH_REPORT.md](SEARCH_REPORT.md)保留全部最终配置和训练策略验证集配对增益；[EXPLORATION_GAINS.md](EXPLORATION_GAINS.md)区分阶段validation增益与默认模型→调优模型的实际测试增益。逐seed同Router Fixed增益见per_split.csv。','']
    with (ROOT/'REPORT.md').open('x') as f:f.write('\n'.join(lines))

def search_report():
    trials=[];effects=[];stage_rows=[];lines=['# Joint-PURF Router-v10 搜索记录','',
        '每格四阶段独立搜索：27个dim×lr×pseudo核心组合；12个gate学习率/预热/gold-only/aux CE/KL策略；局部细化；逐项移除与辅助正则组合。同格相同配置仅训练一次。Min1非CiteSeer的20格引用Original，不计入独立训练次数。','',
        '| Router | Dataset | Seed | d | lr | pseudo | gate lr | warmup | gold-only | aux | KL | alpha | beta | G/L/R weights | val Acc | step | trials |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for v in VARIANTS:
        root=ROOT/v;selection=read(root/'SELECTION.json');stages=[read(root/f'stage{i}.json') for i in range(1,5)]
        for key,item in selection['cells'].items():
            seed,ds=key.split('/');seed=int(seed);dest=root/f'seed{seed}'/ds;chosen=read(dest/'final/selection.json');cfg=chosen['config'];ranking=item['ranking'];byid={r['id']:r for r in ranking}
            lines.append('| '+v+' | '+ds+' | '+str(seed)+' | '+' | '.join(str(cfg[k]) for k in ('dim','lr','cross','gate_lr','warmup','gate_gold','aux','keep'))+f" | {chosen['alpha']:.4f} | {chosen['beta']:.4f} | "+'/'.join(f'{w:.3f}' for w in chosen['weights'])+f" | {pct(chosen['validation']['accuracy'])} | {chosen['validation']['step']} | {len(ranking)} |")
            for r in ranking:
                detail=read(dest/'candidates'/r['id']/'result.json')
                trials.append(dict(variant=v,dataset=ds,seed=seed,candidate_id=r['id'],selected=r['id']==chosen['candidate_id'],**r['config'],
                    validation_accuracy=r['accuracy'],validation_nll=r['nll'],checkpoint_step=r['step'],train_seconds=detail['train_seconds'],parameter_count=detail['parameter_count']))
            anchor=stages[1]['selection_evidence'][key][0]
            for c in stages[1]['candidates'][key]:
                r=byid[cid(c)];effects.append(dict(variant=v,dataset=ds,seed=seed,candidate_id=r['id'],changed_fields=','.join(k for k in c if c[k]!=anchor['config'][k]) or 'anchor',
                    gate_lr=c['gate_lr'],warmup=c['warmup'],gate_gold=c['gate_gold'],aux=c['aux'],keep=c['keep'],delta_validation_accuracy=r['accuracy']-anchor['accuracy'],delta_validation_nll=r['nll']-anchor['nll']))
            prev=byid[cid(DEFAULTS)];seen=set()
            for i,st in enumerate(stages,1):
                seen.update(cid(c) for c in st['candidates'][key]);best=min((byid[c] for c in seen),key=lambda r:(-r['accuracy'],r['nll'],r['id']))
                stage_rows.append(dict(variant=v,dataset=ds,seed=seed,stage=i,unique_candidates=len(seen),validation_accuracy=best['accuracy'],
                    delta_previous=best['accuracy']-prev['accuracy'],delta_default=best['accuracy']-byid[cid(DEFAULTS)]['accuracy']))
                prev=best
    csv_write('search_trials.csv',trials);csv_write('strategy_effects.csv',effects);csv_write('stage_gains.csv',stage_rows)
    lines += ['','## 训练策略相对同格核心 anchor 的 validation 配对增益','',
        '| Router | Dataset | gate lr | warmup | gold-only | aux | KL | Δval Acc pp mean±std | improved seeds |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|']
    groups={}
    for r in effects:groups.setdefault(tuple(r[k] for k in ('variant','dataset','gate_lr','warmup','gate_gold','aux','keep')),[]).append(r['delta_validation_accuracy'])
    for k,values in groups.items():lines.append('| '+' | '.join(map(str,k))+f' | {100*statistics.mean(values):+.3f} ± {100*statistics.pstdev(values):.3f} | {sum(x>0 for x in values)}/{len(values)} |')
    lines += ['',f'实际 {len(trials)} 个PURF配置拟合，另有30×9=270个同Router Fixed成员。PURF累计训练时间 {sum(r["train_seconds"] for r in trials)/3600:.2f} CPU作业墙钟小时（并发求和）。',
        '这些validation汇总只作描述，不用于跨seed/Router决策；单独策略不保证测试增益，交互配置不能解释为某单项策略的净效应。若最终checkpoint位于warmup结束时，alpha/beta可能尚未更新，应结合step和系数判断。','']
    with (ROOT/'SEARCH_REPORT.md').open('x') as f:f.write('\n'.join(lines))

def exploration_report(rows,summary):
    lines=['# 有序探索带来的增益','',
        'validation阶段增益用于描述搜索过程；默认→调优的full-query/eval1000结果用于检验实际预测收益。默认配置在stage1已注册，和最终模型一起在读取query truth前导出冻结，未依测试结果重选。默认：d64, lr0.001, pseudo1, gate lr1，无warmup/gold-only/aux/KL。','']
    with (ROOT/'stage_gains.csv').open() as f:stage_rows=list(csv.DictReader(f))
    lines += ['## 四阶段 validation 累计增益','', '| Router | Dataset | Stage | Δ previous pp | Δ default pp |','|---|---|---:|---:|---:|']
    groups={}
    for r in stage_rows:groups.setdefault((r['variant'],r['dataset'],r['stage']),[]).append(r)
    for k,g in groups.items():lines.append('| '+' | '.join(k)+' | '+' | '.join(f'{100*statistics.mean(float(r[m]) for r in g):+.3f}' for m in ('delta_previous','delta_default'))+' |')
    for pop in ('full_query','eval1000'):
        lines += ['',f'## {pop}：默认与调优后的实际结果','',
            '| Router | Dataset | Default Acc | Tuned Acc | ΔAcc | Default F1 | Tuned F1 | ΔF1 |','|---|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['population']==pop:lines.append('| '+r['variant']+' | '+r['dataset']+' | '+' | '.join(meanstd(r,k) for k in ('default_accuracy','purf_accuracy','delta_default_accuracy','default_macro_f1','purf_macro_f1','delta_default_macro_f1'))+' |')
    lines += ['','小幅均值变化与五seed波动应一起解读；不声称统计显著或单一技巧普遍有效。所有退化结果保留。','']
    with (ROOT/'EXPLORATION_GAINS.md').open('x') as f:f.write('\n'.join(lines))

def audit():
    verify_global();receipt=read(ROOT/'EVALUATION_COMPLETE.json');check(receipt['files']);torch.set_num_threads(2)
    from sklearn.metrics import accuracy_score,f1_score
    from eargtc.data import _install_pyg_optional_extension_stubs
    from eargtc.multiseed_joint.experiment import prepare_arrays
    _install_pyg_optional_extension_stubs()
    results=read(ROOT/'RESULTS.json');rows=results['per_split'];summary=results['mean_std']
    index={(r['variant'],r['dataset'],r['seed'],r['population']):r for r in rows}
    if len(index)!=100 or len(rows)!=100 or len(summary)!=20:raise ValueError('Report row counts differ')
    regs={v:read(ROOT/v/'REGISTRATION.json') for v in VARIANTS};old=read(PREVIOUS/'REGISTRATION.json');replays=0
    for ds in ALL_DATASETS:
        sp=read(regs['original']['inputs'][f'42/{ds}']['paths']['split']);check({sp['graph_path']:sp['graph_sha256']})
        graph=torch.load(sp['graph_path'],weights_only=False,map_location='cpu');truth=graph.y.reshape(-1).numpy();del graph
        for seed in SEEDS:
            oldr=torch.load(old['inputs'][f'{seed}/{ds}']['paths']['router'],weights_only=True,map_location='cpu')
            reference=torch.load(PREVIOUS/f'seed{seed}/{ds}/fixed/predictions.pt',weights_only=True,map_location='cpu')['fixed']
            for v in VARIANTS:
                if v=='min1' and ds!='citeseer':
                    for pop in ('full_query','eval1000'):
                        a=index['original',ds,seed,pop];b=index['min1',ds,seed,pop]
                        if any(a[m]!=b[m] for m in METRICS):raise ValueError('Shared input reported different results')
                    continue
                paths=regs[v]['inputs'][f'{seed}/{ds}']['paths'];split=read(paths['split']);dest=cell_for(v,seed,ds)
                u,e,safe,r=[torch.load(paths[k],weights_only=True,map_location='cpu',mmap=True) for k in ('upstream','embeddings','supervision','router')]
                f,s,p=prepare_arrays(split,u,e,safe,r,ds);h=torch.load(paths['hidden'],weights_only=True,map_location='cpu',mmap=True)['hidden']
                modelobj=torch.load(dest/'final/model.pt',weights_only=True,map_location='cpu');model=make_model(f,h,modelobj['config']);model.load_state_dict(modelobj['state'])
                purf=torch.load(dest/'final/predictions.pt',weights_only=True,map_location='cpu')['probability']
                torch.testing.assert_close(purf,predict(model,f,h,f['node_ids']),atol=0,rtol=0)
                defaults=torch.load(dest/'final/default.pt',weights_only=True,map_location='cpu');dm=torch.load(dest/'candidates'/cid(DEFAULTS)/'checkpoint.pt',weights_only=True,map_location='cpu')
                if defaults['candidate_sha256']!=sha(dest/'candidates'/cid(DEFAULTS)/'checkpoint.pt'):raise ValueError('Default source differs')
                model0=make_model(f,h,DEFAULTS);model0.load_state_dict(dm['states']['accuracy'])
                torch.testing.assert_close(defaults['probability'],predict(model0,f,h,f['node_ids']),atol=0,rtol=0)
                fixed=torch.load(dest/'fixed/predictions.pt',weights_only=True,map_location='cpu')['fixed'];replays+=1
                for pop,idskey in [('full_query','evaluation_full_ids'),('eval1000','evaluation_1000_ids')]:
                    ids=np.asarray(split[idskey]);row=index[v,ds,seed,pop]
                    for name,probs in [('fixed_reference',reference),('purf',purf),('default',defaults['probability']),('fixed_matched',fixed)]:
                        pred=probs.numpy()[ids].argmax(1);c=probs.shape[1]
                        independent=dict(accuracy=float(accuracy_score(truth[ids],pred)),macro_f1=float(f1_score(truth[ids],pred,labels=np.arange(c),average='macro',zero_division=0)))
                        for m,val in independent.items():
                            if abs(val-row[name+'_'+m])>1e-12:raise ValueError('Independent classification mismatch')
                    for router,prefix in [(r,''),(oldr,'reference_')]:
                        ids0=ids[router['accepted'].numpy()[ids]];acc=float(np.mean(router['selected_pred'].numpy()[ids0]==truth[ids0])) if len(ids0) else None
                        if row[prefix+'pseudo_accuracy']!=acc or row[prefix+'pseudo_coverage']!=len(ids0)/len(ids):raise ValueError('Pseudo report mismatch')
                    for baseline in ('reference','matched','default'):
                        prefix=baseline if baseline=='default' else 'fixed_'+baseline
                        for m in ('accuracy','macro_f1'):
                            if abs(row['delta_'+baseline+'_'+m]-(row['purf_'+m]-row[prefix+'_'+m]))>1e-14:raise ValueError('Paired delta mismatch')
                print('AUDIT REPLAY',v,seed,ds,flush=True)
    for r in summary:
        group=[x for x in rows if all(x[k]==r[k] for k in ('variant','population','dataset'))]
        if sorted(x['seed'] for x in group)!=list(SEEDS):raise ValueError('Invalid aggregate seed count')
        for m in METRICS:
            values=[x[m] for x in group]
            for suffix,fn in [('_mean',np.mean),('_std_ddof0',lambda x:np.std(x,ddof=0)),('_std_ddof1',lambda x:np.std(x,ddof=1))]:
                if any(x is None for x in values):
                    if r[m+suffix] is not None:raise ValueError('Undefined metric aggregate differs')
                elif abs(float(fn(values))-r[m+suffix])>1e-12:raise ValueError('Independent aggregate mismatch')
    verify_global();check(receipt['files'])
    write(ROOT/'FINAL_AUDIT.json',dict(all_checks_pass=True,rows=100,groups=20,distinct_model_replays=replays,default_model_replays=replays,
        sklearn_metrics_verified=True,numpy_aggregates_verified=True,shared_20_cells_identical=True,inner_seeds=[],
        frozen_sha256=sha(ROOT/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    sources={}
    for v in VARIANTS:sources.update(regs[v]['code_hashes'])
    check(sources);copied={}
    for src,hsh in sources.items():
        rel=Path(src).relative_to(REPO);dst=ROOT/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        with Path(src).open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        if sha(dst)!=hsh:raise ValueError('Source copy differs')
        copied[str(dst.relative_to(ROOT))]=hsh
    write(ROOT/'SOURCE_SNAPSHOT.json',dict(files=copied,source_files=sources,standalone_data_package=False))
    write(ROOT/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),
        node=os.environ.get('SLURMD_NODENAME'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
    write(ROOT/'COMPLETE.json',dict(all_checks_pass=True,distinct_cells=30,logical_cells=50,rows=100,groups=20,inner_seeds=[],
        training_seed_equals_split_seed=True,files=files,slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('COMPLETE: both Router-v10 Joint-PURF experiments independently audited',flush=True)
