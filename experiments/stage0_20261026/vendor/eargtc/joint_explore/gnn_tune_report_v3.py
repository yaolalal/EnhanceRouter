"""Post-freeze tuned GNN evaluation, independent audits and side-by-side report."""
import os,csv,json,math
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from eargtc.joint_explore.gnn_tune import protocol as p,engine
from eargtc.joint_explore.gnn_downstream.io import read,write,load,save,sha,verify,models,DATASETS,VARIANTS,JOINT,ROOT
from eargtc.joint_explore.gnn_downstream.core import seed_all,encode,adjacency
from eargtc.joint_explore.gnn_downstream.report import truth,csv_write,aggregate,METHODS,METRICS

OUTPUT=p.RUN/'reports/v3'

def compare_replay(prob,replay):
    """Roundoff tolerance diagnosed without truth; all node classes must agree."""
    error=float((prob-replay).abs().max())
    assert error<=5e-5, f'Probability replay drift: {error}'
    assert torch.equal(prob.argmax(1),replay.argmax(1)), 'Replay changed a node class'
    return error

def confusion_metrics(y,pred,classes):
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64)
    cm=np.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes)
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(classes),where=den!=0)
    return float(np.trace(cm)/len(y)),float(f1.mean())

def moments(values):
    if len(values)!=5:raise ValueError('Exactly five seeds required')
    mean=sum(values)/5;ss=sum((v-mean)**2 for v in values)
    return mean,math.sqrt(ss/5),math.sqrt(ss/4)

def evaluation_sources(source):
    pre=read(source/'PREREGISTERED.json');frozen=read(source/'FROZEN.json');delivery=read(source/'DELIVERY.json')
    files={}
    for mapping in (pre.get('code_hashes',{}),pre['reference_hashes'],frozen['files'],delivery['files']):
        for filename,h in mapping.items():
            if filename in files and files[filename]!=h:raise ValueError('Inconsistent source seals: '+filename)
            files[filename]=h
    verify(files)
    return files

def pretest():
    p.require();p.gate();lock=read(p.RUN/'FROZEN.json')
    assert lock['winner_count']==100 and len(lock['winners'])==100 and not lock['query_truth_used']
    assert {(w['dataset'],w['seed'],w['variant']) for w in lock['winners']}=={(ds,seed,variant) for ds in DATASETS for seed in range(42,47) for variant in VARIANTS}
    evaluation_sources(p.SOURCE)
    assert lock['preregistration_sha256']==sha(p.RUN/'PREREGISTERED.json')
    verify(lock['files']);verify(read(p.RUN/'PREREGISTERED.json')['source_input_files'])
    return lock

def checkpoint_prediction(ctx,w):
    c=w['config'];seed_all(ctx.seed)
    if w['legacy']:
        device=torch.device('cpu');e,h=models({**ctx.graph,**ctx.item},device)
    else:e,h=ctx.create(c)
    ck=load(w['checkpoint']);e.load_state_dict(ck['encoder_state']);h.load_state_dict(ck['head_state']);e.eval();h.eval()
    if w['variant'].endswith('frozen'):
        assert all(torch.equal(v,ctx.graph['encoder_state'][k]) for k,v in ck['encoder_state'].items())
    if w['legacy']:
        with torch.no_grad():
            z=ctx.graph['embeddings'] if c['mode']=='frozen' else encode(e,ctx.graph['x'],adjacency(ctx.graph['edge_index'],len(ctx.graph['x'])))
            legacy_logits=h(z);replay=legacy_logits.softmax(1)
        old=load(Path(w['checkpoint']).parent/'probabilities.pt');prob=old['probability']
        assert torch.equal(old['node_ids'],torch.arange(len(prob)))
        compare_replay(prob,replay)
        legacy_nll=torch.nn.functional.cross_entropy(legacy_logits[ctx.item['valid_ids']],ctx.item['valid_targets']).item()
        nll_error=abs(legacy_nll-w['validation']['nll']);assert nll_error<5e-5
    else:
        v=ctx.validate(e,h,c)
        assert v['correct']==w['validation']['correct']
        nll_error=abs(v['nll']-w['validation']['nll']);assert nll_error<5e-5
        with torch.no_grad():
            z=ctx.cached if c['mode']=='frozen' else engine.embeddings(e,ctx.ax,ctx.adj,'frozen')
            prob=h(z).softmax(1).cpu()
            # Independent implementation: recompute A @ raw x using original encode.
            z2=ctx.cached if c['mode']=='frozen' else encode(e,ctx.graph['x'].to(ctx.device),ctx.adj)
            replay=h(z2).softmax(1).cpu()
        compare_replay(prob,replay)
    valid=ctx.item['valid_ids'];vy=ctx.item['valid_targets']
    assert int((prob[valid].argmax(1)==vy).sum())==w['validation']['correct']
    assert torch.isfinite(prob).all() and bool((prob>=0).all())
    torch.testing.assert_close(prob.sum(1),torch.ones(len(prob)),atol=2e-6,rtol=0)
    record={'dataset':ctx.item['dataset'],'seed':ctx.seed,'variant':w['variant'],'candidate':w['id'],
            'legacy':w['legacy'],'max_probability_error':float((replay-prob).abs().max()),'validation_nll_error':nll_error,'all_node_classes_identical':True,'probability_atol':5e-5}
    return prob,record

def paired_record(local,ds,seed,pop,mode):
    rec={'dataset':ds,'split_seed':seed,'population':pop,'strategy':mode}
    for label,method in (('joint_fixed','joint_fixed'),('router','router_'+mode),('gold','gold_'+mode)):
        row=next(r for r in local if r['method']==method)
        for metric in ('accuracy','macro_f1'):rec[label+'_'+metric]=row[metric]
    for metric in ('accuracy','macro_f1'):
        rec['router_minus_joint_'+metric]=rec['router_'+metric]-rec['joint_fixed_'+metric]
        rec['router_minus_gold_'+metric]=rec['router_'+metric]-rec['gold_'+metric]
        rec['gold_minus_joint_'+metric]=rec['gold_'+metric]-rec['joint_fixed_'+metric]
    return rec

def format_report(rows,groups,lock,configs):
    def fmt(pop,ds,method,metric):
        r=next(r for r in groups if (r['population'],r['dataset'],r['method'])==(pop,ds,method))
        return f"{100*r[metric+'_mean']:.2f} ± {100*r[metric+'_std_ddof0']:.2f}"
    lines=['# Router-v9-new：GNN 两种策略调优后的五种子对照','',
        f"完成 {lock['trained_trials']} 次新候选训练，另保留100个历史同划分基线候选；最终冻结100个模型选择结果。每组训练 seed＝划分 seed（42–46），无内层集成。",'',
        '所有表格均为百分数，五种子均值 ± 总体标准差（ddof=0）；CSV同时保存样本标准差（ddof=1）。增益单位为百分点。冻结与微调分别报告，不根据测试成绩在两种策略中择优。','',
        'Joint-Fixed 使用 Router-v9-new 对应的历史固定实现（三成员集成），来源 `runs/router_v9_new_joint_multiseed_20260912_v2`；GNN每划分为单模型。两者输入与模型结构不同，并非仅替换分类器的完全受控消融。','',
        '调优按预先固定的分阶段候选范围展开：先搜索分类头学习率/伪标签权重，再单因素测试正则化、权重渐增和1000轮上限；微调继承本监督分支选中的分类头超参数，比较4种骨架学习率×3种解冻方式，再做单因素精调。每个新候选均从原始SSL骨架和对应原分类头开始；gold-only从未继承Router训练权重。原配置的历史checkpoint也仅凭验证指标参与竞争。','',
        '验证集 Accuracy 优先，NLL用于平局，随后优先更早epoch和确定性候选ID；初始权重可被选中。前20轮逐轮验证，之后每5轮验证，100轮无改善早停。第二轮仍使用同一验证集，存在验证集反复选择的适应性；本轮没有用query真值选配置、epoch、阈值或模型。','',
        '**数据边界：**训练仅读取安全导出的support真值、冻结Router伪标签和用于选模的validation标签；整个图的无标签特征与边参与传导式GNN计算。全部100个选择记录封存后才载入query真值。eval1000是同一query的固定子集，可参与无真值的伪标签训练。Arxiv query不是官方test，Products使用54,025节点子图。','',
        '**继承与计算说明：**这是已查看历史query结果后的后续调优；Router-v9-new有历史posthoc query选择，SSL配置曾根据历史seed42 validation选择，因此不是新的独立盲测。新候选使用Slurm分配的GPU、float32且禁用TF32，head仅前向监督/验证子集，故dropout随机数消费与历史CPU全图head前向不同；设备型号逐候选记录。源模型和历史产物保持原有哈希。','']
    for pop in ('full_query','eval1000'):
        for mode,label in (('frozen','冻结骨架，仅训练分类头'),('finetune','骨架与分类头微调')):
            lines += [f'## {pop} · {label}','','| 数据集 | Joint-Fixed Acc | Joint-Fixed F1 | Router＋GNN Acc | Router＋GNN F1 | gold-only Acc | gold-only F1 |','|---|---:|---:|---:|---:|---:|---:|']
            for ds in DATASETS:
                values=[fmt(pop,ds,m,k) for m in ('joint_fixed','router_'+mode,'gold_'+mode) for k in ('accuracy','macro_f1')]
                lines.append('| '+ds+' | '+' | '.join(values)+' |')
            lines.append('')
        lines += [f'## {pop} · 固定伪标签质量','','两种GNN策略和Joint-Fixed使用同一Router接受集合；gold-only不使用伪标签。','','| 数据集 | 伪标签准确率 | 覆盖率 |','|---|---:|---:|']
        for ds in DATASETS:lines.append('| '+ds+' | '+fmt(pop,ds,'router_frozen','pseudo_accuracy')+' | '+fmt(pop,ds,'router_frozen','pseudo_coverage')+' |')
        lines.append('')
        lines += [f'## {pop} · 逐 seed 配对增益','','Δ为Router＋GNN减去对照，单位为百分点。原始三方逐seed指标在CSV中完整保留。','','| 数据集 | Seed | 策略 | ΔAcc vs Joint | ΔF1 vs Joint | ΔAcc vs gold | ΔF1 vs gold |','|---|---:|---|---:|---:|---:|---:|']
        for r in rows:
            if r['population']==pop and r['method'].startswith('router'):
                vals=[r[k] for k in ('delta_joint_accuracy','delta_joint_macro_f1','delta_gold_accuracy','delta_gold_macro_f1')]
                lines.append(f"| {r['dataset']} | {r['split_seed']} | {r['method']} | "+' | '.join(f'{v*100:+.2f}' for v in vals)+' |')
        lines.append('')
    lines += ['## 产物与审计','', 'GPU稀疏计算重复执行出现约1e-5量级概率波动。v1严格重放检查失败并保留；通过冻结checkpoint的重复输出及原验证集正确数诊断（不访问query真值）后，最终评价要求概率绝对差≤5e-5且所有节点类别完全一致，验证正确数仍须与封存记录相同。未重新训练或选择模型。v2完成数值计算后遇到配对CSV列名不一致；v3统一列名，并保留前两次评价产物。', '',
        '- [五种子均值与两种标准差](evaluation/mean_std.csv)',
        '- [完整逐 seed 指标](evaluation/per_seed.csv)',
        '- [三方并列逐 seed 与配对增益](evaluation/paired_gains.csv)',
        '- [最终配置与搜索次数](evaluation/selected_configs.csv)',
        '- [相对上一轮的变化](evaluation/change_from_previous.csv)',
        '- [独立数值与checkpoint审计](FINAL_AUDIT.json)',
        '- 搜索协议：CONFIG.json；源代码与输入封存：PREREGISTERED.json；全局选模封存：FROZEN.json。','']
    return '\n'.join(lines)

def evaluate():
    lock=pretest();torch.set_num_threads(2);seed_all(42)
    write(OUTPUT/'EVALUATION_STARTED.json',{'frozen_sha256':sha(p.RUN/'FROZEN.json'),'evaluator_sha256':sha(Path(__file__)),'diagnosis_sha256':sha(p.RUN/'REPLAY_DIAGNOSIS.json')})
    old_joint=list(csv.DictReader((JOINT/'joint/evaluation/per_split.csv').open()))
    rows=[];replays=[];prediction_files={};configs=[];paired=[]
    for ds in DATASETS:
        y=truth(ds);graph=load(p.SOURCE/f'inputs/{ds}/graph.pt')
        for seed in range(42,47):
            item={**load(p.SOURCE/f'inputs/{ds}/seed{seed}.pt'),'dataset':ds};ctx=engine.Context(graph,item,seed)
            joint=load(JOINT/f'seed{seed}/joint/r9_new/{ds}/predictions.pt')
            assert torch.equal(joint['node_ids'],torch.arange(len(y)))
            probs={'joint_fixed':joint['families']['optimized']['ensemble'].numpy()}
            for variant in VARIANTS:
                w=next(w for w in lock['winners'] if (w['dataset'],w['seed'],w['variant'])==(ds,seed,variant))
                prob,replay=checkpoint_prediction(ctx,w);replays.append(replay);probs[variant]=prob.numpy()
                dest=OUTPUT/f'predictions/seed{seed}/{ds}/{variant}.pt'
                if dest.exists():
                    saved=load(dest);assert torch.equal(saved['node_ids'],torch.arange(len(prob)))
                    replay['resume_probability_error']=compare_replay(saved['probability'],prob)
                    prob=saved['probability']
                else:save(dest,{'node_ids':torch.arange(len(prob)),'probability':prob})
                probs[variant]=prob.numpy()
                prediction_files[str(dest)]=sha(dest)
                selection=read(p.RUN/f'cells/seed{seed}/{ds}/{variant}/SELECTION.json')
                configs.append({'dataset':ds,'seed':seed,'variant':variant,'candidate_id':w['id'],'legacy':w['legacy'],
                    'selected_epoch':w['selected_epoch'],'validation_accuracy':w['validation']['accuracy'],'validation_nll':w['validation']['nll'],
                    'trained_candidates':selection['trained_count'],**w['config']})
            pseudo=np.full(len(y),-1,dtype=np.int64);pseudo[item['pseudo_ids'].numpy()]=item['pseudo_targets'].numpy()
            source_router=load(item['source_router']);accept=source_router['accepted_mask']
            assert torch.equal(source_router['node_ids'][accept],item['pseudo_ids']) and torch.equal(source_router['pseudo_label'][accept],item['pseudo_targets'])
            for pop,key in (('full_query','query_ids'),('eval1000','eval1000_ids')):
                ids=item[key].numpy();accepted=pseudo[ids]>=0;local=[]
                for method in METHODS:
                    prob=probs[method];pred=prob[ids].argmax(1);nc=prob.shape[1];uses=method=='joint_fixed' or method.startswith('router')
                    a=float(accuracy_score(y[ids],pred));f=float(f1_score(y[ids],pred,labels=np.arange(nc),average='macro',zero_division=0))
                    ac,fc=confusion_metrics(y[ids],pred,nc);assert abs(a-ac)<1e-12 and abs(f-fc)<1e-12
                    local.append({'dataset':ds,'split_seed':seed,'population':pop,'method':method,'query_count':len(ids),
                        'accepted_count':int(accepted.sum()) if uses else None,
                        'pseudo_accuracy':float(np.mean(pseudo[ids][accepted]==y[ids][accepted])) if uses and accepted.any() else None,
                        'pseudo_coverage':float(accepted.mean()) if uses else None,'accuracy':a,'macro_f1':f})
                ref=next(r for r in old_joint if r['dataset']==ds and int(r['split_seed'])==seed and r['population']==pop and r['router']=='r9_new')
                for k in ('accuracy','macro_f1','pseudo_accuracy','pseudo_coverage'):assert abs(local[0][k]-float(ref[k]))<1e-10
                for r in local:
                    gold=next((q for q in local if r['method'].startswith('router') and q['method']==r['method'].replace('router_','gold_')),None)
                    for metric in ('accuracy','macro_f1'):
                        r['delta_joint_'+metric]=r[metric]-local[0][metric]
                        r['delta_gold_'+metric]=r[metric]-gold[metric] if gold else None
                rows.extend(local)
                for mode in ('frozen','finetune'):
                    paired.append(paired_record(local,ds,seed,pop,mode))
            print('EVALUATED',ds,seed,flush=True)
    groups=aggregate(rows)
    for g in groups:
        local=[r for r in rows if all(r[k]==g[k] for k in ('population','dataset','method'))]
        assert sorted(r['split_seed'] for r in local)==list(range(42,47))
        for metric in METRICS:
            vals=[r[metric] for r in local if r[metric] is not None]
            if not vals:assert g[metric+'_mean'] is None;continue
            for suffix,v in zip(('mean','std_ddof0','std_ddof1'),moments(vals)):assert abs(v-g[metric+'_'+suffix])<1e-12
    old=read(p.SOURCE/'reports/v2/evaluation/results.json')['aggregate'];changes=[]
    for g in groups:
        if g['method']=='joint_fixed':continue
        previous=next(r for r in old if all(r[k]==g[k] for k in ('population','dataset','method')))
        changes.append({**{k:g[k] for k in ('population','dataset','method')},
            **{f'delta_{k}_mean':g[k+'_mean']-previous[k+'_mean'] for k in ('accuracy','macro_f1')}})
    assert len(rows)==250 and len(groups)==50 and len(replays)==100 and len(paired)==100
    write(OUTPUT/'evaluation/results.json',{'rows':rows,'aggregate':groups})
    for name,records in (('per_seed',rows),('mean_std',groups),('paired_gains',paired),('selected_configs',configs),('change_from_previous',changes)):
        csv_write(OUTPUT/f'evaluation/{name}.csv',records)
    report=format_report(rows,groups,lock,configs);dest=OUTPUT/'REPORT.md'
    if dest.exists():assert dest.read_text()==report
    else:dest.write_text(report)
    verify(lock['files']);verify(read(p.RUN/'PREREGISTERED.json')['source_input_files']);p.gate()
    extra_sources=evaluation_sources(p.SOURCE)
    write(OUTPUT/'FINAL_AUDIT.json',{'all_pass':True,'checkpoint_replays':replays,'metric_checks':500,'mean_std_groups':50,
        'per_seed_rows':250,'paired_rows':100,'trained_trials':lock['trained_trials'],'source_and_training_hashes_unchanged':True,'replay_absolute_tolerance':5e-5,'all_replayed_node_classes_identical':True,'historical_evaluation_source_files':extra_sources,
        'query_truth_only_after_global_freeze':True,'slurm_job_id':os.environ['SLURM_JOB_ID'],'predictions':prediction_files,
        'results_sha256':sha(OUTPUT/'evaluation/results.json'),'evaluator_sha256':sha(Path(__file__)),'diagnosis_sha256':sha(p.RUN/'REPLAY_DIAGNOSIS.json')})
    artifacts=[OUTPUT/'REPORT.md',OUTPUT/'FINAL_AUDIT.json',p.RUN/'FROZEN.json',p.RUN/'PREREGISTERED.json']+list((OUTPUT/'evaluation').glob('*'))
    write(OUTPUT/'DELIVERY.json',{'complete':True,'files':{str(f):sha(f) for f in artifacts},'slurm_job_id':os.environ['SLURM_JOB_ID']})
    print('FINAL AUDIT PASSED; REPORT COMPLETE',flush=True)

if __name__=='__main__':evaluate()
