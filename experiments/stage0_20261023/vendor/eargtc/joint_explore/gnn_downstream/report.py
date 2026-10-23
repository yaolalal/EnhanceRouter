"""Post-freeze evaluation and independent confusion-matrix/checkpoint audit."""
import csv,io as textio
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from .io import *
from .train import gate
from .core import encode,adjacency
METHODS=('joint_fixed',)+VARIANTS
NAMES={'joint_fixed':'Joint-Fixed','router_frozen':'Router＋GNN（冻结骨架）',
       'router_finetune':'Router＋GNN（微调骨架）','gold_frozen':'gold-only GNN（冻结骨架）','gold_finetune':'gold-only GNN（微调骨架）'}
METRICS=('accuracy','macro_f1','pseudo_accuracy','pseudo_coverage','delta_joint_accuracy','delta_joint_macro_f1','delta_gold_accuracy','delta_gold_macro_f1')

def pretest():
    require();gate();lock=read(RUN/'FROZEN.json')
    assert lock['cell_count']==100 and len(lock['cells'])==100 and not lock['query_targets_accessed']
    verify(lock['files']);verify(read(RUN/'INPUT_AUDIT.json')['files'])
    return lock

def truth(ds):
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    split=read(UP/f'splits/{ds}_5shot_seed42.json');p=Path(split['graph_path'])
    verify({p:split['graph_sha256']})
    data=torch.load(p,map_location='cpu',weights_only=False)
    y=data.y.reshape(-1).long().numpy();assert len(y)==split['num_nodes']
    return y

def probabilities(ds,seed,method):
    if method=='joint_fixed':
        p=load(JOINT/f'seed{seed}/joint/r9_new/{ds}/predictions.pt')
        prob=p['families']['optimized']['ensemble']
    else:
        p=load(RUN/f'cells/seed{seed}/{ds}/{method}/probabilities.pt');prob=p['probability']
    assert torch.equal(p['node_ids'],torch.arange(len(prob)))
    assert torch.isfinite(prob).all() and bool((prob>=0).all())
    torch.testing.assert_close(prob.sum(1),torch.ones(len(prob)),atol=2e-6,rtol=0)
    return prob.numpy()

def aggregate(rows):
    out=[]
    for pop in ('full_query','eval1000'):
        for ds in DATASETS:
            for method in METHODS:
                group=[r for r in rows if (r['population'],r['dataset'],r['method'])==(pop,ds,method)]
                assert sorted(r['split_seed'] for r in group)==list(range(42,47))
                rec={'population':pop,'dataset':ds,'method':method,'seeds':5}
                for metric in METRICS:
                    values=[r[metric] for r in group if r[metric] is not None]
                    assert len(values) in (0,5)
                    for suffix,value in (('mean',np.mean(values) if values else None),('std_ddof0',np.std(values) if values else None),('std_ddof1',np.std(values,ddof=1) if values else None)):
                        rec[metric+'_'+suffix]=None if value is None else float(value)
                out.append(rec)
    return out

def csv_write(path,rows):
    buf=textio.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==buf.getvalue()
    else:
        with path.open('x') as f:f.write(buf.getvalue())

def markdown(rows,groups):
    lines=['# Router-v9-new：继续训练 SSL-GNN 的五种子实验','',
      '本轮继续训练的 seed 与划分 seed 一致（42–46）；继承分类头的原始训练 seed 单独保存在输入记录中；每组一个模型，无内层集成。历史 Joint-Fixed 为原三成员集成结果，仅作历史对照。', '',
      '每个划分从同一份已有 SSL 骨架和对应分类头独立开始，比较冻结/微调骨架及 Router/gold-only 监督。骨架 lr=1e-4，分类头 lr=1e-3；AdamW、weight_decay=1e-4；最多500轮，每5轮验证，连续20次无改善早停。初始权重也可被 validation 选中。', '',
      'Router 接受的 agreement/disagreement 伪标签均以单位权重参与 CE，support 与伪标签池分别平均；gold-only 不使用伪标签。所有数值计算经 Slurm，四组继续训练统一使用 CPU、每模型2线程。', '',
      '**数据边界：**本轮训练只访问导出的 support 标签和冻结伪标签，validation 只用于选择；全部100组冻结后才评价 query 真值。原始图容器在输入导出时被反序列化，立即只保留特征和边，不访问其 query 标签值。SSL 预训练目标无标签，但共享骨架配置曾在历史 seed42 validation 上选择；Router-v9-new 有 posthoc query 成绩选择历史。因此这不是全新盲测，也不声称整个继承流程不存在跨划分选择信息。', '',
      'full-query 与 eval1000 使用同一冻结模型；eval1000 是 query 子集，可参与无真值伪标签训练。Arxiv query 不是官方 test；Products 是54,025节点子图。表中均为百分数，均值±总体标准差(ddof=0)，CSV另存ddof=1；增益为百分点。gold-only 的伪标签指标不适用。', '']
    for pop in ('full_query','eval1000'):
        for metric,title in (('accuracy','Accuracy'),('macro_f1','Macro-F1'),('pseudo_accuracy','伪标签准确率'),('pseudo_coverage','伪标签覆盖率'),('delta_joint_accuracy','Accuracy 相对 Joint-Fixed 增益'),('delta_joint_macro_f1','Macro-F1 相对 Joint-Fixed 增益'),('delta_gold_accuracy','Accuracy 相对对应 gold-only 增益'),('delta_gold_macro_f1','Macro-F1 相对对应 gold-only 增益')):
            lines += [f'## {pop}：{title}','','| 方法 | '+' | '.join(DATASETS)+' |','|---|'+'---:|'*5]
            for method in METHODS:
                values=[]
                for ds in DATASETS:
                    rec=next(r for r in groups if (r['population'],r['dataset'],r['method'])==(pop,ds,method))
                    mean,sd=rec[metric+'_mean'],rec[metric+'_std_ddof0']
                    values.append('—' if mean is None else f'{100*mean:.2f} ± {100*sd:.2f}')
                lines.append('| '+NAMES[method]+' | '+' | '.join(values)+' |')
            lines.append('')
        lines += [f'## {pop}：逐 seed 分类与增益','','| 数据集 | Seed | 方法 | Acc | F1 | ΔAcc Joint | ΔF1 Joint | ΔAcc gold | ΔF1 gold |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['population']!=pop:continue
            metrics=[row[k] for k in ('accuracy','macro_f1','delta_joint_accuracy','delta_joint_macro_f1','delta_gold_accuracy','delta_gold_macro_f1')]
            lines.append('| '+f"{row['dataset']} | {row['split_seed']} | {NAMES[row['method']]} | "+' | '.join('—' if v is None else f'{v*100:.2f}' for v in metrics)+' |')
        lines.append('')
    lines += ['详表：[逐 seed CSV](evaluation/per_seed.csv)、[均值与标准差 CSV](evaluation/mean_std.csv)。', '',
              '完整性证据：INPUT_AUDIT.json、PREREGISTERED.json、FROZEN.json、FINAL_AUDIT.json。历史权重和原始数据为哈希绑定的外部依赖。']
    return '\n'.join(lines)+'\n'

def evaluate():
    pretest()
    write(RUN/'EVALUATION_STARTED.json',{'frozen_sha256':sha(RUN/'FROZEN.json')})
    old=list(csv.DictReader((JOINT/'joint/evaluation/per_split.csv').open()))
    rows=[]
    for ds in DATASETS:
        y=truth(ds)
        for seed in range(42,47):
            item=load(RUN/f'inputs/{ds}/seed{seed}.pt')
            pseudo=np.full(len(y),-1,dtype=np.int64);pseudo[item['pseudo_ids'].numpy()]=item['pseudo_targets'].numpy()
            preds={m:probabilities(ds,seed,m) for m in METHODS}
            for pop,key in (('full_query','query_ids'),('eval1000','eval1000_ids')):
                ids=item[key].numpy();accepted=pseudo[ids]>=0;pa=float(np.mean(pseudo[ids][accepted]==y[ids][accepted])) if accepted.any() else None
                local=[]
                for method in METHODS:
                    pred=preds[method];nc=pred.shape[1];uses=method=='joint_fixed' or method.startswith('router')
                    row={'dataset':ds,'split_seed':seed,'population':pop,'method':method,'query_count':len(ids),
                         'accepted_count':int(accepted.sum()) if uses else None,
                         'pseudo_accuracy':pa if uses else None,'pseudo_coverage':float(accepted.mean()) if uses else None,
                         'accuracy':float(accuracy_score(y[ids],pred[ids].argmax(1))),
                         'macro_f1':float(f1_score(y[ids],pred[ids].argmax(1),labels=np.arange(nc),average='macro',zero_division=0))}
                    local.append(row)
                joint=local[0]
                ref=next(r for r in old if r['dataset']==ds and int(r['split_seed'])==seed and r['population']==pop and r['router']=='r9_new')
                for k in ('accuracy','macro_f1','pseudo_accuracy','pseudo_coverage'):assert abs(joint[k]-float(ref[k]))<1e-10,(ds,seed,k)
                for row in local:
                    gold=next((r for r in local if r['method']==row['method'].replace('router_','gold_')),None)
                    for metric in ('accuracy','macro_f1'):
                        row['delta_joint_'+metric]=row[metric]-joint[metric]
                        row['delta_gold_'+metric]=row[metric]-gold[metric] if gold else None
                rows.extend(local)
    groups=aggregate(rows)
    write(RUN/'evaluation/results.json',{'rows':rows,'aggregate':groups})
    csv_write(RUN/'evaluation/per_seed.csv',rows);csv_write(RUN/'evaluation/mean_std.csv',groups)
    text=markdown(rows,groups)
    p=RUN/'REPORT.md'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    write(RUN/'evaluation/COMPLETE.json',{'files':{str(p):sha(p) for p in (RUN/'evaluation/results.json',RUN/'evaluation/per_seed.csv',RUN/'evaluation/mean_std.csv',RUN/'REPORT.md')}})
    print('Evaluated',len(rows),'rows',len(groups),'aggregate groups',flush=True)

def audit():
    pretest();verify(read(RUN/'evaluation/COMPLETE.json')['files'])
    results=read(RUN/'evaluation/results.json');rows=results['rows'];assert len(rows)==250
    numerical_checks=0;checkpoint_checks=[]
    for ds in DATASETS:
        y=truth(ds);graph=load(RUN/f'inputs/{ds}/graph.pt')
        for seed in range(42,47):
            item=load(RUN/f'inputs/{ds}/seed{seed}.pt')
            r=load(item['source_router']);ids=r['node_ids'][r['accepted_mask']];pl=r['pseudo_label'][r['accepted_mask']]
            assert torch.equal(ids,item['pseudo_ids']) and torch.equal(pl,item['pseudo_targets'])
            for method in METHODS:
                p=probabilities(ds,seed,method);nc=p.shape[1]
                for pop,key in (('full_query','query_ids'),('eval1000','eval1000_ids')):
                    q=item[key].numpy();cm=np.bincount(y[q]*nc+p[q].argmax(1),minlength=nc*nc).reshape(nc,nc)
                    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(nc),where=den!=0)
                    row=next(r for r in rows if (r['dataset'],r['split_seed'],r['method'],r['population'])==(ds,seed,method,pop))
                    assert abs(row['accuracy']-np.trace(cm)/len(q))<1e-12 and abs(row['macro_f1']-f.mean())<1e-12
                    numerical_checks+=2
                    if method=='joint_fixed' or method.startswith('router'):
                        mask=np.isin(ids.numpy(),q);count=int(mask.sum())
                        assert row['accepted_count']==count
                        assert abs(row['pseudo_coverage']-count/len(q))<1e-12
                        assert abs(row['pseudo_accuracy']-np.mean(pl.numpy()[mask]==y[ids.numpy()[mask]]))<1e-12
                    else:assert row['pseudo_accuracy'] is None and row['pseudo_coverage'] is None
                if method=='joint_fixed':continue
                dest=RUN/f'cells/seed{seed}/{ds}/{method}';meta=read(dest/'metadata.json');history=read(dest/'history.json')
                choices=[(meta['initial_validation']['accuracy'],-meta['initial_validation']['nll'],0)]+[(v['validation_accuracy'],-v['validation_nll'],-v['epoch']) for v in history]
                best=max(choices);assert -best[2]==meta['selected_epoch']
                assert meta['model_seed']==seed and meta['first_step_encoder_updated']==method.endswith('finetune') and meta['first_step_head_updated']
                device=torch.device('cpu');e,h=models({**graph,**item},device)
                ck=load(dest/'checkpoint.pt');e.load_state_dict(ck['encoder_state']);h.load_state_dict(ck['head_state']);e.eval();h.eval()
                if method.endswith('frozen'):assert all(torch.equal(v,graph['encoder_state'][k]) for k,v in ck['encoder_state'].items())
                with torch.no_grad():
                    z=encode(e,graph['x'].to(device),adjacency(graph['edge_index'].to(device),len(y))) if method.endswith('finetune') else graph['embeddings'].to(device)
                    replay=h(z).softmax(1).cpu().numpy()
                assert np.allclose(replay,p,atol=3e-6,rtol=1e-5) and np.array_equal(replay.argmax(1),p.argmax(1))
                checkpoint_checks.append({'dataset':ds,'seed':seed,'method':method,'max_probability_error':float(np.max(np.abs(replay-p)))})
                del e,h,z
            print('Independent audit',ds,seed,flush=True)
    for row in rows:
        group=[r for r in rows if (r['dataset'],r['split_seed'],r['population'])==(row['dataset'],row['split_seed'],row['population'])]
        joint=next(r for r in group if r['method']=='joint_fixed');gold=next((r for r in group if r['method']==row['method'].replace('router_','gold_')),None)
        for k in ('accuracy','macro_f1'):
            assert abs(row['delta_joint_'+k]-(row[k]-joint[k]))<1e-12
            if gold:assert abs(row['delta_gold_'+k]-(row[k]-gold[k]))<1e-12
            else:assert row['delta_gold_'+k] is None
    assert len(results['aggregate'])==50
    for rec in results['aggregate']:
        selected=[r for r in rows if all(r[k]==rec[k] for k in ('population','dataset','method'))]
        assert sorted(r['split_seed'] for r in selected)==list(range(42,47))
        for metric in METRICS:
            vals=[r[metric] for r in selected if r[metric] is not None]
            if not vals:assert rec[metric+'_mean'] is None;continue
            mean=sum(vals)/5;ss=sum((v-mean)**2 for v in vals)
            for suffix,value in (('mean',mean),('std_ddof0',(ss/5)**.5),('std_ddof1',(ss/4)**.5)):
                assert abs(rec[metric+'_'+suffix]-value)<1e-12
    verify(read(RUN/'INPUT_AUDIT.json')['files'])
    write(RUN/'FINAL_AUDIT.json',{'all_pass':True,'metric_checks':numerical_checks,'checkpoint_replays':checkpoint_checks,
         'row_count':250,'aggregate_count':50,'training_cells':100,'protected_and_source_hashes_verified':True,
         'results_sha256':sha(RUN/'evaluation/results.json'),'slurm_job_id':os.environ['SLURM_JOB_ID']})
    print('FINAL AUDIT PASSED',flush=True)
