"""Re-evaluate frozen Router outputs on complete, training-disjoint Query pools.

No fitting, model selection, or deferred B0-budget metric is executed here.
Run `cell INDEX OUTPUT` as a Slurm array, then `aggregate OUTPUT` afterok.
"""
from d3_recovery import *
import csv, hashlib, statistics
from datetime import datetime, timezone, timedelta

AMENDMENT = ROOT/'provenance/full-query-amendment-20261027/AMENDMENT.json'
PRIOR = RUN/'current-results/20261027T134000Z'
VARIANTS = ['b0', *[f'd1_frozen/{f}' for f in (.25,.5,.75,1.)],
            *[f'd1_numeric/{f}' for f in (.25,.5,.75,1.)],
            *[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],
            'd3','d4/confidence_router','d4/confidence_preference']
ROUTER_METRICS = ('trust_auroc','preference_auroc',
                  *[f'{b}_Prec_{p:02d}' for b in ('D','A') for p in (1,5,20,100)],
                  'A_correctness_auroc','deployed_precision')
METRICS = (*ROUTER_METRICS,'accuracy','macro_f1')

def preflight():
    require()
    amendment=read(AMENDMENT);freeze=read(RUN/'CODE_FREEZE.json')
    for rel,expected in freeze['files'].items():
        if rel in amendment['files']:
            r=amendment['files'][rel]
            assert r['original_sha256']==expected
            assert digest(ROOT/r['archived_original'])==expected
            assert digest(ROOT/rel)==r['current_sha256']
        else:
            assert digest(ROOT/rel)==expected,rel
    for rel,r in amendment['files'].items():
        assert digest(ROOT/r['archived_original'])==r['original_sha256']
        assert digest(ROOT/rel)==r['current_sha256']
    assert digest(RUN/'STAGE0_MODELS_READY_20261027.json')=='1341e4df5072bb75562fe791543923872bf6f211ceb51352c29c4612575f0c54'
    for rel,h in read(PRIOR/'COMPLETE.json')['files'].items():
        assert digest(PRIOR/rel)==h,rel
    from enhancerouter.io import guard_writes
    guard_writes()
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',2)))

def stats(values):
    a=[float(v) for v in values if v is not None]
    return dict(n=len(a),mean=statistics.mean(a) if a else None,
                std_ddof0=statistics.pstdev(a) if a else None)

def id_hash(ids):
    import numpy as np
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()

def score_files(path):
    seal_record=read(path/'COMPLETE.json');records={}
    for rel in ('agreement-scores.pt','disagreement-scores.pt','config.json','diagnostics.json','controlled-audit.json'):
        if rel in seal_record['files']:
            assert digest(path/rel)==seal_record['files'][rel],(path,rel)
            records[rel]=seal_record['files'][rel]
    return {b:load(path/f'{b}-scores.pt') for b in ('agreement','disagreement')},dict(
        path=str(path),seal_sha256=digest(path/'COMPLETE.json'),files=records)

def validate_bank(scores,query,g,l):
    import numpy as np
    parts=[]
    for branch,data in scores.items():
        ids=data['ids'].numpy().astype(np.int64,copy=False)
        assert ids.ndim==1 and len(np.unique(ids))==len(ids)
        same=g[ids]==l[ids]
        assert np.all(same if branch=='agreement' else ~same)
        for name in ('score','trust','route','accepted'):
            assert len(data[name])==len(ids),(branch,name)
        assert np.isfinite(data['score'].numpy()).all()
        assert np.isfinite(data['trust'].numpy()).all()
        if branch=='disagreement':assert np.isfinite(data['preference'].numpy()).all()
        parts.append(ids)
    assert np.array_equal(np.sort(np.concatenate(parts)),query),'score bank is not the complete registered Query'

def router_metrics(scores,query,g,l,y):
    import numpy as np
    from report import auc
    result=dict(query_count=len(query),query_ids_sha256=id_hash(query))
    deployed_total=deployed_correct=0
    for branch,prefix in (('agreement','A'),('disagreement','D')):
        d=scores[branch];raw=d['ids'].numpy().astype(np.int64,copy=False)
        # sorted-ID search avoids an all-pairs membership allocation on large graphs.
        loc=np.searchsorted(query,raw);mask=loc<len(query)
        mask[mask]=query[loc[mask]]==raw[mask]
        nodes=raw[mask];score=d['score'].numpy()[mask]
        route=d['route'].numpy()[mask]
        assert np.isin(route,(0,1)).all()
        pred=np.where(route==0,g[nodes],l[nodes]) if prefix=='D' else g[nodes]
        correct=pred==y[nodes];order=np.lexsort((nodes,-score))
        result[branch+'_count']=len(nodes)
        for pct in (1,5,20,100):
            k=(pct*len(nodes)+99)//100
            result[f'{prefix}_k_{pct:02d}']=k
            result[f'{prefix}_Prec_{pct:02d}']=float(correct[order[:k]].mean()) if k else None
        accepted=d['accepted'].numpy()[mask].astype(bool)
        deployed_total+=int(accepted.sum());deployed_correct+=int(correct[accepted].sum())
        result[prefix+'_deployed_count']=int(accepted.sum())
        result[prefix+'_deployed_precision']=float(correct[accepted].mean()) if accepted.any() else None
        if prefix=='D':
            trust=(g[nodes]==y[nodes])|(l[nodes]==y[nodes]);pref=g[nodes]==y[nodes]
            result.update(trust_positive_count=int(trust.sum()),
                          preference_population_count=int(trust.sum()),
                          preference_gnn_positive_count=int(pref[trust].sum()),
                          trust_auroc=auc(trust,d['trust'].numpy()[mask]),
                          preference_auroc=auc(pref[trust],d['preference'].numpy()[mask][trust]))
        else:result['A_correctness_auroc']=auc(correct,d['trust'].numpy()[mask])
    assert result['agreement_count']+result['disagreement_count']==len(query)
    result.update(deployed_count=deployed_total,deployed_precision=deployed_correct/deployed_total if deployed_total else None)
    return result

def cell(index,out):
    preflight()
    import numpy as np
    sh,se,ds=CELLS[index];sp=split(sh,se,ds)
    q=np.sort(np.asarray(sp['unlabeled_ids'],dtype=np.int64))
    assert np.array_equal(q,np.sort(sp['evaluation_full_ids']))
    assert len(q)==len(np.unique(q))
    assert not set(q)&set(sp['support_ids']+sp['valid_ids'])
    eval_ids=evaluation_ids(se,ds);assert len(eval_ids)==1000 and set(eval_ids)<=set(q)
    truth=graph(ds).y.reshape(-1).long().numpy()
    up=upstream_bank(sh,se,ds);g=up['gnn_logits'].argmax(1).numpy();l=up['llm_logits'].argmax(1).numpy();del up
    base_scores,base_source=score_files(folder(sh,se,ds)/'b0');validate_bank(base_scores,q,g,l)
    base_metrics=router_metrics(base_scores,q,g,l,truth)
    prior=read(PRIOR/'RESULTS.json')
    final_index={(r['key'],r['variant']):r for r in prior['per_cell']+prior['completed_L2']}
    rows=[]
    for variant in VARIANTS:
        path=variant_folder(sh,se,ds,variant.replace('d1_frozen/','d1/'))
        scores,source=(base_scores,base_source) if variant=='b0' else score_files(path)
        population=q;vg,vl=g,l;excluded=[];paired_query=q;scope='complete original Query'
        if variant=='d3':
            small=read(folder(sh,se,ds)/'small_native/split.json')
            population=np.sort(np.asarray(small['unlabeled_ids'],dtype=np.int64))
            assert not set(population)&set(small['support_ids']+small['valid_ids'])
            assert set(q)<=set(population) and set(eval_ids)<=set(population)
            up=upstream_bank(sh,se,ds,True)
            vg=up['gnn_logits'].argmax(1).numpy();vl=up['llm_logits'].argmax(1).numpy();del up
            scope='complete D3 small-validation Query; matched comparison uses original Query'
            validate_bank(scores,population,vg,vl)
        else:
            validate_bank(scores,q,g,l)
        if variant.startswith('d2'):
            cfg=read(path/'config.json');diag=read(path/'diagnostics.json')
            excluded=cfg['parameters']['parents']
            assert len(excluded)==len(set(excluded))==diag['actual_N']
            assert set(excluded)<=set(q) and not set(excluded)&set(eval_ids)
            population=np.setdiff1d(q,np.asarray(excluded,dtype=np.int64),assume_unique=True)
            assert len(population)==len(q)-len(excluded)
            assert not set(population)&set(excluded)
            paired_query=population;scope='complete original Query minus this D2 added-training parent set'
        metrics=base_metrics if variant=='b0' else router_metrics(scores,population,vg,vl,truth)
        matched_base=base_metrics if variant not in ('d3',) and not variant.startswith('d2') else router_metrics(base_scores,paired_query,g,l,truth)
        paired_variant=metrics if variant!='d3' else router_metrics(scores,paired_query,vg,vl,truth)
        assert paired_variant['query_ids_sha256']==matched_base['query_ids_sha256']
        r=dict(key=key(sh,se,ds),shot=sh,seed=se,dataset=ds,variant=variant,scope=scope,
               **metrics,paired_variant_metrics=paired_variant,matched_B0=matched_base,
               removed_training_parent_ids=excluded,removed_training_parent_count=len(excluded),
               original_query_count=len(q),original_query_ids_sha256=id_hash(q),
               final_eval_count=1000,final_eval_ordered_ids_sha256=id_hash(eval_ids),source=source)
        previous=final_index.get((r['key'],variant))
        if previous is not None:
            r.update({m:previous[m] for m in ('accuracy','macro_f1') if m in previous})
        if variant.startswith('d2'):
            r.update(requested_N=diag['requested_N'],actual_N=diag['actual_N'],extra_T_rows=diag['extra_T_rows'],extra_P_rows=diag['extra_P_rows'])
        if variant=='d1_frozen/1.0':
            for m in ROUTER_METRICS:assert r[m]==base_metrics[m],(r['key'],m)
        rows.append(r)
        print('FULL_QUERY_METRICS',r['key'],variant,len(population),flush=True)
        if variant!='b0':del scores
    target=out/'cells'/f'{index:02d}'
    write(target/'RESULTS.json',dict(index=index,key=key(sh,se,ds),per_cell=rows,
          all_checks_pass=True,report_code_sha256=digest(__file__),amendment_sha256=digest(AMENDMENT),
          old_code_freeze_sha256=digest(RUN/'CODE_FREEZE.json'),prior_complete_sha256=digest(PRIOR/'COMPLETE.json'),
          slurm_job=os.environ['SLURM_JOB_ID'],classification_reused_unchanged=True,models_retrained=False))
    seal(target,dict(all_checks_pass=True,complete_query_metrics=True,rows=len(rows)))

def aggregate(out):
    preflight();rows=[];proof=[]
    for i in range(75):
        p=out/'cells'/f'{i:02d}';record=read(p/'COMPLETE.json')
        for rel,h in record['files'].items():assert digest(p/rel)==h
        d=read(p/'RESULTS.json');assert d['all_checks_pass'] and d['report_code_sha256']==digest(__file__)
        assert d['index']==i and len(d['per_cell'])==18
        rows.extend(d['per_cell']);proof.append(dict(index=i,seal_sha256=digest(p/'COMPLETE.json'),slurm_job=d['slurm_job']))
    assert len(rows)==1350
    summary=[]
    for ds in ('ALL',)+DS:
        for v in VARIANTS:
            group=[r for r in rows if r['variant']==v and (ds=='ALL' or r['dataset']==ds)]
            assert len(group)==(75 if ds=='ALL' else 15)
            s=dict(dataset=ds,variant=v,cells=len(group),
                   query_count_min=min(r['query_count'] for r in group),query_count_max=max(r['query_count'] for r in group),
                   metrics={m:stats([r.get(m) for r in group]) for m in METRICS},
                   paired_variant={m:stats([r['paired_variant_metrics'].get(m) for r in group]) for m in ROUTER_METRICS},
                   matched_B0={m:stats([r['matched_B0'].get(m) for r in group]) for m in ROUTER_METRICS},
                   paired_delta_pp={m:stats([100*(r['paired_variant_metrics'][m]-r['matched_B0'][m]) for r in group if r['paired_variant_metrics'].get(m) is not None and r['matched_B0'].get(m) is not None]) for m in ROUTER_METRICS})
            summary.append(s)
    data=dict(created_at_shanghai=datetime.now(timezone(timedelta(hours=8))).isoformat(),
              scope='Router full Query; D2 excludes added-training parents; D3 paired subset distinguished; final classification ordered eval1000',
              per_cell=rows,summary=summary,cell_proofs=proof,
              metrics=dict(Trust='AUROC on disagreement nodes, target: at least one expert is correct',
                           Preference='AUROC only on disagreement nodes with at least one correct expert, positive: GNN correct',
                           D_Prec='Sort disagreement ranking score descending then node ID ascending; routed expert correctness; k=ceil(percent*disagreement count)',
                           A_Prec='Sort agreement score descending then node ID ascending; common expert prediction correctness; k=ceil(percent*agreement count)',
                           A_correctness_auroc='Agreement reliability AUROC, supplemental; distinct from disagreement Trust AUROC',
                           deployed_precision='Correctness of unchanged model accepted nodes within the evaluation pool; its own acceptance budget, not a matched B0 budget'),
              primary_D_Prec_at_B0=None,A_Prec_at_B0=None,L1_B0_budget='deferred at user request',
              models_retrained=False,classification_reused_unchanged=True,Stage0_complete=False,Stage1_started=False,
              source=dict(report_code_sha256=digest(__file__),amendment_sha256=digest(AMENDMENT),
                          old_code_freeze_sha256=digest(RUN/'CODE_FREEZE.json'),prior_complete_sha256=digest(PRIOR/'COMPLETE.json'),
                          global_gate_sha256=digest(RUN/'STAGE0_MODELS_READY_20261027.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    write(out/'RESULTS.json',data)
    for filename,records in [('per_cell.csv',rows),('summary.csv',[
            dict(dataset=s['dataset'],variant=s['variant'],cells=s['cells'],query_count_min=s['query_count_min'],query_count_max=s['query_count_max'],
                 **{m:s['metrics'][m]['mean'] for m in METRICS},**{m+'_defined':s['metrics'][m]['n'] for m in METRICS}) for s in summary])]:
        fields=sorted({k for r in records for k,v in r.items() if not isinstance(v,(dict,list))})
        with (out/filename).open('x',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    def fmt(x,is_auc=False):return '—' if x is None else f'{x:.4f}' if is_auc else f'{100*x:.2f}'
    cols=('trust_auroc','preference_auroc','D_Prec_01','D_Prec_05','D_Prec_20','D_Prec_100','A_Prec_20')
    lines=['# 全量 Query Router 指标更正报告','',
           '时间（Asia/Shanghai）：'+data['created_at_shanghai'],'',
           '全部1350份模型输出已完成重新核验与计算：B0 75、旧冻结D1 300、新数值重调D1 300、D2 450、D3 75、D4 150。没有重新训练模型。旧报告在eval1000上计算的Router指标由本报告取代。最终分类继续沿用原始顺序固定的eval1000，数值未改。','',
           'Router使用各配置完整Query，排除support/validation。D2另剔除该配置实际新增训练父节点，匹配B0也剔除同一名单。D3完整Query包含缩小验证集后释放的节点；与B0配对时仅使用双方共同的原Query，下面单列。每组节点数、实际k、名单SHA256、训练节点排除名单、输入路径与哈希均保存在RESULTS.json。','',
           'Trust/Preference AUROC定义保持原实现：Trust在分歧节点判断至少一位专家正确；Preference仅在其中至少一位正确的节点判断GNN是否正确。D-P@k在分歧分支按评分排序后检验所选专家，A-P@k在一致分支检验共同预测。k分别为各分支节点总数的1%、5%、20%、100%向上取整，非固定eval1000的比例。并列分数按节点ID升序。AUROC为0–1，其余精度为%。各组等权平均，非将所有图节点混在一起；无定义AUROC不计入均值，有效组数见JSON/CSV。','',
           'D-Prec@B0/A-Prec@B0的原预算争议继续暂缓；没有执行接纳判断或阶段1。历史B0的5-shot选择曾使用Query真值，该历史限制仍保留。','']
    for ds in ('ALL',)+DS:
        lines += ['## '+ds+'：完整 Query','',
                  '| 配置 | 组数 | Query节点数范围 | Trust | Pref | D-P@1% | D-P@5% | D-P@20% | D-P@100% | A-P@20% |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s in summary:
            if s['dataset']!=ds:continue
            values=[fmt(s['metrics'][m]['mean'],m.endswith('auroc')) for m in cols]
            lines.append('| '+s['variant']+' | '+str(s['cells'])+' | '+str(s['query_count_min'])+'–'+str(s['query_count_max'])+' | '+' | '.join(values)+' |')
        lines.append('')
    lines += ['## D2 / D3：相同节点名单上的配对比较','',
              '每行先列B0与变体的Trust/Pref/D-P@20%，最后列D-P@20%的配对平均差（百分点）。D2名单随arm/N/seed变化，不能用上表未剔除节点的B0均值替代。D3此处使用原Query，区别于上表D3完整Query。','',
              '| 数据集 | 配置 | B0 Trust | 变体Trust | B0 Pref | 变体Pref | B0 D-P@20% | 变体D-P@20% | 差值pp |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        if not (s['variant'].startswith('d2') or s['variant']=='d3'):continue
        vals=[]
        for m in ('trust_auroc','preference_auroc','D_Prec_20'):
            vals.extend(fmt(s[k][m]['mean'],m.endswith('auroc')) for k in ('matched_B0','paired_variant'))
        delta=s['paired_delta_pp']['D_Prec_20']['mean']
        lines.append('| '+s['dataset']+' | '+s['variant']+' | '+' | '.join(vals)+f' | {delta:.2f} |')
    lines += ['','## 最终分类：固定eval1000，结果未变','',
              '旧冻结D1与D2没有额外训练Joint分类器，因此没有最终分类结果；不以缺失值代替0。每格为五个split seeds的准确率均值±总体标准差（%）。','',
              '| 数据集 | shot | B0 | D1 25% | D1 50% | D1 75% | D1 100% | D3 | D4 confidence_router | D4 confidence_preference |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sh in SHOTS:
        for ds in DS:
            values=[]
            for v in ('b0','d1_numeric/0.25','d1_numeric/0.5','d1_numeric/0.75','d1_numeric/1.0','d3','d4/confidence_router','d4/confidence_preference'):
                a=[r['accuracy'] for r in rows if (r['shot'],r['dataset'],r['variant'])==(sh,ds,v)]
                assert len(a)==5
                values.append(f'{100*statistics.mean(a):.2f} ± {100*statistics.pstdev(a):.2f}')
            lines.append('| '+ds+' | '+str(sh)+' | '+' | '.join(values)+' |')
    lines += ['','## 复核与定位','',
              '- 计算代码：experiments/stage0_20261027/report_full_query.py。cell入口逐组核查完整Query、分支分割、训练/验证排除、D2新增父节点及B0相同名单，再计算指标；aggregate入口核查全部75组封存哈希。',
              '- 指令登记：provenance/full-query-amendment-20261027/AMENDMENT.json。旧624文件冻结登记不变，两个根指令文件的修改前原文已归档，其他622个冻结文件逐一验证原哈希。',
              '- RESULTS.json包含1350条逐组结果、匹配B0与变体指标、75份Slurm封存证明；per_cell.csv为扁平指标，summary.csv为汇总及AUROC有效组数。',
              '- 原始scores.pt与模型保持服务器原路径，见RESULTS.json中source字段。旧Release逐节点CSV只是eval1000抽样视图，不能作为全量Query指标的计算全集。',
              '- 完整参数与模型训练输出仍在此前Release；本次仅增加指标更正结果与可复现代码，不覆盖旧封存文件。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    seal(out,dict(all_checks_pass=True,rows=1350,cells=75,full_query_router=True,
                  classification_eval1000_unchanged=True,model_training_unchanged=True,
                  L1_B0_budget_evaluated=False,Stage0_complete=False))
    print('FULL_QUERY_REPORT_COMPLETE',out,flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='cell':cell(int(sys.argv[2]),Path(sys.argv[3]).resolve())
    elif mode=='aggregate':aggregate(Path(sys.argv[2]).resolve())
    else:raise ValueError(mode)
