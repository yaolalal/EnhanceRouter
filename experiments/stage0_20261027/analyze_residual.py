"""Frozen-checkpoint residual decomposition; no fitting or model selection.

Slurm entrypoints: dataset INDEX OUTPUT; aggregate OUTPUT.
All classification/label-change statistics use the exact ordered eval1000.
Router diagnostics are reused from the sealed full-Query report.
"""
from report_full_query import preflight
from d3_recovery import *
import csv,statistics,math,gc
from datetime import datetime,timezone,timedelta

DS_ORDER=('cora','citeseer','pubmed','arxiv','ogbn-products')
VARIANTS=('b0','d1_numeric/0.25','d1_numeric/1.0')
METRICS=('joint_accuracy','repair_only_accuracy','without_repair_accuracy',
         'label_change_fraction','rescued_fraction','harmed_fraction','wrong_to_wrong_fraction',
         'gnn_accuracy','llm_accuracy','alpha','beta','weight_R')

def csv_write(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or sorted({k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))})
    with path.open('x',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

class Sources:
    def __init__(self):self.registered=all_sources();self.checked={};self.tensors={}
    def check(self,path,expected=None):
        p=Path(path);expected=expected or self.registered.get(str(p))
        assert expected is not None,('unregistered source',str(p))
        if isinstance(expected,str):expected=dict(sha256=expected)
        kind='sha256' if 'sha256' in expected else 'md5'
        fingerprint=(str(p.resolve()),kind,expected[kind])
        if fingerprint not in self.checked:
            actual=digest(p,kind);assert actual==expected[kind],('source changed',p)
            self.checked[fingerprint]=dict(path=str(p),resolved=str(p.resolve()),hash_algorithm=kind,hash=actual,bytes=p.stat().st_size)
        return self.checked[fingerprint]
    def tensor(self,path,expected=None,cache=False):
        record=self.check(path,expected);identity=(record['hash_algorithm'],record['hash'])
        if cache and identity in self.tensors:return self.tensors[identity],record
        obj=load(path)
        if cache:self.tensors[identity]=obj
        return obj,record

def inputs(sh,se,ds,variant,src):
    base=folder(sh,se,ds);out=base/variant;complete=read(out/'COMPLETE.json')
    for name in ('config.json','predictions-reference.json'):
        if name in complete['files']:assert digest(out/name)==complete['files'][name]
    variant_config=read(out/'config.json');cfg=variant_config['joint'];hist=read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'].get(key(sh,se,ds))
    canonical=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']
    if variant=='b0' and hist is not None:
        checkpoint=Path(hist['checkpoint']);cp_expected=None
        epath=hist['inputs']['embeddings'];hpath=hist['inputs']['hidden']
        assert cfg==hist['config']
    else:
        checkpoint=out/'checkpoint.pt';cp_expected=complete['files']['checkpoint.pt']
        epath=canonical['inputs']['embeddings'];hpath=canonical['inputs']['hidden']
    if hist is not None:
        upath=Path(hist['inputs']['upstream']);up_expected=None
        spath=Path(hist['inputs']['supervision']);s_expected=None
    else:
        upath=base/'standard_native/upstream.pt'
        up_expected=read(upath.parent/'BANK_COMPLETE.json')['files'][upath.name]
        spath=upath.parent/'supervision.pt';s_expected=read(upath.parent/'BANK_COMPLETE.json')['files'][spath.name]
    member,cp_record=src.tensor(checkpoint,cp_expected)
    embeddings,e_record=src.tensor(epath,cache=True);hidden,h_record=src.tensor(hpath,cache=True)
    upstream,u_record=src.tensor(upath,up_expected)
    supervision,s_record=src.tensor(spath,s_expected)
    assert supervision['support_ids'].tolist()==split(sh,se,ds)['support_ids']
    pred=out/'predictions.pt'
    if pred.exists():prediction,p_record=src.tensor(pred,complete['files']['predictions.pt'])
    else:
        ref=read(out/'predictions-reference.json');prediction,p_record=src.tensor(ref['path'],ref['sha256'])
    assert member['config']==cfg and member['training_seed']==se and not member['inner_ensemble']
    return cfg,member,embeddings['embeddings'],hidden['hidden'],upstream,prediction,dict(
        checkpoint=cp_record,embeddings=e_record,hidden=h_record,upstream=u_record,predictions=p_record,
        supervision=s_record,joint_gold_support_count=len(supervision['support_ids']),
        router_target_parent_selection_count=len(variant_config.get('parent_ids',supervision['support_ids'])),
        variant_path=str(out),variant_seal_sha256=digest(out/'COMPLETE.json'))

def one(sh,se,ds,variant,truth,src,out,prior):
    import torch,numpy as np
    from eargtc.joint_explore.joint_purf_v10_v1 import training
    cfg,member,e,h,up,reference,provenance=inputs(sh,se,ds,variant,src)
    ids=torch.tensor(evaluation_ids(se,ds),dtype=torch.long)
    assert len(ids)==1000 and len(ids.unique())==1000
    sp=split(sh,se,ds)
    assert not set(ids.tolist())&set(sp['support_ids']+sp['valid_ids'])
    f=dict(gnn_embeddings=e[ids].float(),logp_g=up['gnn_logits'][ids].float().log_softmax(1),
           logp_l=up['llm_logits'][ids].float().log_softmax(1))
    hs=h[ids];model=training.make_model(f,hs,cfg);model.load_state_dict(member['states']['accuracy'])
    local_ids=torch.arange(1000)
    with torch.inference_mode():
        pj,pr=training.predict(model,f,hs,local_ids,repair=True)
        a,b=model.coefficients();a=float(a);b=float(b)
    assert math.isfinite(a+b) and a>0 and b>0
    pg=f['logp_g'].exp().double();pl=f['logp_l'].exp().double()
    without=(a*pg+b*pl)/(a+b)
    reconstructed=(a*pg+b*pl+pr.double())/(a+b+1)
    torch.testing.assert_close(pj.double(),reconstructed,rtol=2e-6,atol=2e-7)
    expected=reference['probability'][ids]
    torch.testing.assert_close(pj,expected,rtol=0,atol=2e-6)
    assert torch.equal(pj.argmax(1),expected.argmax(1)),('classification labels differ',sh,se,ds,variant)
    for probability in (pj,pr,without):
        assert bool(torch.isfinite(probability).all())
        torch.testing.assert_close(probability.double().sum(1),torch.ones(1000,dtype=torch.double),rtol=0,atol=2e-6)
    y=truth[ids];j=pj.argmax(1);r=pr.argmax(1);n=without.argmax(1)
    g=pg.argmax(1);l=pl.argmax(1);changed=j!=n
    rescued=(n!=y)&(j==y);harmed=(n==y)&(j!=y);wrong_to_wrong=changed&(n!=y)&(j!=y)
    assert torch.equal(changed,rescued|harmed|wrong_to_wrong)
    jc=int((j==y).sum());nc=int((n==y).sum())
    assert jc-nc==int(rescued.sum())-int(harmed.sum())
    history=member['history'];chosen=member['checkpoints']['accuracy'];selected_step=chosen['step']
    assert history[0]['step']==0 and all(history[t]['step']<history[t+1]['step'] for t in range(len(history)-1))
    selected=[v for v in history if v['step']==selected_step];assert len(selected)==1
    assert abs(selected[0]['alpha']-a)<2e-7 and abs(selected[0]['beta']-b)<2e-7
    native=prior[(key(sh,se,ds),variant)]
    assert abs(jc/1000-native['accuracy'])<1e-12
    row=dict(key=key(sh,se,ds),dataset=ds,shot=sh,seed=se,variant=variant,
             historical_45=(variant=='b0' and se<=44),eval_count=1000,
             eval_ordered_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),
             joint_accuracy=jc/1000,repair_only_accuracy=int((r==y).sum())/1000,without_repair_accuracy=nc/1000,
             gnn_accuracy=int((g==y).sum())/1000,llm_accuracy=int((l==y).sum())/1000,
             label_change_count=int(changed.sum()),label_change_fraction=int(changed.sum())/1000,
             rescued_count=int(rescued.sum()),rescued_fraction=int(rescued.sum())/1000,
             harmed_count=int(harmed.sum()),harmed_fraction=int(harmed.sum())/1000,
             wrong_to_wrong_count=int(wrong_to_wrong.sum()),wrong_to_wrong_fraction=int(wrong_to_wrong.sum())/1000,
             change_on_expert_agreement_count=int((changed&(g==l)).sum()),
             change_on_expert_disagreement_count=int((changed&(g!=l)).sum()),
             alpha=a,beta=b,weight_G=a/(a+b+1),weight_L=b/(a+b+1),weight_R=1/(a+b+1),
             selected_step=selected_step,stop_step=history[-1]['step'],logged_points=len(history),
             log_interval=cfg['validate_every'],config=cfg,history=history,source=provenance,
             joint_gold_support_count=provenance['joint_gold_support_count'],
             router_target_parent_selection_count=provenance['router_target_parent_selection_count'],
             classification_prediction_verified=True,mixture_equation_verified=True,
             router_full_query={m:native[m] for m in ('D_Prec_01','D_Prec_05','D_Prec_20','D_Prec_100','deployed_precision','D_deployed_precision','A_deployed_precision','deployed_count')},
             training_performed=False,query_truth_used_for_selection=False)
    target=out/'cells'/f'shot{sh}'/f'seed{se}'/ds/variant
    write(target/'RESULTS.json',row)
    trace=[dict(step=x['step'],alpha=x['alpha'],beta=x['beta'],weight_G=x['weights'][0],weight_L=x['weights'][1],weight_R=x['weights'][2],
                validation_accuracy=x['accuracy'],repair_validation_accuracy=x['repair']['accuracy'],selected_checkpoint=(x['step']==selected_step)) for x in history]
    csv_write(target/'coefficient_history.csv',trace)
    records=[]
    for t,node in enumerate(ids.tolist()):
        records.append(dict(eval_order=t,node_id=node,true_class=int(y[t]),joint_pred=int(j[t]),repair_only_pred=int(r[t]),
                            without_repair_pred=int(n[t]),gnn_pred=int(g[t]),llm_pred=int(l[t]),
                            changed_by_repair=bool(changed[t]),rescued=bool(rescued[t]),harmed=bool(harmed[t]),
                            changed_wrong_to_wrong=bool(wrong_to_wrong[t]),joint_confidence=float(pj[t].max()),
                            repair_confidence=float(pr[t].max()),without_repair_confidence=float(without[t].max())))
    csv_write(target/'eval1000_decomposition.csv',records)
    seal(target,dict(all_checks_pass=True,analysis_code_sha256=digest(__file__),frozen_inference_only=True))
    print('RESIDUAL_CELL',key(sh,se,ds),variant,'J',jc,'R',int((r==y).sum()),'noR',nc,'flips',int(changed.sum()),flush=True)
    return row

def dataset(index,out):
    preflight();ds=DS_ORDER[index];src=Sources()
    graph_path=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']['graph'];src.check(graph_path)
    truth=graph(ds).y.reshape(-1).long()
    prior_data=read(ROOT/'reports/full-query-20261027/RESULTS.json')
    prior={(r['key'],r['variant']):r for r in prior_data['per_cell']}
    rows=[]
    for sh in SHOTS:
        for se in DEV:
            variants=VARIANTS if ds=='arxiv' else ('b0',)
            for v in variants:
                rows.append(one(sh,se,ds,v,truth,src,out,prior));gc.collect()
    write(out/'datasets'/ds/'RESULTS.json',dict(dataset=ds,per_cell=rows,all_checks_pass=True,
          analysis_code_sha256=digest(__file__),source_checks=list(src.checked.values()),slurm_job=os.environ['SLURM_JOB_ID']))
    seal(out/'datasets'/ds,dict(all_checks_pass=True,cells=len(rows)))

def stats(a):
    a=[float(x) for x in a]
    return dict(n=len(a),mean=statistics.mean(a),std_ddof0=statistics.pstdev(a))

def plot(out,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=('#D55E00','#0072B2');figdir=out/'figures';figdir.mkdir()
    index={(r['shot'],r['seed'],r['dataset'],r['variant']):r for r in rows}
    def curves(ax,r,style='-',prefix=''):
        for field,color in zip(('alpha','beta'),colors):
            ax.plot([x['step'] for x in r['history']],[x[field] for x in r['history']],style,color=color,marker='.',ms=3,lw=1.4,label=prefix+field)
            ax.scatter([r['selected_step']],[r[field]],color=color,s=32,marker='o' if style=='-' else 's',facecolors=color if style=='-' else 'none',zorder=4)
        ax.grid(alpha=.22);ax.set_xlabel('Training step');ax.set_ylabel('Coefficient')
    with PdfPages(figdir/'B0_historical45_alpha_beta.pdf') as book:
        for sh in SHOTS:
            fig,axes=plt.subplots(5,3,figsize=(12,14),layout='constrained')
            for i,ds in enumerate(DS_ORDER):
                for k,se in enumerate((42,43,44)):
                    r=index[(sh,se,ds,'b0')];ax=axes[i,k];curves(ax,r)
                    ax.axvline(r['selected_step'],color='.4',ls=':',lw=.8)
                    ax.set_title(f'{ds} | seed {se} | selected step {r["selected_step"]}')
            axes[0,0].legend(loc='best')
            fig.suptitle(f'FairRouter / B0: {sh}-shot, seeds 42-44\nActual logged validation checkpoints; circles mark selected model',fontsize=13)
            fig.savefig(figdir/f'B0_shot{sh}_alpha_beta.png',dpi=170);book.savefig(fig);plt.close(fig)
    for name,control in [('Arxiv_D1_25_vs_100','d1_numeric/1.0'),('Arxiv_D1_25_vs_B0','b0')]:
        fig,axes=plt.subplots(3,5,figsize=(19,10),layout='constrained')
        for i,sh in enumerate(SHOTS):
            for k,se in enumerate(DEV):
                r=index[(sh,se,'arxiv','d1_numeric/0.25')];c=index[(sh,se,'arxiv',control)];ax=axes[i,k]
                curves(ax,c,'--','control ');curves(ax,r,'-','25% ')
                ax.set_title(f'{sh}-shot | seed {se}\nselected: 25%={r["selected_step"]}, control={c["selected_step"]}')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=4)
        fig.suptitle(f'Arxiv numeric-retuned D1: 25% vs {"100% internal control" if control!="b0" else "historical/native B0"}\nSolid =25%; dashed =control; markers =actual logged steps; large circles/squares =selected checkpoint',fontsize=12,y=1.05)
        fig.savefig(figdir/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    # Individually readable panels for each of the requested45 B0 and15 D1 cells.
    for r in rows:
        if not r['historical_45'] and r['variant']!='d1_numeric/0.25':continue
        fig,ax=plt.subplots(figsize=(6.5,4.2),layout='constrained');curves(ax,r);ax.legend()
        ax.axvline(r['selected_step'],color='.4',ls=':',lw=.8)
        ax.set_title(f'{r["dataset"]} | {r["shot"]}-shot | seed {r["seed"]}\n{r["variant"]}; selected step {r["selected_step"]}, stopped {r["stop_step"]}')
        stem=f'{r["variant"].replace("/","_")}_shot{r["shot"]}_seed{r["seed"]}_{r["dataset"]}'
        fig.savefig(figdir/(stem+'.png'),dpi=170);fig.savefig(figdir/(stem+'.pdf'));plt.close(fig)

def aggregate(out):
    preflight();rows=[];proof=[]
    for ds in DS_ORDER:
        p=out/'datasets'/ds;seal_record=read(p/'COMPLETE.json')
        for rel,h in seal_record['files'].items():assert digest(p/rel)==h
        r=read(p/'RESULTS.json');assert r['analysis_code_sha256']==digest(__file__) and r['all_checks_pass']
        rows+=r['per_cell'];proof.append(dict(dataset=ds,slurm_job=r['slurm_job'],seal_sha256=digest(p/'COMPLETE.json')))
    assert len(rows)==105 and sum(r['historical_45'] for r in rows)==45
    for r in rows:
        p=out/'cells'/f'shot{r["shot"]}'/f'seed{r["seed"]}'/r['dataset']/r['variant']
        for rel,h in read(p/'COMPLETE.json')['files'].items():assert digest(p/rel)==h
    summaries=[]
    for cohort in ('B0_historical45','B0_all75','Arxiv_B0','Arxiv_D1_25','Arxiv_D1_100'):
        group=[r for r in rows if (r['historical_45'] if cohort=='B0_historical45' else
               r['variant']=='b0' if cohort=='B0_all75' else
               r['dataset']=='arxiv' and r['variant']=={'Arxiv_B0':'b0','Arxiv_D1_25':'d1_numeric/0.25','Arxiv_D1_100':'d1_numeric/1.0'}[cohort])]
        for ds in ('ALL',)+DS_ORDER:
            for sh in ('ALL',)+SHOTS:
                g=[r for r in group if (ds=='ALL' or r['dataset']==ds) and (sh=='ALL' or r['shot']==sh)]
                if g:summaries.append(dict(cohort=cohort,dataset=ds,shot=sh,cells=len(g),metrics={m:stats([r[m] for r in g]) for m in METRICS}))
    arxiv=[];idx={(r['key'],r['variant']):r for r in rows}
    for sh in SHOTS:
        for se in DEV:
            for control in ('b0','d1_numeric/1.0'):
                k=key(sh,se,'arxiv');a=idx[(k,'d1_numeric/0.25')];b=idx[(k,control)]
                arxiv.append(dict(key=k,shot=sh,seed=se,control=control,
                    classification_delta_pp=100*(a['joint_accuracy']-b['joint_accuracy']),
                    repair_only_delta_pp=100*(a['repair_only_accuracy']-b['repair_only_accuracy']),
                    without_repair_delta_pp=100*(a['without_repair_accuracy']-b['without_repair_accuracy']),
                    change_fraction_delta_pp=100*(a['label_change_fraction']-b['label_change_fraction']),
                    **{m+'_delta_pp':100*(a['router_full_query'][m]-b['router_full_query'][m]) for m in ('D_Prec_01','D_Prec_05','D_Prec_20','deployed_precision')}))
    comparisons=[]
    for control in ('b0','d1_numeric/1.0'):
        for sh in ('ALL',)+SHOTS:
            g=[r for r in arxiv if r['control']==control and (sh=='ALL' or r['shot']==sh)]
            comparisons.append(dict(control=control,shot=sh,cells=len(g),metrics={m:stats([r[m] for r in g]) for m in g[0] if m.endswith('_pp')}))
    csv_write(out/'per_cell.csv',rows);csv_write(out/'arxiv_paired.csv',arxiv)
    traces=[]
    for r in rows:
        for x in r['history']:traces.append(dict(key=r['key'],dataset=r['dataset'],shot=r['shot'],seed=r['seed'],variant=r['variant'],step=x['step'],alpha=x['alpha'],beta=x['beta'],weight_G=x['weights'][0],weight_L=x['weights'][1],weight_R=x['weights'][2],selected_checkpoint=(x['step']==r['selected_step'])))
    csv_write(out/'coefficient_history.csv',traces)
    # Preserve the exact complete-cohort summaries in older reports for claim tracing.
    old=[]
    for name in ('d1-d4-current-20261027T065850Z','d1-d4-current-20261027T074658Z','all-results-20261027'):
        p=ROOT/'reports'/name/'RESULTS.json';d=read(p)
        old.append(dict(report=str(p),sha256=digest(p),router_scope='old eval1000 sampling; superseded',
                        arxiv_complete_cohort=[x for x in d['numeric_complete_cohort'] if x['dataset']=='arxiv' and x['fraction'] in (.25,1.)]))
    plot(out,rows)
    data=dict(created_at_shanghai=datetime.now(timezone(timedelta(hours=8))).isoformat(),per_cell=rows,
              summary=summaries,arxiv_paired=arxiv,arxiv_summary=comparisons,old_report_claim_trace=old,proofs=proof,
              formula='pJ=(alpha*pG+beta*pL+pR)/(alpha+beta+1)',without_pR='(alpha*pG+beta*pL)/(alpha+beta), same trained alpha/beta; no retraining',
              contested_region='argmax(pJ) differs from argmax(without_pR) on the exact ordered eval1000',
              historical_B0_requested_curves=45,Arxiv_D1_25_curves=15,classification_models_analyzed=105,
              trace_scope='Original stored validation-checkpoint history, usually every50 optimizer steps; intermediate steps were not recorded; no synthetic interpolation or replay',
              checkpoint_rule='Original validation accuracy then NLL then earliest step; always selected accuracy state, not last state',
              numerical_model_files_changed=False,training_performed=False,query_truth_used_for_selection=False,
              source=dict(analysis_code_sha256=digest(__file__),old_code_freeze_sha256=digest(RUN/'CODE_FREEZE.json'),instruction_amendment_sha256=digest(ROOT/'provenance/full-query-amendment-20261027/AMENDMENT.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    write(out/'RESULTS.json',data)
    render_report(out,data)
    seal(out,dict(all_checks_pass=True,models=105,requested_B0_curves=45,Arxiv_D1_25_curves=15,frozen_inference_only=True,classification_eval1000=True))
    print('RESIDUAL_AUDIT_COMPLETE',out,'models105 requested_B0_curves45 Arxiv_D1_25_curves15',flush=True)

def render_report(out,data):
    rows=data['per_cell'];fmt=lambda x:f'{100*x:.2f}'
    lines=['# B0 / FairRouter residual模块与25%重调参D1诊断','',
           '时间（Asia/Shanghai）：'+data['created_at_shanghai'],'',
           '45组历史B0训练曲线：3/5/10-shot × seeds42–44 × 5数据集。分类消融另外覆盖全部75组B0；Arxiv同时覆盖seeds42–46、三种shot的D1 25%和100%内部对照，共105份已冻结模型。无重新训练，无新模型选择。所有分类准确率与标签改变率均在各seed原顺序eval1000上计算；Router伪标签指标引用已封存的完整Query报告。','',
           '## 代码与公式','',
           '实际模型是 `src/eargtc/joint_explore/joint_purf_v10_v1/model.py` 的 `UnifiedRepair`：两侧embedding分别LayerNorm、线性投影、GELU，拼接后经64宽隐藏层与分类层得到pR。专家输入detach，alpha/beta经softplus为正。','',
           '`pJ=(alpha*pG+beta*pL+pR)/(alpha+beta+1)`。用户省略的分母不影响argmax。去掉pR后使用`(alpha*pG+beta*pL)/(alpha+beta)`，不重新训练、不重调alpha/beta。pR单独准确率使用argmax(pR)。这里pR是单独的归一化概率分类头，不是可正可负的logit残差。','',
           '| 核查内容 | 仓库位置 |','|---|---|',
           '| residual结构、alpha/beta、融合 | [model.py](../../src/eargtc/joint_explore/joint_purf_v10_v1/model.py#L8-L37) |',
           '| 优化器、监督/伪标签损失、checkpoint和系数日志 | [training.py](../../src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L34-L112) |',
           '| B0历史大表重训与精确一致性检查 | [cli.py](../../src/enhancerouter/cli.py#L84-L142) |',
           '| 专家特征、gold/selection与伪标签池 | [adapter.py](../../src/enhancerouter/adapter.py#L16-L90) |',
           '| 当前B0原生Joint训练入口 | [compute.py](../../experiments/stage0_20261026/compute.py#L103-L121) |',
           '| 25%重调参D1及Joint受控搜索 | [d1_numeric.py](../../experiments/stage0_20261027/d1_numeric.py#L29-L48) |',
           '| 本次拆分、核验与作图 | [analyze_residual.py](../../experiments/stage0_20261027/analyze_residual.py) |','',
           '训练使用AdamW：residual参数学习率lr、weight_decay=1e-4；alpha/beta原始参数学习率lr×gate_lr、不做weight decay。损失包含gold混合预测NLL、按cross加权的两个方向伪标签NLL，及配置允许的aux residual NLL和keep一致区KL；warmup/gate_gold决定何时截断系数梯度。按validation accuracy、NLL、较早step选checkpoint。','',
           '## 45组B0系数曲线','',
           '每个子图一组配置，两条线分别为原始正系数alpha/beta。原训练history在step0和验证时刻记录，默认每50steps；图中点全部来自原记录，不冒充逐optimizer-step轨迹。圆点/竖线标记最终选中的checkpoint；曲线可继续到早停末步，末点不一定是选中的模型。','',
           '[45组矢量PDF](figures/B0_historical45_alpha_beta.pdf)','']
    for sh in SHOTS:lines+= [f'### {sh}-shot','',f'![B0 {sh}-shot](figures/B0_shot{sh}_alpha_beta.png)','']
    lines+=['## 历史45组B0：pR单独、移除pR及标签改变区','',
            '百分数均以固定1000节点为分母；每格为对应shot、dataset的三个seed均值。改变区为加入pR后相对无pR的argmax改变比例。改对/改错各自以1000为分母。','',
            '| 数据集 | shot | 完整pJ准确率 | pR单独准确率 | 去pR准确率 | pR改变标签比例 | 改对 | 改错 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for sh in SHOTS:
        for ds in DS_ORDER:
            g=[r for r in rows if r['historical_45'] and r['shot']==sh and r['dataset']==ds]
            vals=[fmt(statistics.mean(r[m] for r in g)) for m in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction','rescued_fraction','harmed_fraction')]
            lines.append('| '+ds+' | '+str(sh)+' | '+' | '.join(vals)+' |')
    lines+=['','全部75组B0及每个seed原始值见per_cell.csv；JSON还保留alpha/beta、归一化wR、选中step、真实文件路径和哈希。逐节点的改对、改错、改成另一错误类见cells/内eval1000_decomposition.csv。精确恒等式：完整正确数−去pR正确数＝改对数−改错数；已对105份模型逐一验证。','',
            '## Arxiv：25%重调参实验的对应关系','',
            '25%实验定位为`runs/stage0-20261026-v2/cells/shot{3,5,10}/seed{42..46}/arxiv/d1_numeric/0.25`。`d1_numeric/1.0`为相同重调流程的100%内部对照，不能直接当作历史B0。下表使用完整15组，明确区分两种参照；旧eval1000 Router快照另存JSON的old_report_claim_trace，不能混入完整Query结果。','',
            '这里25%仅下采样Router目标训练bank的support父节点，见d1_numeric.py的parents/getter；后续compute.joint仍用原split及supervision，adapter.py仍把全部support_ids作为gold_ids。因此Joint residual的直接gold监督没有缩成25%。逐组router_target_parent_selection_count与joint_gold_support_count已单列，不能把本实验解释为整个端到端模型只剩25%金标签。','',
            '| 参照 | shot | 组数 | 25%−参照 P@1% pp | P@5% pp | P@20% pp | 实际接受伪标签精度pp | 最终分类准确率pp |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in data['arxiv_summary']:
        vals=[c['metrics'][m]['mean'] for m in ('D_Prec_01_delta_pp','D_Prec_05_delta_pp','D_Prec_20_delta_pp','deployed_precision_delta_pp','classification_delta_pp')]
        lines.append('| '+c['control']+' | '+str(c['shot'])+' | '+str(c['cells'])+' | '+' | '.join(f'{v:+.2f}' for v in vals)+' |')
    lines+=['','“接近10pp”需要指定P@k与参照；本报告不把P@1%、P@20%或不同完成快照混用。上表最终分类差值为完整五seed结果，请据此核对用户提到的−0.10pp。','',
            '## Arxiv：每个shot、五个seed的分类分解','',
            '| shot | seed | 配置 | pJ准确率 | pR单独准确率 | 去pR准确率 | pR改变标签比例 | 改对 | 改错 | alpha | beta | wR | 选中step |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sh in SHOTS:
        for se in DEV:
            for v in VARIANTS:
                r=next(x for x in rows if (x['shot'],x['seed'],x['dataset'],x['variant'])==(sh,se,'arxiv',v))
                vals=[fmt(r[m]) for m in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction','rescued_fraction','harmed_fraction')]
                lines.append(f'| {sh} | {se} | {v} | '+' | '.join(vals)+f' | {r["alpha"]:.4f} | {r["beta"]:.4f} | {100*r["weight_R"]:.2f} | {r["selected_step"]} |')
    lines+=['','## Arxiv：五seed训练系数曲线','',
            '每个面板对应一个shot/seed。第一图比较25%与100%内部对照；第二图比较25%与B0。实线为25%，虚线为参照，橙色为alpha、蓝色为beta；较大的圆/方标记为各自validation选中的checkpoint。','',
            '![Arxiv25% vs100%](figures/Arxiv_D1_25_vs_100.png)','',
            '[矢量PDF：25% vs100%](figures/Arxiv_D1_25_vs_100.pdf)','',
            '![Arxiv25% vs B0](figures/Arxiv_D1_25_vs_B0.png)','',
            '[矢量PDF：25% vs B0](figures/Arxiv_D1_25_vs_B0.pdf)','',
            '## 文件及验证','',
            '- per_cell.csv：105份冻结分类器的全部分解指标，B0共75份；要求的历史45份有historical_45=true标记。',
            '- coefficient_history.csv：原训练已保存的全部step记录和归一化三分支权重，含105份模型。',
            '- arxiv_paired.csv：15组25%分别对照B0/100%的配对差。',
            '- cells/：每组完整结果、系数history、1000节点分解CSV、完成哈希。',
            '- figures/：三张B0合集、一个三页45组PDF、Arxiv两张15组配对图，以及45组B0和15组25%D1各自的单独PNG/PDF。',
            '- 全部预测已核对冻结完整pJ概率（绝对容差2e-6）、argmax标签逐节点完全一致、accuracy与已发布值一致；无训练、无按Query标签选模型。',
            '- pR改变标签比例只衡量固定checkpoint的直接推理贡献，不等价于重新训练一个无residual模型的效果；pR单独推理也不是独立训练的分类器。',
            '- 原始模型和输入只读；本次输出保存于新的residual-audit运行目录，未覆盖任何旧结果。','']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    if sys.argv[1]=='dataset':dataset(int(sys.argv[2]),Path(sys.argv[3]).resolve())
    elif sys.argv[1]=='aggregate':aggregate(Path(sys.argv[2]).resolve())
    else:raise ValueError(sys.argv[1])
