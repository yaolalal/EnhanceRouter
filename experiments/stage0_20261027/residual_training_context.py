"""Supplement the frozen residual audit with actual supervision/loss context."""
from analyze_residual import *

def main(source,out):
    preflight();rows=[];seals={}
    for ds in DS_ORDER:
        p=source/'datasets'/ds;r=read(p/'COMPLETE.json')
        for name,h in r['files'].items():assert digest(p/name)==h
        data=read(p/'RESULTS.json');assert data['analysis_code_sha256']==digest(Path(__file__).with_name('analyze_residual.py'))
        rows+=data['per_cell'];seals[ds]=digest(p/'COMPLETE.json')
    assert len(rows)==105
    native_data=read(ROOT/'reports/full-query-20261027/RESULTS.json')
    native={(r['key'],r['variant']):r for r in native_data['per_cell']}
    arxiv=[r for r in rows if r['dataset']=='arxiv'];assert len(arxiv)==45
    index={(r['key'],r['variant']):r for r in arxiv}
    context=[]
    for r in arxiv:
        n=native[(r['key'],r['variant'])];cfg=r['config']
        base=index[(r['key'],'b0')]
        for field in ('embeddings','hidden','upstream'):
            assert r['source'][field]['hash']==base['source'][field]['hash'],(r['key'],field)
        assert r['joint_gold_support_count']==base['joint_gold_support_count']
        if r['variant'].startswith('d1_numeric'):
            fraction=float(r['variant'].split('/')[1])
            assert r['router_target_parent_selection_count']==max(1,int(r['joint_gold_support_count']*fraction+.5))
        context.append(dict(key=r['key'],shot=r['shot'],seed=r['seed'],variant=r['variant'],
            router_target_parent_selection_count=r['router_target_parent_selection_count'],
            joint_gold_support_count=r['joint_gold_support_count'],
            **{k:cfg[k] for k in ('dim','lr','cross','gate_lr','warmup','gate_gold','aux','keep')},
            accepted_disagreement_nodes=n['D_deployed_count'],accepted_agreement_nodes=n['A_deployed_count'],
            disagreement_training_pool_used=cfg['cross']>0,agreement_KL_pool_used=cfg['keep']>0,
            accepted_disagreement_precision=n['D_deployed_precision'],accepted_agreement_precision=n['A_deployed_precision'],
            accepted_all_precision=n['deployed_precision'],D_Prec_01=n['D_Prec_01'],D_Prec_20=n['D_Prec_20'],
            **{m:r[m] for m in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction','weight_R')}))
    csv_write(out/'arxiv_training_context.csv',context)
    summaries=[]
    for cohort in ('B0_historical45','B0_all75','Arxiv_B0','Arxiv_D1_25','Arxiv_D1_100'):
        group=[r for r in rows if (r['historical_45'] if cohort=='B0_historical45' else r['variant']=='b0' if cohort=='B0_all75' else r['dataset']=='arxiv' and r['variant']=={'Arxiv_B0':'b0','Arxiv_D1_25':'d1_numeric/0.25','Arxiv_D1_100':'d1_numeric/1.0'}[cohort])]
        for ds in ('ALL',)+DS_ORDER:
            for sh in ('ALL',)+SHOTS:
                g=[r for r in group if (ds=='ALL' or r['dataset']==ds) and (sh=='ALL' or r['shot']==sh)]
                if g:summaries.append(dict(cohort=cohort,dataset=ds,shot=sh,cells=len(g),**{m:statistics.mean(r[m] for r in g) for m in METRICS}))
    csv_write(out/'component_summary.csv',summaries)
    outdata=dict(all_checks_pass=True,arxiv_training_context=context,component_summary=summaries,
                 identical_expert_input_hashes_across_Arxiv_variants=True,joint_gold_support_unchanged=True,
                 dataset_seals=seals,native_report_sha256=digest(ROOT/'reports/full-query-20261027/RESULTS.json'),
                 code_sha256=digest(__file__),slurm_job=os.environ['SLURM_JOB_ID'],training_performed=False)
    write(out/'RESULTS.json',outdata)
    lines=['# Arxiv：训练监督与伪标签口径补充','',
           '本表解释当前已完成的冻结实验，不提出或选择新方法。已逐组核对B0、D1 25%、D1 100%的GNN embedding、LLM hidden和专家logits输入哈希一致，Joint gold support数量一致；变化仅来自原实验允许的训练池、数值超参和学习结果。','',
           'Router的P@1%只衡量完整Query分歧分支排序前1%的正确率，并不等于Joint实际接收的所有伪标签精度。后者的节点数和精度另列；各变体自己的接收比例可不同，不把它当作固定B0预算指标。','',
           'Joint训练始终有gold NLL；cross>0时才使用两个分歧伪标签池的NLL，keep>0时才使用一致区KL。损失按各抽样batch取均值并乘配置系数，因此池节点数不等同于总损失权重。','',
           '| shot | seed | 配置 | Router父节点 | Joint gold节点 | cross | aux | keep | 已接受D节点 | 已接受A节点 | D精度% | A精度% | 所有接受精度% |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def percent(x):return '—' if x is None else f'{100*x:.2f}'
    for r in context:
        values=[str(r[k]) for k in ('shot','seed','variant','router_target_parent_selection_count','joint_gold_support_count','cross','aux','keep','accepted_disagreement_nodes','accepted_agreement_nodes')]
        values += [percent(r[k]) for k in ('accepted_disagreement_precision','accepted_agreement_precision','accepted_all_precision')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['','## pR作用的汇总（固定eval1000）','',
              '| cohort | 数据集 | shot | 组数 | pJ% | pR单独% | 去pR% | 改变标签% | 改对% | 改错% | wR% |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['dataset']!='ALL' or (r['cohort'].startswith('B0_') and r['shot']!='ALL'):continue
        values=[percent(r[m]) for m in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction','rescued_fraction','harmed_fraction','weight_R')]
        lines.append('| '+r['cohort']+' | '+r['dataset']+' | '+str(r['shot'])+' | '+str(r['cells'])+' | '+' | '.join(values)+' |')
    lines += ['','pR单独推理仍使用原先训练出的参数，其中可包含Router伪标签训练的影响。固定权重移除pR衡量直接推理贡献，不等价于重新训练无pR模型，也不能单独证明Router在训练中没有作用。','']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    seal(out,dict(all_checks_pass=True,Arxiv_models=45,training_performed=False))
    print('RESIDUAL_TRAINING_CONTEXT_PASS',out,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
