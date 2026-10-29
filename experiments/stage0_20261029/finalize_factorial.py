"""Add paired references and a source-backed readout; preserve initial publication."""
from factorial_joint import *
import shutil,tarfile

def main():
    registered();paired=OUT/'paired-reference';check_seal(paired)
    dest=ROOT/'reports/joint-factorial-3x2-20261029';old=read(dest/'PUBLICATION_SHA256.json')
    for rel,h in old.items():assert digest(dest/rel)==h,rel
    provenance=dest/'publication-provenance';provenance.mkdir()
    shutil.copy2(dest/'PUBLICATION_SHA256.json',provenance/'ORIGINAL_PUBLICATION_SHA256.json')
    shutil.copy2(dest/'README.md',provenance/'ORIGINAL_README.md')
    shutil.copytree(paired,dest/'paired-reference')
    data=read(OUT/'report/RESULTS.json');scope=read(OUT/'scope-supplement/RESULTS.json');pr=read(paired/'RESULTS.json')
    summary={r['variant']:r for r in data['summary'] if r['dataset']=='ALL' and r['shot']=='ALL'}
    scopes={r['variant']:r for r in scope['summary'] if r['dataset']=='ALL'}
    def f(x,scale=1):return 'NA' if x is None else f'{x*scale:.3f}'
    lines=['# 3×2完成结果与解读','',
        '全部525份分类结果完成，300份新训练及225份复用模型通过输入/超参/伪标签池核验；独立从checkpoint重新推理eval1000，预测标签逐节点完全一致，概率容差2e-6。',
        '主要结论：Joint调参混杂确实存在；D4重调恢复了部分分类差距。纠正性指标目前没有稳定优于整体伪标签精度，不能据此宣称已普遍解释Fig10的PubMed/Products现象。',
        '', '## 设置及定义','',
        'D1使用已完成numeric25% Router；D2使用用户确认的随机a1000；D4是confidence_router。每个处理的两组Joint使用完全相同Router/接受名单/伪标签/专家/划分/完整gold support，只改变是否重新搜索Joint数值超参。固定超参仍重训权重。',
        '主参照pGL=(alpha_B0*pG+beta_B0*pL)/(alpha_B0+beta_B0)，采用B0最终checkpoint固定系数，避免重调同时移动参照。每个模型自身alpha/beta的结果另列。',
        '纠正集合C={已接受且pseudo != argmax(pGL)}；纠正性精度=其中伪标签正确数/|C|。空集合为NA。净准确率纠正=原错变对−原对变错；标签正误差=正确纠正标签−错误纠正标签（含错转错）。原始计数和归一化值都保留，不能混称。',
        'Router/伪标签在完整Query；D2剔除新增训练父节点，B0同名单配对。分类仍是原顺序eval1000。D2新增gold仅进入Router，Joint继续用原support。',
        '', '## 七设置：分类与纠正性指标','',
        '每行75组配置均值；纠正性精度只平均有定义配置，n明确列出。因此跨方法原始均值不一定基于相同有效配置，共同有效配置对照见paired-reference。',
        '', '| Setting | Final accuracy% | Corrective precision% | Defined n/75 | Own-pGL corrective precision% | Own-pGL defined n/75 | Corrective nodes | Net accuracy corrections | Correct-labels minus wrong-labels | Net/1000 Query |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in LABELS:
        s=scopes[v];m=summary[v]['metrics'];a=s['corrective_precision'];b=s['own_pGL_corrective_precision']
        lines.append('| '+' | '.join([v,f(m['accuracy']['mean'],100),f(a['mean'],100),str(a['n'])+'/75',f(b['mean'],100),str(b['n'])+'/75',f(s['corrective_count']['mean']),f(s['net']['mean']),f(s['signed_labels']['mean']),f(m['corrective_net_per_1000_query']['mean'])])+' |')
    lines+=['','## Router与伪标签','', '同一处理两种Joint设置的这些指标完全相同，下表只列一次。D2配对B0单列，不能用原Query的B0指标代替。精度单位%。',
        '', '| Router | Trust AUROC | Pref AUROC | D P@1% | D P@5% | D P@20% | D accepted precision% | A+D accepted precision% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in ('B0','D1_25__fixed','D2a_1000__fixed','D4_confidence__fixed'):
        group=[r for r in data['rows'] if r['variant']==v]
        vals=[stat([r['router_metrics'][k] for r in group])['mean'] for k in ('trust_auroc','preference_auroc','D_Prec_01','D_Prec_05','D_Prec_20','D_deployed_precision','deployed_precision')]
        lines.append('| '+v+' | '+' | '.join(f(x,1 if i<2 else 100) for i,x in enumerate(vals))+' |')
    b=next(r['matched_B0_router'] for r in pr['summary'] if r['dataset']=='ALL' and r['treatment']=='D2a_1000')
    lines.append('| B0 matched D2 Query | '+' | '.join(f(b[k]['mean'],1 if i<2 else 100) for i,k in enumerate(('trust_auroc','preference_auroc','D_Prec_01','D_Prec_05','D_Prec_20','D_deployed_precision','deployed_precision')))+' |')
    lines+=['','## Joint调参的影响','', '| Treatment | Retuned−fixed accuracy pp | Cluster 95% CI pp | B0 cross 0/.25/1 | Retuned cross 0/.25/1 |', '|---|---:|---|---|---|']
    for t in TREATMENTS:
        e=next(r for r in scope['tuning_effects'] if r['dataset']=='ALL' and r['treatment']==t)
        a=summary[t+'__fixed']['cross_counts'];b=summary[t+'__retuned']['cross_counts']
        lines.append('| '+' | '.join([t,f(e['mean_pp']),'['+', '.join(f(x) for x in e['ci95_pp'])+']','/'.join(str(a[str(c)]) for c in (0.,.25,1.)),'/'.join(str(b[str(c)]) for c in (0.,.25,1.))])+' |')
    lines+=['', '历史45格cross=0有2格、cross=.25有13格的说法已核实。扩到75格B0为4/20/51，D4重调后为27/20/28；其中Cora15格有13格选cross=0。验证集确实会选择关小或关闭分歧伪标签。重调还允许dim、学习率、aux、keep等改变，不能把全部收益单独归因cross。',
        '', '## 纠正性指标是否解释最终分类','',
        '使用相对B0的配对差值；下面两种相关在相同有定义配置上计算。整体精度取Fig10对应的实际接受分歧伪标签精度。按dataset×seed簇重采样，保持shot相依；原报告也保留A+D整体精度、Spearman、净纠正及分数据集结果。',
        '', '| Treatment | Joint | Common n | Corrective-precision Pearson r | D-precision Pearson r | Difference 95% CI |','|---|---|---:|---:|---:|---|']
    for r in scope['associations']:
        if r['dataset']=='ALL' and r['method']=='Pearson' and r['comparison']=='D_accepted_precision_delta_pp':
            ci=r['difference_ci95'];lines.append('| '+' | '.join([r['treatment'],r['regime'],str(r['n']),f(r['corrective_r']),f(r['precision_r']),'['+', '.join(f(x) for x in ci)+']' if ci else 'NA'])+' |')
    lines+=['', 'D4重调后，纠正性精度与分类变化的相关接近0，而分歧伪标签整体精度仍呈正相关；D1/D2的方向也不稳定。因此尚不支持“分类跟纠正性精度相关、跟整体精度不相关”的普遍判断。不显著不能证明不相关；这是探索性离线分析。',
        '', '## Fig10范围及反例','',
        '5-shot、seeds42–44的PubMed/Products全部七设置均已单列在scope-supplement。PubMed的confidence纠正集合很小，并有空集合，不能忽略分母。',
        'Products/5-shot/seed44是同一配置内的反例：B0纠正性精度44.44%（9节点）、净纠正0、最终准确率71.90%；D4纠正性精度83.33%（6节点）、净纠正+4，但最终准确率固定Joint69.30%、重调Joint69.10%。两种纠正指标提高仍未保证分类提高，不能把所提指标写成已成立的统一机制。',
        '', '## 定位','',
        '- [完整主报告](report/REPORT.md)：五数据集、三shot、五seed全部指标。',
        '- [分歧区/有效分母/同样本相关区间](scope-supplement/REPORT.md)。',
        '- [D2匹配B0与共同有效配置对照](paired-reference/REPORT.md)。',
        '- [逐模型CSV](report/per_cell.csv)、[逐节点证据](cells/)、[525份配置及训练history](training-records/)、[新训练全部候选选择](training-selection/)。',
        '- [独立复核通过证据](proofs/INDEPENDENT_AUDIT_PASS.json)。历史5-shot配方来源的已知限制继续见[基线说明](../../docs/FAIRROUTER_BASELINE.md#原始历史选择的限制)。',
        '', '所有核验与补充数值计算都通过Slurm；旧模型/报告保持封存，未推进阶段1或选择旧L1争议口径。']
    (dest/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n')
    with (dest/'README.md').open('a') as fp:fp.write('\n[本次结果解读与七设置总表](INTERPRETATION.md) · [同名单B0与共同有效配置对照](paired-reference/REPORT.md)\n')
    write(provenance/'FINAL_AMENDMENT.json',dict(source_report_seal_sha256=digest(OUT/'report/COMPLETE.json'),source_scope_seal_sha256=digest(OUT/'scope-supplement/COMPLETE.json'),paired_reference_seal_sha256=digest(paired/'COMPLETE.json'),finalizer_code_sha256=digest(__file__),numeric_models_changed=False,original_manifest_sha256=digest(provenance/'ORIGINAL_PUBLICATION_SHA256.json')))
    manifest={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='PUBLICATION_SHA256.json'}
    (dest/'PUBLICATION_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    asset=ROOT/'runs/joint-factorial-3x2-20261029-results-final.tar.gz';assert not asset.exists()
    with tarfile.open(asset,'w:gz') as t:
        t.add(dest,arcname=str(dest.relative_to(ROOT)));t.add(THIS,arcname=str(THIS.relative_to(ROOT)));t.add(ROOT/'docs/JOINT_FACTORIAL_3X2.md',arcname='docs/JOINT_FACTORIAL_3X2.md')
    receipt=dict(all_checks_pass=True,publication=str(dest),manifest_files=len(manifest),archive=str(asset),archive_sha256=digest(asset),archive_bytes=asset.stat().st_size,all525_checkpoint_labels_match=True,code_sha256=digest(__file__))
    write(OUT/'FINAL_PUBLICATION.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
