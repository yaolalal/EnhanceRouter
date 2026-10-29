"""Explicit matched-B0 summaries for D2 and common-defined corrective pairs."""
from factorial_joint import *

def main():
    registered();report=OUT/'report';check_seal(report);data=read(report/'RESULTS.json');dest=OUT/'paired-reference';assert not dest.exists()
    summary=[]
    for ds in ('ALL',*DS):
        for t in TREATMENTS:
            group=[r for r in data['rows'] if r['treatment']==t and r['regime']=='fixed' and (ds=='ALL' or r['dataset']==ds)]
            c=[r for r in group if r['corrective']['corrective_precision'] is not None and r['matched_B0_corrective']['corrective_precision'] is not None]
            summary.append(dict(dataset=ds,treatment=t,cells=len(group),
                matched_B0_router={k:stat([r['matched_B0_router'].get(k) for r in group]) for k in ('trust_auroc','preference_auroc','D_Prec_01','D_Prec_05','D_Prec_20','D_deployed_precision','deployed_precision')},
                matched_B0_corrective={k:stat([r['matched_B0_corrective'].get(k) for r in group]) for k in ('corrective_precision','corrective_count','net_corrections','corrective_right_minus_wrong','net_per_1000_query')},
                common_defined_cells=len(c),common_defined_B0_corrective_precision=stat([r['matched_B0_corrective']['corrective_precision'] for r in c]),
                common_defined_variant_corrective_precision=stat([r['corrective']['corrective_precision'] for r in c])))
    write(dest/'RESULTS.json',dict(summary=summary,source_report_seal_sha256=digest(report/'COMPLETE.json'),code_sha256=digest(__file__)))
    lines=['# 同名单B0参照','', 'D2的B0在剔除该配置新增训练节点后的相同Query上评估。最终分类eval1000不变。纠正性精度另外同时报告双方都有定义的配置均值，避免不同有效样本数的均值直接比较。',
        '', '| Dataset | Paired treatment | n | B0 Trust | B0 Pref | B0 D P@1% | B0 D accepted prec% | B0 all accepted prec% | B0 corrective prec% | B0 CP defined n | B0 net |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x,scale=1):return 'NA' if x is None else f'{x*scale:.3f}'
    for r in summary:
        m=r['matched_B0_router'];c=r['matched_B0_corrective']
        lines.append('| '+' | '.join([r['dataset'],r['treatment'],str(r['cells']),fmt(m['trust_auroc']['mean']),fmt(m['preference_auroc']['mean']),fmt(m['D_Prec_01']['mean'],100),fmt(m['D_deployed_precision']['mean'],100),fmt(m['deployed_precision']['mean'],100),fmt(c['corrective_precision']['mean'],100),str(c['corrective_precision']['n']),fmt(c['net_corrections']['mean'])])+' |')
    lines+=['','| Dataset | Treatment | Common-defined n | B0 corrective precision% | Variant corrective precision% |','|---|---|---:|---:|---:|']
    for r in summary:lines.append('| '+' | '.join([r['dataset'],r['treatment'],str(r['common_defined_cells']),fmt(r['common_defined_B0_corrective_precision']['mean'],100),fmt(r['common_defined_variant_corrective_precision']['mean'],100)])+' |')
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(dest,dict(all_checks_pass=True,paired_reference_only=True));print('PAIRED_REFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
