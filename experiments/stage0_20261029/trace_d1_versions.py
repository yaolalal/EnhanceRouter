"""Trace frozen versus numeric D1 precision from sealed full-Query reports."""
from pathlib import Path
import os,json,hashlib,statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/d1-version-trace-20261029'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID'),'All numerical work requires Slurm'
    src=ROOT/'reports/full-query-20261027'
    manifest=json.loads((src/'PUBLICATION_SHA256.json').read_text())
    for name in ('RESULTS.json','COMPLETE.json'):
        assert digest(src/name)==manifest[name],name
    data=json.loads((src/'RESULTS.json').read_text());rows=data['per_cell']
    selected=[r for r in rows if r['variant'] in ('b0','d1_frozen/0.25','d1_numeric/0.25')]
    assert len(selected)==225
    index={(r['key'],r['variant']):r for r in selected}
    for r in selected:
        b=index[(r['key'],'b0')]
        assert r['query_ids_sha256']==b['query_ids_sha256']
        assert r['original_query_ids_sha256']==b['original_query_ids_sha256']
        assert r['final_eval_ordered_ids_sha256']==b['final_eval_ordered_ids_sha256']
        assert r['removed_training_parent_count']==0
    summary=[]
    for dataset in ('ALL','cora','citeseer','pubmed','arxiv','ogbn-products'):
        for variant in ('b0','d1_frozen/0.25','d1_numeric/0.25'):
            g=[r for r in selected if r['variant']==variant and (dataset=='ALL' or r['dataset']==dataset)]
            m={};delta={};defined={}
            for field in ('D_deployed_precision','deployed_precision','D_Prec_01','D_Prec_05','D_Prec_20'):
                valid=[r for r in g if r[field] is not None]
                assert len(valid)==len(g)
                m[field]=100*statistics.mean(r[field] for r in valid)
                delta[field]=statistics.mean(100*(r[field]-index[(r['key'],'b0')][field]) for r in valid)
                defined[field]=len(valid)
            summary.append(dict(dataset=dataset,variant=variant,cells=len(g),means_percent=m,delta_vs_B0_pp=delta,defined=defined))
    OUT.mkdir(exist_ok=False)
    result=dict(all_checks_pass=True,source_report_sha256=digest(src/'RESULTS.json'),source_manifest_sha256=digest(src/'PUBLICATION_SHA256.json'),code_sha256=digest(Path(__file__)),slurm_job=os.environ['SLURM_JOB_ID'],scope='complete same Query, all75 configurations; no model training or selection',rows=selected,summary=summary)
    (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# D1冻结版与数值重调版溯源','','来源为已封存的完整Query报告；75组配置，Arxiv为15组。均值按配置等权计算，原始Query名单及eval1000哈希逐组相同。仅重新汇总已有计数，没有重训或重新选择模型。','','| Dataset | Router version | n | 接受分歧精度% | 相对B0 pp | 全部接受精度% | 相对B0 pp |','|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        m=s['means_percent'];d=s['delta_vs_B0_pp']
        lines.append(f"| {s['dataset']} | {s['variant']} | {s['cells']} | {m['D_deployed_precision']:.4f} | {d['D_deployed_precision']:+.4f} | {m['deployed_precision']:.4f} | {d['deployed_precision']:+.4f} |")
    lines+=['','旧d1_frozen/0.25：保留25%目标support父节点及其六视图，固定B0数值超参、完整已拟合输入变换、输出归一化、校准、路由规则和接受比例，仅重拟合受影响评分器权重；当时没有额外训练Joint。',
        '新d1_numeric/0.25：同样按seed+913抽取25%父节点，固定特征模式/训练组合/模型族/已拟合输入变换/专家/增强/源数据；允许按原validation流程重选评分器数值超参、校准与路由策略。',
        '3×2中的D1_25__fixed及D1_25__retuned均使用新的numeric25% Router；fixed仅指Joint超参冻结，不能把它解释成旧的冻结Router实验。',
        '', '[完整Query来源](../full-query-20261027/README.md)；[3×2设置](../../docs/JOINT_FACTORIAL_3X2.md)；[旧冻结入口](../../experiments/stage0_20261026/compute.py)；[新numeric入口](../../experiments/stage0_20261027/d1_numeric.py)。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'PUBLICATION_SHA256.json').write_text(json.dumps({p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file()},indent=2)+'\n')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
