"""Read-only Slurm aggregation of sealed metrics; equal-weight shot averages."""
from pathlib import Path
import json, hashlib, os, statistics, datetime

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'reports/comprehensive-b0-3x2-20261029'
OUT=ROOT/'reports/focused-shot-average-20261029'
DATASETS=[('arxiv','Arxiv','D1_25'),('pubmed','PubMed','D4_confidence'),('ogbn-products','Products','D4_confidence')]
METRICS={
 'Trust':('router_raw','trust_auroc'),'Pref':('router_raw','preference_auroc'),
 **{f'P{k}':('router_calibrated',f'D_Prec_{k:02d}') for k in (1,5,20,100)},
 **{f'{b}_precision':('pseudo',f'{b}_precision') for b in ('A','D','all')},
 'CP':('corrective_D','precision'),
 **{k:('classification',k) for k in ('pR_accuracy','without_pR_accuracy','joint_accuracy','subset_fraction','subset_pR_accuracy','subset_without_pR_accuracy','subset_joint_accuracy')}
}
COUNTS={**{f'{b}_count':('pseudo',f'{b}_count') for b in ('A','D','all')},
 **{k:('corrective_D',k) for k in ('C','R','H','W','net','right_minus_wrong')},'S':('classification','subset_count')}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def get(row,path):
 for k in path: row=row[k]
 return row
def mean(xs):
 xs=[x for x in xs if x is not None]
 return statistics.mean(xs) if xs else None
def summarize(rows,ds,v,shot):
 assert len(rows)==(15 if shot=='AVG' else 5)
 metrics={}
 for k,path in METRICS.items():
  vals=[get(r,path) for r in rows]
  byshot={str(s):mean([get(r,path) for r in rows if r['shot']==s]) for s in (3,5,10)}
  value=mean(vals)
  if shot=='AVG':
   value=statistics.mean(byshot.values()) if all(x is not None for x in byshot.values()) else None
  metrics[k]={'mean':value,'n':sum(x is not None for x in vals),'N':len(vals),'shot_means':byshot}
 counts={k:{'total':sum(get(r,path) for r in rows),'mean':statistics.mean(get(r,path) for r in rows)} for k,path in COUNTS.items()}
 assert counts['C']['total']==sum(counts[k]['total'] for k in ('R','H','W'))
 return {'dataset':ds,'variant':v,'shot':shot,'metrics':metrics,'counts':counts,'pooled_CP':counts['R']['total']/counts['C']['total'] if counts['C']['total'] else None,'keys':[r['key'] for r in rows]}

def table(headers,rows):
 assert all(len(r)==len(headers) for r in rows)
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def render(summary):
 lines=['# D1–Arxiv、D4–PubMed／Products：按数据集集中展示及跨shot平均','',
 '生成时间：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'',
 '仅对上一版封存的逐配置指标重新汇总，不重训、不推理、不改模型/标签/评估名单。D1为numeric25% Router；冻结/重调仅指Joint超参。',
 '每个shot先对seeds42–46取均值，再将3/5/10-shot三个均值等权平均。AUROC单位0–1，其余精度/占比单位%。表中仅显示均值，逐shot标准差见原完整报告。',
 'C为空的seed不计算纠正性精度；先计算各shot有效seed均值，再对三个shot等权平均，不能把不同shot的有效seed直接混在一起平均。每格注明有效n/N。若某shot全部无定义，则跨shot平均记NA，不填0。',
 '所有数量均列累计节点—配置次数；单shot累计5组，平均行累计15组，不是去重节点数，也不是平均每shot数量。伪标签数量另附每配置均数。',
 'Router和伪标签使用完整Query；纠正性仅使用实际接受的disagreement，参照每个模型自身pGL。分类始终为原eval1000；S={argmax pR != argmax自身pGL}。各模型S不同，子集精度不是同名单配对比较。',
 'C=R+H+W：R改对，H改错，W错换错；纠正性精度R/C，净纠正R−H。累计R/C另列，与三个shot等权平均不同。',
 '', '[原完整报告、实现对照与指标定义](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/comprehensive-b0-3x2-20261029/REPORT.md)。','']
 def m(s,k):
  d=s['metrics'][k];x=d['mean'];scale=1 if k in ('Trust','Pref') else 100
  return 'NA' if x is None else f'{x*scale:.{4 if scale==1 else 2}f}'
 def count(s,k):
  c=s['counts'][k];return f'{c["total"]:,} / {c["mean"]:.2f}'
 def label(s,t):
  return ['**三shot平均**' if s['shot']=='AVG' else str(s['shot']), 'B0' if s['variant']=='B0' else t+('冻结' if s['variant'].endswith('__fixed') else '重调')]
 for ds,name,treat in DATASETS:
  t='D1' if treat=='D1_25' else 'D4';vv=['B0',treat+'__fixed',treat+'__retuned']
  group=[next(s for s in summary if (s['dataset'],s['shot'],s['variant'])==(ds,shot,v)) for shot in (3,5,10,'AVG') for v in vv]
  lines+=['','## '+name+' · '+t,'','### Router：校准前AUROC、校准后D-PR','', 'Joint冻结/重调共用同一Router，此表只列一次。','',
   table(['shot','Router','Trust','Pref','D-PR@1%','@5%','@20%','@100%'],[label(s,t)[:1]+['B0' if s['variant']=='B0' else t]+[m(s,k) for k in ('Trust','Pref','P1','P5','P20','P100')] for s in group if not s['variant'].endswith('__retuned')]),
   '', '### 实际接受伪标签：精度与数量','',
   table(['shot','Router','A精度%','A数量Σ/每配置均数','D精度%','D数量Σ/每配置均数','合计精度%','合计数量Σ/每配置均数'],[label(s,t)[:1]+['B0' if s['variant']=='B0' else t]+[m(s,'A_precision'),count(s,'A_count'),m(s,'D_precision'),count(s,'D_count'),m(s,'all_precision'),count(s,'all_count')] for s in group if not s['variant'].endswith('__retuned')]),
   '', '### 实际接受分歧标签：自身pGL纠正性','',
   table(['shot','配置','纠正性精度%（有效n/N）','累计R/C%','C / R / H / W（累计）','净R−H','R−H−W'],[label(s,t)+[m(s,'CP')+f' ({s["metrics"]["CP"]["n"]}/{s["metrics"]["CP"]["N"]})','NA' if s['pooled_CP'] is None else f'{s["pooled_CP"]*100:.2f}',' / '.join(f'{s["counts"][k]["total"]:,}' for k in ('C','R','H','W')),s['counts']['net']['total'],s['counts']['right_minus_wrong']['total']] for s in group]),
   '', '### 分类：eval1000及S子集','', '三项顺序统一为pR / 自身pGL（w/o pR）/ 完整pJ。','',
   table(['shot','配置','eval1000三项准确率%','S占比%','S人数Σ/每配置均数','S内三项准确率%'],[label(s,t)+[' / '.join(m(s,k) for k in ('pR_accuracy','without_pR_accuracy','joint_accuracy')),m(s,'subset_fraction'),count(s,'S'),' / '.join(m(s,k) for k in ('subset_pR_accuracy','subset_without_pR_accuracy','subset_joint_accuracy'))] for s in group])]
 lines+=['','## 来源和复核','',
 '源文件：reports/comprehensive-b0-3x2-20261029/RESULTS.json；源SHA256：'+sha(SOURCE/'RESULTS.json')+'。',
 '汇总代码：[experiments/focused_shot_average_20261029/report.py](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/focused_shot_average_20261029/report.py)。',
 '新汇总JSON：[RESULTS.json](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/focused-shot-average-20261029/RESULTS.json)。',
 '历史5-shot模型选择来源限制见原完整报告；本次仅做描述性汇总，不使用Query真值调参。','']
 return '\n'.join(lines)

def main():
 assert os.environ.get('SLURM_JOB_ID'),'Numerical work must use Slurm'
 assert not OUT.exists(),'Do not overwrite a prior report'
 manifest=json.loads((SOURCE/'PUBLICATION_SHA256.json').read_text())
 assert sha(SOURCE/'RESULTS.json')==manifest['RESULTS.json']
 original=json.loads((SOURCE/'RESULTS.json').read_text());rows=original['rows'];summary=[]
 for ds,name,t in DATASETS:
  for shot in (3,5,10,'AVG'):
   for v in ('B0',t+'__fixed',t+'__retuned'):
    g=[r for r in rows if r['dataset']==ds and r['variant']==v and (shot=='AVG' or r['shot']==shot)]
    s=summarize(g,ds,v,shot);summary.append(s)
    if shot!='AVG':
     old=next(x for x in original['summary'] if (x['dataset'],x['shot'],x['variant'])==(ds,shot,v))
     mapping={'Trust':'trust_raw','Pref':'preference_raw','CP':'corrective_precision',**{f'P{k}':f'D_P{k:03d}' for k in (1,5,20,100)},**{f'{b}_precision':f'pseudo_{b}_precision' for b in ('A','D','all')}}
     for k,d in s['metrics'].items():
      prev=old['metrics'][mapping.get(k,k)]['mean']
      assert (d['mean'] is None and prev is None) or abs(d['mean']-prev)<1e-12,(ds,shot,v,k)
 OUT.mkdir(parents=True)
 result={'all_checks_pass':True,'summary':summary,'source_sha256':sha(SOURCE/'RESULTS.json'),'source_manifest_sha256':sha(SOURCE/'PUBLICATION_SHA256.json'),'source_code_sha256':sha(Path(__file__)),'slurm_job':os.environ['SLURM_JOB_ID'],'shot_average':'equal-weight average of per-shot valid-seed means; counts total across15 cells'}
 (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'REPORT.md').write_text(render(summary))
 (OUT/'COMPLETE.json').write_text(json.dumps({'all_checks_pass':True,'slurm_job':os.environ['SLURM_JOB_ID'],'files':{n:sha(OUT/n) for n in ('RESULTS.json','REPORT.md')}},indent=2)+'\n')
 print(json.dumps({'all_checks_pass':True,'output':str(OUT),'report_sha256':sha(OUT/'REPORT.md')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
