"""Slurm-only descriptive paired counts; no training, selection or inference."""
from pathlib import Path
from fractions import Fraction
import os,json,csv,hashlib,statistics,datetime,collections

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'reports/comprehensive-b0-3x2-20261029'
OUT=ROOT/'reports/pseudo-repair-patterns-20261031'
LABELS=['B0','D1_25__fixed','D1_25__retuned','D2a_1000__fixed','D2a_1000__retuned','D4_confidence__fixed','D4_confidence__retuned']
NAMES=dict(zip(LABELS,['B0','D1冻结','D1重调','D2冻结','D2重调','D4冻结','D4重调']))
DS=[('cora','Cora'),('citeseer','Citeseer'),('pubmed','PubMed'),('arxiv','Arxiv'),('ogbn-products','Products')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def mean(x):return statistics.mean(x) if x else None
def csvout(name,rows):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def pct(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'NA'
def frac(r,b):
 p=r['pseudo'];return Fraction(p[b+'_correct'],p[b+'_count']) if p[b+'_count'] else None
def paired_summary(rows):
 out={'pairs':len(rows)}
 for branch in ('all','D'):
  g=[r for r in rows if r[branch+'_delta_pp'] is not None]
  down=[r for r in g if r[branch+'_down']]
  cases=[r for r in down if r['pR_up']]
  robust=[r for r in g if r[branch+'_down_1pp'] and r['pR_up_05pp']]
  out[branch]={'defined':len(g),'pseudo_down':len(down),'down_pR_up':len(cases),'strong_down1_pRup05':len(robust),'cases_cross_zero':sum(r['cross']==0 for r in cases),'cases_gold_only':sum(r['cross']==0 and r['keep']==0 for r in cases)}
 return out
def impact_summary(g):
 return {'models':len(g),'same_accuracy':sum(r['joint_gain_nodes']==0 for r in g),
  'within05pp':sum(abs(r['joint_gain_nodes'])<=5 for r in g),'within1pp':sum(abs(r['joint_gain_nodes'])<=10 for r in g),
  'gain_gt1pp':sum(r['joint_gain_nodes']>10 for r in g),'harm_gt1pp':sum(r['joint_gain_nodes']<-10 for r in g),
  'mean_gain_pp':mean([r['joint_gain_pp'] for r in g]),'median_gain_pp':statistics.median([r['joint_gain_pp'] for r in g]),
  'mean_abs_gain_pp':mean([abs(r['joint_gain_pp']) for r in g]),'mean_changed_pct':mean([r['changed_count']/10 for r in g]),
  'near1_changed_ge5pct':sum(abs(r['joint_gain_nodes'])<=10 and r['changed_count']>=50 for r in g)}

def main():
 assert os.environ.get('SLURM_JOB_ID') and not OUT.exists()
 manifest=read(SRC/'PUBLICATION_SHA256.json');assert sha(SRC/'RESULTS.json')==manifest['RESULTS.json']
 source=read(SRC/'RESULTS.json');rows=source['rows'];by={(r['index'],r['variant']):r for r in rows};assert len(by)==600
 pairs=[];impacts=[];proof=[]
 for i in range(75):
  p=SRC/'cells'/f'{i:02d}'/'eval1000_labels_and_subset.csv';rel=str(p.relative_to(SRC));assert sha(p)==manifest[rel]
  proof.append({'index':i,'file':rel,'sha256':sha(p)})
  blocks=collections.defaultdict(list)
  with p.open() as f:
   for x in csv.DictReader(f):blocks[x['variant']].append(x)
  for v in LABELS:
   r=by[i,v];g=blocks[v];c=r['classification'];assert len(g)==1000 and [int(x['eval_order']) for x in g]==list(range(1000))
   R=H=W=0;jc=nc=rc=0
   for x in g:
    y,j,n,pr=(int(x[k]) for k in ('truth','pJ','pGL_own','pR'))
    jc+=j==y;nc+=n==y;rc+=pr==y
    if j!=n:
     if j==y:R+=1
     elif n==y:H+=1
     else:W+=1
   assert (jc,nc,rc)==(c['eval_joint_correct'],c['eval_without_pR_correct'],c['eval_pR_correct'])
   assert R+H+W==c['old_joint_changed_count'] and R-H==jc-nc
   impacts.append({'index':i,'dataset':r['dataset'],'shot':r['shot'],'seed':r['seed'],'variant':v,
    'pR_pct':rc/10,'pGL_pct':nc/10,'pJ_pct':jc/10,'joint_gain_nodes':jc-nc,'joint_gain_pp':(jc-nc)/10,
    'changed_count':R+H+W,'changed_pct':(R+H+W)/10,'rescued':R,'harmed':H,'wrong_to_wrong':W,'cross':r['config']['cross'],'keep':r['config']['keep']})
   if v=='B0':continue
   base=by[i,'B0_matched_D2' if v.startswith('D2a_') else 'B0'];bc=base['classification']
   a={'index':i,'dataset':r['dataset'],'shot':r['shot'],'seed':r['seed'],'variant':v,'B0_reference':base['variant'],
     'pR_B0_pct':bc['eval_pR_correct']/10,'pR_pct':rc/10,'pR_delta_pp':(rc-bc['eval_pR_correct'])/10,
     'pR_up':rc>bc['eval_pR_correct'],'pR_up_05pp':rc-bc['eval_pR_correct']>=5,
     'pJ_B0_pct':bc['eval_joint_correct']/10,'pJ_pct':jc/10,'pJ_delta_pp':(jc-bc['eval_joint_correct'])/10,
     'cross_B0':base['config']['cross'],'cross':r['config']['cross'],'aux':r['config']['aux'],'keep':r['config']['keep']}
   for branch in ('all','D'):
    q0,q1=frac(base,branch),frac(r,branch);defined=q0 is not None and q1 is not None;delta=q1-q0 if defined else None
    a.update({branch+'_B0_pct':float(q0*100) if q0 is not None else None,branch+'_pct':float(q1*100) if q1 is not None else None,
     branch+'_delta_pp':float(delta*100) if defined else None,branch+'_down':bool(defined and delta<0),branch+'_down_1pp':bool(defined and delta<=-Fraction(1,100)),
     branch+'_count_B0':base['pseudo'][branch+'_count'],branch+'_count':r['pseudo'][branch+'_count']})
   if v.startswith('D2a_'):assert r['query_ids_sha256']==base['query_ids_sha256']
   pairs.append(a)
 assert len(pairs)==450 and len(impacts)==525
 pair_summary={v:paired_summary([r for r in pairs if r['variant']==v]) for v in LABELS[1:]}
 pair_summary['ALL']=paired_summary(pairs)
 impact_summaries={v:impact_summary([r for r in impacts if r['variant']==v]) for v in LABELS}
 impact_summaries['ALL']=impact_summary(impacts)
 by_dataset={ds:{'pairs':paired_summary([r for r in pairs if r['dataset']==ds]),'impact':impact_summary([r for r in impacts if r['dataset']==ds])} for ds,name in DS}
 cases=[r for r in pairs if r['all_down'] and r['pR_up']]
 dcases=[r for r in pairs if r['D_down'] and r['pR_up']]
 robust=[r for r in cases if r['all_down_1pp'] and r['pR_up_05pp']]
 examples=[]
 for ds,name in DS:
  examples.extend(sorted([r for r in robust if r['dataset']==ds],key=lambda r:(r['all_delta_pp'],-r['pR_delta_pp'],r['variant'],r['shot'],r['seed']))[:3])
 near=sorted([r for r in impacts if abs(r['joint_gain_nodes'])<=10 and r['changed_count']>=50],key=lambda r:(-r['changed_count'],r['dataset'],r['shot'],r['seed'],r['variant']))
 lines=['# 伪标签下降但pR上升、以及pGL与pJ接近：全量描述性核查','',
 '生成时间：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'',
 '范围：B0及六配置，每种75格，525个模型。第一问比较六配置各自与同格B0，共450次配对；这些不是450个独立试验，同dataset/seed跨shot及不同Joint设置共享数据/Router。仅统计已冻结结果，不训练、不调参、不作因果或显著性结论。',
 '第一问主伪标签精度=完整Query上实际接受的agreement+disagreement合计精度，另列实际接受disagreement。D2和B0在剔除新增训练节点的相同Query名单上比较。pR及分类均为原eval1000。伪标签下降用正确数/接受数的精确有理数比较；pR上升用1000节点的整数正确数比较。',
 '第一问严格条件：伪标签精度下降且pR准确率上升；明显条件：伪标签下降至少1pp且pR上升至少0.5pp。两个Joint设置的Router质量相同，pR及所选cross/keep可以不同；cross=0或cross=keep=0的例子单列。',
 '第二问delta=accuracy(pJ)-accuracy(自身pGL)，统计abs(delta)<=0.5pp、<=1pp和精确相等。阈值仅用于描述，不是实验接纳门槛。pGL和pJ从同一checkpoint拆分，没有重训无pR模型；不能把准确率接近直接解释成pR对预测没影响。',
 '本次另逐节点检查pJ!=pGL：改对Rj、改错Hj、错换错Wj，变化数=Rj+Hj+Wj，准确率净变化=(Rj-Hj)/1000。这是最终预测变化，不是伪标签C/R/H/W。','',
 '## 第一问：各配置的发生频率','',
 table(['配置','合计伪标签下降','下降且pR上升/全部','下降后pR上升/下降格','明显条件','分歧下降且pR上升','主案例cross=0 / gold-only'],[[NAMES[v],pct(s['all']['pseudo_down'],75),pct(s['all']['down_pR_up'],75),pct(s['all']['down_pR_up'],s['all']['pseudo_down']),s['all']['strong_down1_pRup05'],s['D']['down_pR_up'],str(s['all']['cases_cross_zero'])+' / '+str(s['all']['cases_gold_only'])] for v,s in pair_summary.items() if v!='ALL']),
 '', '总体：'+json.dumps(pair_summary['ALL'],ensure_ascii=False),'',
 '## 第一问：按数据集','',table(['数据集','合计下降格','下降且pR上升','下降后pR上升比例','明显条件','分歧下降且pR上升'],[[name,d['pairs']['all']['pseudo_down'],pct(d['pairs']['all']['down_pR_up'],90),pct(d['pairs']['all']['down_pR_up'],d['pairs']['all']['pseudo_down']),d['pairs']['all']['strong_down1_pRup05'],d['pairs']['D']['down_pR_up']] for ds,name in DS for d in [by_dataset[ds]]]),
 '', '## 第一问：明显案例示例（每数据集最多3例）','',
 table(['数据集','shot/seed','配置','合计伪标签B0→当前%','伪标签变化pp','pR B0→当前%','pR变化pp','完整分类变化pp','cross B0→当前','keep'],[[r['dataset'],str(r['shot'])+'/'+str(r['seed']),NAMES[r['variant']],f'{r["all_B0_pct"]:.2f}→{r["all_pct"]:.2f}',f'{r["all_delta_pp"]:+.2f}',f'{r["pR_B0_pct"]:.2f}→{r["pR_pct"]:.2f}',f'{r["pR_delta_pp"]:+.2f}',f'{r["pJ_delta_pp"]:+.2f}',f'{r["cross_B0"]:g}→{r["cross"]:g}',r['keep']] for r in examples]),
 '', '## 第一问：全部符合严格条件的配置定位','']
 for ds,name in DS:
  groups=[]
  for v in LABELS[1:]:
   for shot in (3,5,10):
    seeds=[str(r['seed']) for r in cases if (r['dataset'],r['variant'],r['shot'])==(ds,v,shot)]
    if seeds:groups.append([NAMES[v],shot,', '.join(seeds)])
  lines+=['','### '+name,'',table(['配置','shot','seeds'],groups) if groups else '无。']
 lines+=['','## 第二问：各模型中pGL与pJ准确率接近的频率','',
 table(['配置','完全相等','相差≤0.5pp','相差≤1pp','提升>1pp','下降>1pp','平均净提升pp','平均标签变化%','差≤1pp但标签变化≥5%'],[[NAMES.get(v,'全部'),pct(s['same_accuracy'],s['models']),pct(s['within05pp'],s['models']),pct(s['within1pp'],s['models']),s['gain_gt1pp'],s['harm_gt1pp'],f'{s["mean_gain_pp"]:+.2f}',f'{s["mean_changed_pct"]:.2f}',s['near1_changed_ge5pct']] for v,s in impact_summaries.items()]),
 '', '## 第二问：按数据集（七配置合计各105模型）','',
 table(['数据集','相差≤0.5pp','相差≤1pp','平均净提升pp','平均标签变化%','差≤1pp但标签变化≥5%'],[[name,pct(s['within05pp'],s['models']),pct(s['within1pp'],s['models']),f'{s["mean_gain_pp"]:+.2f}',f'{s["mean_changed_pct"]:.2f}',s['near1_changed_ge5pct']] for ds,name in DS for s in [by_dataset[ds]['impact']]]),
 '', '## 第二问：净准确率变化小、但预测变化较多的例子','',
 table(['数据集','shot/seed','配置','pGL→pJ%','准确率变化pp','标签变化数/1000','改对Rj','改错Hj','错换错Wj'],[[r['dataset'],str(r['shot'])+'/'+str(r['seed']),NAMES[r['variant']],f'{r["pGL_pct"]:.2f}→{r["pJ_pct"]:.2f}',f'{r["joint_gain_pp"]:+.2f}',r['changed_count'],r['rescued'],r['harmed'],r['wrong_to_wrong']] for r in near[:12]]),
 '', '## 原始结果、代码与限制','',
 '- [全部450次配对及伪标签、pR、Joint差值](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/pseudo-repair-patterns-20261031/paired_all450.csv)。',
 '- [合计伪标签下降且pR上升的全部案例](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/pseudo-repair-patterns-20261031/cases_all_precision.csv)；[分歧口径全部案例](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/pseudo-repair-patterns-20261031/cases_disagreement_precision.csv)。',
 '- [525模型逐节点变化核查汇总](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/pseudo-repair-patterns-20261031/residual_impact_all525.csv)；[结构化汇总](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/pseudo-repair-patterns-20261031/RESULTS.json)。',
 '- [汇总代码](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/pseudo_repair_patterns_20261031/report.py)。',
 '- 来源：reports/comprehensive-b0-3x2-20261029/RESULTS.json及其75组eval1000_labels_and_subset.csv；manifest及逐文件SHA256均校验。',
 '- 伪标签精度下降可伴随接受节点/数量变化，不是固定节点上统一加噪；重调同时改变多项超参。不能据此说伪标签下降导致pR上升，也不能用跨数据集原始难度差异推出机制。',
 '- cross=0不使用分歧伪标签损失；cross=keep=0时Joint只有support监督及可选support辅助损失。这些案例仍保留并标注，不能据此归因伪标签质量。',
 '- 历史5-shot的选择来源限制仍保留，详见原完整报告；不在本次重新选择模型。','']
 OUT.mkdir(parents=True)
 csvout('paired_all450.csv',pairs);csvout('cases_all_precision.csv',cases);csvout('cases_disagreement_precision.csv',dcases);csvout('residual_impact_all525.csv',impacts)
 result={'all_checks_pass':True,'slurm_job':os.environ['SLURM_JOB_ID'],'source_sha256':sha(SRC/'RESULTS.json'),'source_manifest_sha256':sha(SRC/'PUBLICATION_SHA256.json'),'source_code_sha256':sha(Path(__file__)),'paired_summary':pair_summary,'impact_summary':impact_summaries,'by_dataset':by_dataset,'examples':examples,'cancellation_examples':near[:12],'pair_rows':pairs,'impact_rows':impacts,'node_evidence':proof,'definitions':{'pseudo_primary':'actual accepted agreement+disagreement precision on full Query; D2 paired matched B0','pseudo_secondary':'actual accepted disagreement precision','near_accuracy_pp':[0,.5,1],'near_node_changes':'pJ != own pGL, original eval1000','strong_case':'pseudo precision decreases >=1pp and pR increases >=0.5pp'},'training':False,'selection':False}
 (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'REPORT.md').write_text('\n'.join(lines))
 seal={'all_checks_pass':True,'pairs':450,'models':525,'slurm_job':os.environ['SLURM_JOB_ID'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'COMPLETE.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps(seal),flush=True)

if __name__=='__main__':main()
