"""Slurm-only read-only export of selected Joint cross, all 525 frozen models."""
from pathlib import Path
import os,json,hashlib,csv,collections,datetime

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'reports/comprehensive-b0-3x2-20261029'
OUT=ROOT/'reports/cross-all75-20261031'
LABELS=['B0','D1_25__fixed','D1_25__retuned','D2a_1000__fixed','D2a_1000__retuned','D4_confidence__fixed','D4_confidence__retuned']
NAMES=['B0','D1冻结','D1重调','D2冻结','D2重调','D4冻结','D4重调']
DS=[('cora','Cora'),('citeseer','Citeseer'),('pubmed','PubMed'),('arxiv','Arxiv'),('ogbn-products','Products')]
NUMERIC={'dim','cross','lr','gate_lr','warmup','aux','keep'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def table(h,rows):
 assert all(len(r)==len(h) for r in rows)
 return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def writecsv(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 assert os.environ.get('SLURM_JOB_ID')
 assert not OUT.exists(),'Preserve prior outputs'
 manifest=read(SOURCE/'PUBLICATION_SHA256.json')
 assert sha(SOURCE/'RESULTS.json')==manifest['RESULTS.json']
 original=read(SOURCE/'RESULTS.json');assert original['all_checks_pass']
 rows=[r for r in original['rows'] if r['variant'] in LABELS];assert len(rows)==525
 by={(r['dataset'],r['shot'],r['seed'],r['variant']):r for r in rows};assert len(by)==525
 evidence=[];wide=[];long=[];counts={};changes={}
 for ds,name in DS:
  for shot in (3,5,10):
   for seed in range(42,47):
    w={'dataset':ds,'shot':shot,'seed':seed};base=by[(ds,shot,seed,'B0')]['config']
    for v in LABELS:
     r=by[(ds,shot,seed,v)];cfg=r['config'];cross=cfg['cross'];assert cross in (0.,.25,1.)
     if v.endswith('__fixed'):assert cfg==base,(r['key'],v,'fixed configuration changed')
     elif v!='B0':assert {k:x for k,x in cfg.items() if k not in NUMERIC}=={k:x for k,x in base.items() if k not in NUMERIC}
     model=Path(r['source']['model_path']);sealpath=model/'COMPLETE.json';seal=read(sealpath)
     assert sha(sealpath)==r['source']['model_seal_sha256']
     cpath=model/'config.json';assert sha(cpath)==seal['files']['config.json']
     saved=read(cpath);actual=saved.get('joint') or saved.get('parameters',{}).get('joint_config');assert actual==cfg
     selection=model/'joint-selection.json';sel=None;selhash=None
     if selection.exists():
      assert sha(selection)==seal['files']['joint-selection.json'];sel=read(selection);selhash=sha(selection);assert sel['config']==cfg
     method=('inherited historical B0 selection (55 cells)' if seed<=44 or shot==5 else 'new native validation search (20 cells)') if v=='B0' else ('frozen corresponding B0 full Joint configuration' if v.endswith('__fixed') else 'native four-stage numeric Joint search on own validation')
     receipt={'dataset':ds,'shot':shot,'seed':seed,'variant':v,'cross':cross,'config':cfg,'model_path':str(model),'model_seal_sha256':sha(sealpath),'config_sha256':sha(cpath),'selection_path':str(selection) if sel else None,'selection_sha256':selhash,'selection_mode':method,'selection_record':sel}
     evidence.append(receipt);w[v]=cross
     long.append({'dataset':ds,'shot':shot,'seed':seed,'variant':v,'cross':cross,'B0_cross':base['cross'],'aux':cfg['aux'],'keep':cfg['keep'],'selection_mode':method,'model_path':str(model),'config_sha256':sha(cpath),'model_seal_sha256':sha(sealpath)})
    wide.append(w)
 for v in LABELS:
  vals=[r['config']['cross'] for r in rows if r['variant']==v];assert len(vals)==75
  counts[v]={str(x):vals.count(x) for x in (0.,.25,1.)}
  assert sum(counts[v].values())==75
  if v!='B0':
   pairs=[(w['B0'],w[v]) for w in wide]
   changes[v]={'changed':sum(a!=b for a,b in pairs),'lowered':sum(b<a for a,b in pairs),'raised':sum(b>a for a,b in pairs),'transitions':{f'{a:g}->{b:g}':sum(x==a and y==b for x,y in pairs) for a in (0.,.25,1.) for b in (0.,.25,1.)}}
 lines=['# B0与六种配置的Joint cross：全部75格','',
 '生成时间（Asia/Shanghai）：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'',
 '本次只读封存配置，不训练、不重新选择超参数。5数据集×3shots×5seeds=75格；每格七配置，共525个cross取值。已逐项核对模型完成封存、config.json哈希、报告配置以及可用的joint-selection.json。',
 '', '## cross如何确定','',
 'cross是Joint分歧伪标签损失的超参数，不是Router接受比例、预测融合alpha/beta，也不是梯度自动学习的参数。每个分歧专家池的损失系数为cross/2；cross=0时两个分歧池不进入Joint损失。',
 '原生Joint搜索的cross候选为{0,0.25,1}。第一阶段与dim、lr联合搜索；第二阶段围绕胜出配置搜索其他设置；第三阶段再次尝试cross候选；第四阶段包含cross=0等移除项。候选依据各自validation selection子集的完整分类accuracy排序，其次NLL，再按配置ID打破平局。每个候选内部也用validation选择checkpoint。不是按Query伪标签精度选择cross，不是单独只调cross。',
 'B0的75格由55格历史选中配方和20格新增原生验证搜索组成，保留原选值，不在本次重选。历史5-shot的CHOICES.json标记selection_uses_query_truth=true，因此不能把全部历史B0的选值宣称为纯validation选择；本次原样披露此已知来源限制。',
 'D1冻结、D2冻结、D4冻结：完整Joint配置逐格继承对应B0，故cross应逐格相同。D1重调、D2重调、D4重调：各自Router固定后，使用同一原生四阶段验证流程重选允许的数值项（dim/cross/lr/gate_lr/warmup/aux/keep），其他Joint设置固定B0。重调可能仍选中与B0相同的cross。',
 'D1为numeric25% Router；D2为随机a1000；D4为confidence_router。冻结仅指Joint超参，不冻结Joint模型权重。',
 '', '## 75格取值分布','',table(['配置','cross=0','cross=0.25','cross=1','相对B0改变格数'],[[NAMES[i],counts[v]['0.0'],counts[v]['0.25'],counts[v]['1.0'],0 if v=='B0' else changes[v]['changed']] for i,v in enumerate(LABELS)]),
 '', '## 全部逐格取值','', '每行固定dataset、shot、seed，七列并排。每种配置恰好75个取值。','']
 for ds,name in DS:
  lines+=['','### '+name,'',table(['shot','seed',*NAMES],[[w['shot'],w['seed'],*[f'{w[v]:g}' for v in LABELS]] for w in wide if w['dataset']==ds])]
 lines+=['','## 代码和结果定位','',
 '- [cross进入损失](https://github.com/yaolalal/EnhanceRouter/blob/main/src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L84)。',
 '- [候选cross=0/0.25/1](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/stage0_20261026/native_joint_search.py#L10)；[验证集候选排序及四阶段](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/stage0_20261026/native_joint_search.py#L49)。',
 '- [3×2冻结/重调入口](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/stage0_20261029/factorial_joint.py#L123)；[数值项约束](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/stage0_20261027/d1_numeric.py#L21)。',
 '- [B0历史选择限制](https://github.com/yaolalal/EnhanceRouter/blob/main/docs/FAIRROUTER_BASELINE.md#L33)。',
 '- [75行宽表](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/cross-all75-20261031/cross_75_wide.csv)；[525行长表，含aux/keep及模型来源](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/cross-all75-20261031/cross_525_long.csv)。',
 '- [全部配置、来源哈希及选择记录](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/cross-all75-20261031/RESULTS.json)。',
 '- 原报告：reports/comprehensive-b0-3x2-20261029/RESULTS.json；SHA256：'+sha(SOURCE/'RESULTS.json')+'。',
 '- 本次Slurm作业：'+os.environ['SLURM_JOB_ID']+'；导出代码SHA256：'+sha(Path(__file__))+'。','']
 OUT.mkdir(parents=True)
 writecsv(OUT/'cross_75_wide.csv',wide);writecsv(OUT/'cross_525_long.csv',long)
 result={'all_checks_pass':True,'cells':75,'models':525,'labels':LABELS,'counts':counts,'changes_vs_B0':changes,'wide':wide,'evidence':evidence,'source_sha256':sha(SOURCE/'RESULTS.json'),'source_manifest_sha256':sha(SOURCE/'PUBLICATION_SHA256.json'),'code_sha256':sha(Path(__file__)),'slurm_job':os.environ['SLURM_JOB_ID'],'training':False,'selection':False}
 (OUT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 seal={'all_checks_pass':True,'cells':75,'models':525,'slurm_job':os.environ['SLURM_JOB_ID'],'files':{n:sha(OUT/n) for n in ('REPORT.md','RESULTS.json','cross_75_wide.csv','cross_525_long.csv')}}
 (OUT/'COMPLETE.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps(seal),flush=True)

if __name__=='__main__':main()
