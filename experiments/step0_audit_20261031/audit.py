"""Read-only Slurm audit of all 525 selected accuracy checkpoints and history."""
from pathlib import Path
import os,sys,json,csv,hashlib,statistics,datetime,gc

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'reports/comprehensive-b0-3x2-20261029'
COMP=ROOT/'reports/joint-factorial-components-20261029'
PAT=ROOT/'reports/pseudo-repair-patterns-20261031'
OUT=ROOT/'runs/step0-audit-20261031-v1'
PUB=ROOT/'reports/step0-audit-20261031'
DS=[('cora','Cora'),('citeseer','Citeseer'),('pubmed','PubMed'),('arxiv','Arxiv'),('ogbn-products','Products')]
LABELS=['B0','D1_25__fixed','D1_25__retuned','D2a_1000__fixed','D2a_1000__retuned','D4_confidence__fixed','D4_confidence__retuned']
NAMES=dict(zip(LABELS,['B0','D1冻结','D1重调','D2冻结','D2重调','D4冻结','D4重调']))
def read(p):return json.loads(p.read_text())
def sha(p,kind='sha256'):
 h=hashlib.new(kind)
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def source():
 m=read(SRC/'PUBLICATION_SHA256.json');assert sha(SRC/'RESULTS.json')==m['RESULTS.json']
 d=read(SRC/'RESULTS.json');assert d['all_checks_pass'];return d
def csvout(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def key(x):return (-x['accuracy'],x['nll'],x['step'])

def worker(index):
 assert os.environ.get('SLURM_JOB_ID')
 sys.path.insert(0,str(ROOT/'src'))
 import torch
 from eargtc.joint_explore.joint_purf_v10_v1.model import UnifiedRepair
 torch.set_num_threads(2)
 ds=DS[index][0];dest=OUT/ds;assert not dest.exists(),'Preserve interrupted output'
 dest.mkdir(parents=True)
 source_data=source();rr=[r for r in source_data['rows'] if r['dataset']==ds and r['variant'] in LABELS];assert len(rr)==105
 cm=read(COMP/'PUBLICATION_SHA256.json');cpub=COMP/'datasets'/ds/'RESULTS.json';assert sha(cpub)==cm[f'datasets/{ds}/RESULTS.json']
 published={(r['key'],r['variant']):r for r in read(cpub)['rows']}
 results=[];histories=[]
 for r in rr:
  v=r['variant'];folder=Path(r['source']['model_path']);assert sha(folder/'COMPLETE.json')==r['source']['model_seal_sha256']
  seal=read(folder/'COMPLETE.json')
  if v=='B0':
   receipt=r['source']['B0_inputs']['checkpoint'];p=Path(receipt['path']);kind=receipt['hash_algorithm'];expected=receipt['hash']
  else:p=folder/'checkpoint.pt';kind='sha256';expected=seal['files']['checkpoint.pt']
  assert sha(p,kind)==expected,(r['key'],v,'checkpoint hash')
  cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
  assert cp['config']==r['config'] and cp['training_seed']==r['seed']
  h=cp['history'];chosen=cp['checkpoints']['accuracy'];state=cp['states']['accuracy'];first=h[0]
  assert first['step']==0 and all(a['step']<b['step'] for a,b in zip(h,h[1:]))
  best=min(h,key=key);assert key(best)==key(chosen),(r['key'],v,'history winner mismatch')
  step=chosen['step'];assert step==published[r['key'],v]['selected_step']
  alpha=float(torch.nn.functional.softplus(state['raw_alpha']));beta=float(torch.nn.functional.softplus(state['raw_beta']))
  assert abs(alpha-r['alpha'])<1e-7 and abs(beta-r['beta'])<1e-7
  assert abs(first['alpha']-1)<1e-7 and abs(first['beta']-1)<1e-7
  # Reconstruct the exact initialization from the original seed and dimensions.
  torch.manual_seed(r['seed'])
  fresh=UnifiedRepair(state['g_project.weight'].shape[1],state['l_project.weight'].shape[1],state['output.weight'].shape[0],r['config']['dim'])
  initial=fresh.state_dict();same_all=all(torch.equal(state[k],initial[k]) for k in initial)
  same_body=all(torch.equal(state[k],initial[k]) for k in initial if k not in ('raw_alpha','raw_beta'))
  if step==0:assert same_all and abs(alpha-1)<1e-7 and abs(beta-1)<1e-7,(r['key'],v,'step0 not original init')
  c=r['classification'];no_exceed=all(x['accuracy']<=first['accuracy'] for x in h[1:])
  tie_improved=any(x['accuracy']==first['accuracy'] and x['nll']<first['nll'] for x in h[1:])
  later_exceed=[x['step'] for x in h[1:] if x['accuracy']>first['accuracy']]
  row={'index':r['index'],'key':r['key'],'dataset':ds,'shot':r['shot'],'seed':r['seed'],'variant':v,
   'selected_step':step,'step0_selected':step==0,'stop_step':h[-1]['step'],'validations':len(h),
   'initial_val_accuracy':first['accuracy'],'selected_val_accuracy':chosen['accuracy'],'max_history_val_accuracy':max(x['accuracy'] for x in h),
   'initial_val_nll':first['nll'],'selected_val_nll':chosen['nll'],'never_accuracy_exceeds_initial':no_exceed,'later_equal_acc_lower_nll':tie_improved,
   'alpha':alpha,'beta':beta,'initial_alpha':first['alpha'],'initial_beta':first['beta'],'pR_weight':1/(alpha+beta+1),
   'selected_state_equals_seed_initialization':same_all,'selected_body_equals_seed_initialization':same_body,
   'pR_pct':c['eval_pR_correct']/10,'pGL_pct':c['eval_without_pR_correct']/10,'pJ_pct':c['eval_joint_correct']/10,
   'joint_gain_pp':(c['eval_joint_correct']-c['eval_without_pR_correct'])/10,
   'joint_gain_nodes':c['eval_joint_correct']-c['eval_without_pR_correct'],'changed_count':c['old_joint_changed_count'],'changed_pct':c['old_joint_changed_count']/10,
   'cross':r['config']['cross'],'aux':r['config']['aux'],'keep':r['config']['keep'],'dim':r['config']['dim'],
   'checkpoint_path':str(p),'checkpoint_hash_algorithm':kind,'checkpoint_hash':expected,'model_path':str(folder),'model_seal_sha256':r['source']['model_seal_sha256']}
  results.append(row);histories.append({'key':r['key'],'variant':v,'checkpoint_path':str(p),'checkpoint_hash':expected,'selected_accuracy_checkpoint':chosen,
   'selected_nll_checkpoint':cp['checkpoints'].get('nll'),'history':h,'later_accuracy_exceeds_initial_steps':later_exceed})
  if step==0 or len(results)%15==0:print('STEP_AUDIT',r['key'],v,step,len(results),flush=True)
  del fresh,initial,cp,state;gc.collect()
 write(dest/'RESULTS.json',{'all_checks_pass':True,'dataset':ds,'rows':results,'histories':histories,'source_sha256':sha(SRC/'RESULTS.json'),'code_sha256':sha(Path(__file__)),'component_source_sha256':sha(cpub),'slurm_job':os.environ['SLURM_JOB_ID']})
 csvout(dest/'per_model.csv',results)
 write(dest/'COMPLETE.json',{'all_checks_pass':True,'models':105,'files':{n:sha(dest/n) for n in ('RESULTS.json','per_model.csv')}})
 print('STEP0_DATASET_PASS',ds,105,flush=True)

def summary(rows):
 z=[r for r in rows if r['step0_selected']];trained=[r for r in rows if not r['step0_selected']]
 def sub(g):
  return {'n':len(g),'within05pp':sum(abs(r['joint_gain_nodes'])<=5 for r in g),'within1pp':sum(abs(r['joint_gain_nodes'])<=10 for r in g),
   'same_accuracy':sum(r['joint_gain_nodes']==0 for r in g),'mean_gain_pp':statistics.mean([r['joint_gain_pp'] for r in g]) if g else None,
   'mean_changed_pct':statistics.mean([r['changed_pct'] for r in g]) if g else None,'mean_pR_pct':statistics.mean([r['pR_pct'] for r in g]) if g else None,
   'max_abs_gain_pp':max([abs(r['joint_gain_pp']) for r in g],default=None)}
 return {'models':len(rows),'step0_count':len(z),'never_acc_exceeds_initial':sum(r['never_accuracy_exceeds_initial'] for r in rows),
  'nonzero_without_acc_improvement':sum(not r['step0_selected'] and r['never_accuracy_exceeds_initial'] for r in rows),
  'zero_still_trained':sum(r['stop_step']>0 for r in z),'zero_equal_init':sum(r['selected_state_equals_seed_initialization'] for r in z),
  'nonzero_alpha_beta_one':sum(not r['step0_selected'] and abs(r['alpha']-1)<1e-7 and abs(r['beta']-1)<1e-7 for r in rows),
  'step0':sub(z),'trained':sub(trained),'all':sub(rows)}
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def p(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else 'NA'
def f(x):return 'NA' if x is None else f'{x:.2f}'

def aggregate():
 assert os.environ.get('SLURM_JOB_ID') and not PUB.exists();rows=[];histories=[];proof=[]
 for ds,name in DS:
  dest=OUT/ds;seal=read(dest/'COMPLETE.json')
  for n,h in seal['files'].items():assert sha(dest/n)==h
  d=read(dest/'RESULTS.json');assert d['all_checks_pass'] and d['code_sha256']==sha(Path(__file__))
  rows.extend(d['rows']);histories.extend(d['histories']);proof.append({'dataset':ds,'seal_sha256':sha(dest/'COMPLETE.json'),'job':d['slurm_job']})
 assert len(rows)==525;by={(r['index'],r['variant']):r for r in rows}
 stats={v:summary([r for r in rows if r['variant']==v]) for v in LABELS};stats['ALL']=summary(rows)
 datasets={ds:summary([r for r in rows if r['dataset']==ds]) for ds,name in DS}
 zero=[r for r in rows if r['step0_selected']];no_exceed_nonzero=[r for r in rows if not r['step0_selected'] and r['never_accuracy_exceeds_initial']]
 near=[r for r in rows if abs(r['joint_gain_nodes'])<=10]
 assert stats['ALL']['step0']['within1pp']+stats['ALL']['trained']['within1pp']==len(near)
 pm=read(PAT/'COMPLETE.json');assert sha(PAT/'RESULTS.json')==pm['files']['RESULTS.json'];pattern=read(PAT/'RESULTS.json')
 cases=[r for r in pattern['pair_rows'] if r['all_down'] and r['pR_up']]
 crossings=[]
 for r in cases:
  b=by[r['index'],'B0'];current=by[r['index'],r['variant']]
  crossings.append({'dataset':r['dataset'],'shot':r['shot'],'seed':r['seed'],'variant':r['variant'],'pseudo_delta_pp':r['all_delta_pp'],'pR_delta_pp':r['pR_delta_pp'],
   'B0_selected_step':b['selected_step'],'current_selected_step':current['selected_step'],'B0_step0':b['step0_selected'],'current_step0':current['step0_selected']})
 transitions={k:sum((r['B0_step0'],r['current_step0'])==v for r in crossings) for k,v in [('B0zero_current_trained',(True,False)),('both_zero',(True,True)),('B0trained_current_zero',(False,True)),('both_trained',(False,False))]}
 lines=['# 第0步checkpoint核查：525模型的真实history与初始化状态','',
 '生成时间：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'',
 '只读已冻结checkpoint和history，不训练、不更换checkpoint、不改变既有预测。5数据集×3shots×5seeds×7配置=525模型。逐个验证checkpoint文件哈希、config、training_seed、history最优记录与已发布selected_step一致。',
 '每个模型还按同一seed和原网络维度重建初始化，逐参数torch.equal对比选中状态；不计算训练梯度。step0模型必须全部参数精确等于原初始化，alpha/beta=1。重建初始化仅用于证据核验，不替代已冻结模型。',
 '训练先在step0验证，再每50步等既定间隔验证；本报告的“始终未超过”仅指已记录验证检查点，不声称未记录的每个梯度step都验证过。',
 '选择键是(-accuracy,NLL,step)：accuracy更高优先，同accuracy时NLL更低优先，完全相同保留更早step。因此“后续accuracy未超过0步”不充分保证返回0步；最终selected step必须直接核查。',
 '当alpha=beta=1时，pJ=(pG+pL+pR)/3=(2/3)pGL+(1/3)pR。pR仍占1/3，并非自动无影响；严格均匀pR不改变argmax，但随机pR不保证均匀。是否接近pGL以下同时核查准确率差与标签变化数。',
 '', '## 各配置的step0发生频率','',
 table(['配置','选中step0','后续accuracy未超过0步','未超过但选中后续step','step0状态等于初始化'],[[NAMES.get(v,'全部'),p(s['step0_count'],s['models']),s['never_acc_exceeds_initial'],s['nonzero_without_acc_improvement'],str(s['zero_equal_init'])+'/'+str(s['step0_count'])] for v,s in stats.items()]),
 '', '## 按数据集','',table(['数据集','step0/105模型','step0模型准确率差≤1pp','step0平均pGL→pJ变化pp','step0平均标签变化%'],[[name,p(s['step0_count'],s['models']),p(s['step0']['within1pp'],s['step0_count']),f(s['step0']['mean_gain_pp']),f(s['step0']['mean_changed_pct'])] for ds,name in DS for s in [datasets[ds]]]),
 '', '## step0是否解释pGL与pJ接近','',table(['模型组','模型数','差≤0.5pp','差≤1pp','准确率相等','平均净变化pp','平均标签变化%','最大绝对变化pp'],[[name,s['n'],p(s['within05pp'],s['n']),p(s['within1pp'],s['n']),s['same_accuracy'],f(s['mean_gain_pp']),f(s['mean_changed_pct']),f(s['max_abs_gain_pp'])] for name,s in [('step0',stats['ALL']['step0']),('非step0',stats['ALL']['trained']),('全部',stats['ALL']['all'])]]),
 '', '## 全部step0模型','',table(['数据集','shot/seed','配置','训练结束step','初始验证acc%','选中验证acc%','alpha/beta','eval1000 pR/pGL/pJ%','pJ−pGL pp','改变标签数'],[[r['dataset'],str(r['shot'])+'/'+str(r['seed']),NAMES[r['variant']],r['stop_step'],f(100*r['initial_val_accuracy']),f(100*r['selected_val_accuracy']),f(r['alpha'])+'/'+f(r['beta']),'/'.join(f(r[k]) for k in ('pR_pct','pGL_pct','pJ_pct')),f(r['joint_gain_pp']),r['changed_count']] for r in zero]),
 '', '## accuracy未超过初始但NLL平局规则选中后续step','',table(['数据集','shot/seed','配置','选中step','初始→选中验证acc%','初始→选中NLL'],[[r['dataset'],str(r['shot'])+'/'+str(r['seed']),NAMES[r['variant']],r['selected_step'],f(100*r['initial_val_accuracy'])+'→'+f(100*r['selected_val_accuracy']),f(r['initial_val_nll'])+'→'+f(r['selected_val_nll'])] for r in no_exceed_nonzero]),
 '', '## 上一问73个伪标签下降但pR上升案例的checkpoint状态','',json.dumps(transitions,ensure_ascii=False),
 '', '## PubMed 5-shot重点对照','',table(['seed','配置','选中step','结束step','alpha','beta','pR权重','pR/pGL/pJ%'],[[r['seed'],NAMES[r['variant']],r['selected_step'],r['stop_step'],f(r['alpha']),f(r['beta']),f(100*r['pR_weight'])+'%','/'.join(f(r[k]) for k in ('pR_pct','pGL_pct','pJ_pct'))] for r in rows if r['dataset']=='pubmed' and r['shot']==5 and r['variant'] in ('B0','D4_confidence__fixed','D4_confidence__retuned')]),
 '', '## 溯源','',
 '- [选择键](https://github.com/yaolalal/EnhanceRouter/blob/main/src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L18)；[保存最优状态](https://github.com/yaolalal/EnhanceRouter/blob/main/src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L53)；[先验证step0再训练](https://github.com/yaolalal/EnhanceRouter/blob/main/src/eargtc/joint_explore/joint_purf_v10_v1/training.py#L74)。',
 '- [全部525模型表](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/step0-audit-20261031/per_model.csv)；[全部step0表](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/step0-audit-20261031/step0_models.csv)；[history和文件哈希](https://github.com/yaolalal/EnhanceRouter/blob/main/reports/step0-audit-20261031/RESULTS.json)。',
 '- [只读核查代码](https://github.com/yaolalal/EnhanceRouter/blob/main/experiments/step0_audit_20261031/audit.py)。原始checkpoint和图缓存保留服务器；CSV/JSON逐模型列出路径和哈希。',
 '- 历史B0选择来源限制仍保留；本次不会排除step0、重选模型或重新计算另一套实验结果。','']
 PUB.mkdir(parents=True);csvout(PUB/'per_model.csv',rows);csvout(PUB/'step0_models.csv',zero);csvout(PUB/'counterexample_checkpoint_status.csv',crossings)
 write(PUB/'RESULTS.json',{'all_checks_pass':True,'models':525,'rows':rows,'histories':histories,'stats':stats,'by_dataset':datasets,'counterexample_transitions':transitions,'counterexample_rows':crossings,
  'proof':proof,'code_sha256':sha(Path(__file__)),'source_sha256':sha(SRC/'RESULTS.json'),'pattern_source_sha256':sha(PAT/'RESULTS.json'),'slurm_job':os.environ['SLURM_JOB_ID'],'training':False,'selection':False})
 (PUB/'REPORT.md').write_text('\n'.join(lines));write(PUB/'COMPLETE.json',{'all_checks_pass':True,'models':525,'files':{p.name:sha(p) for p in PUB.iterdir() if p.is_file()},'code_sha256':sha(Path(__file__)),'slurm_job':os.environ['SLURM_JOB_ID']})
 print(json.dumps({'all_checks_pass':True,'overall':stats['ALL'],'report':str(PUB/'REPORT.md'),'report_sha256':sha(PUB/'REPORT.md')}),flush=True)

if __name__=='__main__':
 if sys.argv[1]=='worker':worker(int(os.environ.get('SLURM_ARRAY_TASK_ID',sys.argv[2] if len(sys.argv)>2 else -1)))
 elif sys.argv[1]=='aggregate':aggregate()
 else:raise ValueError(sys.argv[1])
