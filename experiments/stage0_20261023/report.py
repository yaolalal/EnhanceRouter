from common import *

def auc(y,s):
 import numpy as np
 from sklearn.metrics import roc_auc_score
 y=np.asarray(y);s=np.asarray(s)
 return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

def level1(path,shot,seed,ds,truth,budget):
 import numpy as np,torch
 out={}
 up=load(folder(shot,seed,ds)/'standard/upstream.pt');test=read(RUN/'EVALUATION_IDS.json')[ds];testset=set(test)
 gp=up['gnn_logits'].argmax(1);lp=up['llm_logits'].argmax(1)
 for branch in ['agreement','disagreement']:
  data=load(path/f'{branch}-scores.pt');mask=torch.tensor([int(i) in testset for i in data['ids']]);ids=data['ids'][mask].long()
  score=data['score'][mask].numpy();route=data['route'][mask].numpy();order=np.lexsort((ids.numpy(),-score));k=budget[branch]
  selected=np.where(route==0,gp[ids].numpy(),lp[ids].numpy()) if branch=='disagreement' else gp[ids].numpy()
  correct=selected==truth[ids].numpy();precision=float(correct[order[:k]].mean()) if k else None
  out['D-Prec@B0' if branch=='disagreement' else 'A-Prec@B0']=precision
  out[branch+'_count']=len(ids);out[branch+'_accepted']=k
  if branch=='disagreement':
   t=(gp[ids]==truth[ids])|(lp[ids]==truth[ids]);p=(gp[ids]==truth[ids]);ts=data['trust'][mask].numpy();ps=data['preference'][mask].numpy()
   out['trust_auroc']=auc(t.numpy(),ts);out['preference_auroc']=auc(p[t].numpy(),ps[t.numpy()])
   for rho in [.01,.05,.2,1.]:
    n=min(len(ids),int(np.ceil(rho*len(ids))));out[f'D-Prec@{rho}']=float(correct[order[:n]].mean()) if n else None
 return out

def accuracy(path,ids,truth):
 prob=load(path/'predictions.pt')['probability']
 from enhancerouter.cli import metrics
 return metrics(prob[ids],truth[ids])

def boot(values,seed=731901):
 import numpy as np
 a=np.asarray(values,float);a=a[np.isfinite(a)]
 if not len(a):return dict(n=0,mean=None,lower=None,upper=None)
 rng=np.random.default_rng(seed);means=np.empty(10000)
 for start in range(0,10000,1000):means[start:start+1000]=a[rng.integers(0,len(a),size=(1000,len(a)))].mean(1)
 lo,hi=np.quantile(means,[.025,.975]);return dict(n=len(a),mean=float(a.mean()),lower=float(lo),upper=float(hi))

def compare(rows,reference,metric):
 diffs=[];datasets={}
 for r in rows:
  other=reference[r['key']]
  if r.get(metric) is None or other.get(metric) is None:continue
  delta=100*(r[metric]-other[metric]);diffs.append(delta);datasets.setdefault(r['dataset'],[]).append(delta)
 result=boot(diffs);result['per_dataset']={d:boot(v) for d,v in datasets.items()}
 result['L1_gate']=bool(result['n']==75 and result['lower']>0 and sum(x['mean']>0 for x in result['per_dataset'].values())>=4 and all(x['mean']>=-1 for x in result['per_dataset'].values()))
 return result

def main():
 setup();import numpy as np,torch,csv,statistics
 out=RUN/'report';out.mkdir()
 for shot,seed,ds in CELLS:
  f=folder(shot,seed,ds)
  for name in ['b0','d3',*[f'd1/{x}' for x in [.25,.5,.75,1.]],*[f'd2{arm}/{n}' for arm in ['a','b'] for n in [50,200,1000]],'d4/confidence_router','d4/confidence_preference']:
   assert (f/name/'COMPLETE.json').exists(),(shot,seed,ds,name)
 write(RUN/'ALL_STAGE0_MODELS_FROZEN.json',dict(cells=75,models_frozen=True,held_out_evaluations=0,created_at=time.time()))
 allrows=[];baseline={};labelcache={}
 variants=['b0','d3',*[f'd1/{x}' for x in [.25,.5,.75,1.]],*[f'd2{arm}/{n}' for arm in ['a','b'] for n in [50,200,1000]],'d4/confidence_router','d4/confidence_preference']
 for shot,seed,ds in CELLS:
  if ds not in labelcache:labelcache[ds]=graph(ds).y.reshape(-1).long()
  truth=labelcache[ds];base=folder(shot,seed,ds)/'b0';cfg=read(base/'config.json');ids=torch.tensor(read(RUN/'EVALUATION_IDS.json')[ds])
  up=load(folder(shot,seed,ds)/'standard/upstream.pt');same=up['gnn_logits'][ids].argmax(1)==up['llm_logits'][ids].argmax(1)
  budget={}
  for branch in ['agreement','disagreement']:
   deployed=load(base/f'{branch}-scores.pt');on_test=torch.isin(deployed['ids'],ids)
   budget[branch]=int(deployed['accepted'][on_test].sum())
  for variant in variants:
   path=folder(shot,seed,ds)/variant;r=dict(key=key(shot,seed,ds),dataset=ds,shot=shot,seed=seed,variant=variant)
   if variant!='d3':r.update(level1(path,shot,seed,ds,truth,budget))
   if variant in ('b0','d3','d4/confidence_router','d4/confidence_preference'):r.update(accuracy(path,ids,truth))
   if variant.startswith('d2'):r.update(read(path/'diagnostics.json'))
   if variant in ('b0','d3'):
    mode='small_stratified' if variant=='d3' else 'standard'
    safe=load(folder(shot,seed,ds)/mode/'supervision.pt');r['validation_count']=len(safe['valid_ids']);r['support_count']=len(safe['support_ids'])
    if variant=='d3':r['validation_actual_per_class']=read(folder(shot,seed,ds)/mode/'HEAD_COMPLETE.json')['actual_per_class']
   allrows.append(r)
   if variant=='b0':baseline[r['key']]=r
 comparisons={}
 for variant in variants[1:]:
  group=[r for r in allrows if r['variant']==variant];metric='accuracy' if variant=='d3' else 'D-Prec@B0'
  comparisons[variant]=compare(group,baseline,metric)
  if variant.startswith('d4'):comparisons[variant]['L2']=compare(group,baseline,'accuracy')
 summary=[]
 metrics=['D-Prec@B0','A-Prec@B0','trust_auroc','preference_auroc','D-Prec@0.01','D-Prec@0.05','D-Prec@0.2','D-Prec@1.0','accuracy','macro_f1']
 for ds in DS:
  for variant in variants:
   group=[r for r in allrows if r['dataset']==ds and r['variant']==variant]
   r=dict(dataset=ds,variant=variant,cells=len(group))
   for m in metrics:
    values=[x[m] for x in group if x.get(m) is not None]
    r[m+'_mean']=float(np.mean(values)) if values else None;r[m+'_defined']=len(values)
   summary.append(r)
 historical=[]
 oldruns={3:ROOT/'runs/stage0-b0-historical-3shot-20261023',5:ROOT/'runs/acceptance-5shot-20261022-v2',10:ROOT/'runs/stage0-b0-historical-10shot-20261023'}
 for shot,oldrun in oldruns.items():
  check=read(oldrun/'evaluation/ACCEPTANCE.json');assert check['all_checks_pass'] and check['cells']==15
  historical.extend(check['per_seed'])
 write(out/'RESULTS.json',dict(protocol=read(RUN/'PROTOCOL.json'),protocol_amendments=[read(RUN/f'PROTOCOL_AMENDMENT_{j:02d}.json') for j in [1,2]],historical_reproduction=historical,dev_per_cell=allrows,summary=summary,paired_comparisons=comparisons,
  historical_45_exact=True,dev_cells=75,held_out_evaluations=0,stage1_gate=comparisons['d2b/200'],source=metadata()))
 for name,data in [('dev_per_cell.csv',allrows),('summary.csv',summary),('historical_45.csv',historical)]:
  # Nested diagnostic metadata is preserved in JSON, not stringified into numeric CSV columns.
  fields=sorted({k for r in data for k,v in r.items() if not isinstance(v,(dict,list))})
  with (out/name).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows({k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in data)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for ds in DS:
  get=lambda v:next(x for x in summary if x['dataset']==ds and x['variant']==v)
  axes[0].plot([25,50,75,100],[100*get(f'd1/{v}')['D-Prec@B0_mean'] if get(f'd1/{v}')['D-Prec@B0_mean'] is not None else float('nan') for v in [.25,.5,.75,1.]],marker='o',label=ds)
  for arm,style in [('a','--'),('b','-')]:axes[1].plot([50,200,1000],[100*get(f'd2{arm}/{n}')['D-Prec@B0_mean'] if get(f'd2{arm}/{n}')['D-Prec@B0_mean'] is not None else float('nan') for n in [50,200,1000]],style,marker='o',label=f'{ds} D2{arm}')
 axes[0].set(xlabel='Target support parents retained (%)',ylabel='D-Prec@B0 (%)',title='D1: router supervision amount');axes[0].legend(fontsize=7)
 axes[1].set(xlabel='Requested oracle parents (actual capped)',ylabel='D-Prec@B0 (%)',title='D2: oracle augmentation');axes[1].legend(fontsize=5,ncol=2)
 x=np.arange(len(DS));full=[100*next(r for r in summary if r['dataset']==d and r['variant']=='b0')['accuracy_mean'] for d in DS];small=[100*next(r for r in summary if r['dataset']==d and r['variant']=='d3')['accuracy_mean'] for d in DS]
 axes[2].bar(x-.18,full,.36,label='standard validation');axes[2].bar(x+.18,small,.36,label='k labels/class');axes[2].set_xticks(x,DS,rotation=25);axes[2].set(ylabel='Accuracy (%)',title='D3: validation label dependence');axes[2].legend(fontsize=7)
 fig.tight_layout();fig.savefig(out/'stage0_diagnostics.png',dpi=180);plt.close(fig)
 changed=[]
 checked=all_sources() | read(HERE/'source-code.json')
 for receipt in (RUN/'shared').glob('text*/*.json'):
  checked.update(read(receipt).get('source_extra',{}))
 for p,entry in checked.items():
  if digest(p,'md5')!=entry['md5']:changed.append(p)
 assert not changed,changed
 write(out/'SOURCE_MD5_RECHECK.json',dict(checked_files=len(checked),unchanged=True,files={p:entry['md5'] for p,entry in checked.items()}))
 text=['EnhanceRouter Stage 0 report','', 'Historical B0: 45/45 exact reproduction PASS.',
  'New development protocol: 75 cells, dev seeds142-146; held-out1042-1046 reserved and not evaluated.',
  'Evaluation: same1000 node IDs per dataset for all shots and all seeds, disjoint from every new support/validation.',
  'B0 new-dev recipe/calibration and candidate catalog are documented in PROTOCOL.json; historical score replay remains separate.','']
 for v in variants[1:]:
  r=comparisons[v]
  if r['mean'] is None:
   text.append(f'{v}: no defined paired primary metrics; gate not passed');continue
  text.append(f"{v}: paired mean delta={r['mean']:.4f} pp; 95%CI=[{r['lower']:.4f},{r['upper']:.4f}], n={r['n']}; L1_gate={r['L1_gate']}")
  text.append('  Dataset deltas: '+', '.join(f"{d} {z['mean']:+.4f}pp" for d,z in r['per_dataset'].items()))
 gate=comparisons['d2b/200']['L1_gate'];text+=['',f'Stage1 decision: {"proceed" if gate else "skip under preregistered gate"}; primary diagnostic is D2b requested N=200.',
  'Oracle outcomes are analyses, not an admissible deployed method. Requested N and actual N are reported separately.',
  'D1 freezes experts/prototypes; it measures router training-data amount, not total expert label budget.',
  'All frozen-source MD5 checks passed. All new writes were guarded to EnhanceRouter; every registered frozen input/code MD5 is unchanged.']
 (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
 seal(out,dict(all_checks_pass=True,historical_cells=45,dev_cells=75,held_out_evaluations=0,source_md5_unchanged=True))
 print('\n'.join(text),flush=True)

if __name__=='__main__':main()
