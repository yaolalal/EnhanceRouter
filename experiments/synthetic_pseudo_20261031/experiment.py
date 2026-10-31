"""Truth-built synthetic disagreement labels: preregister, replay gates, train."""
from pathlib import Path
import os,sys,json,csv,hashlib,itertools,copy,gc,shutil,datetime,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
import training as kernel
RUN=ROOT/'runs/synthetic_pseudo_20261031-v1'
PUB=ROOT/'reports/synthetic_pseudo_20261031'
SOURCE=ROOT/'reports/comprehensive-b0-3x2-20261029'
DS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SEEDS=(42,43,44,45,46)
CELLS=list(itertools.product(SEEDS,DS))
RULES=tuple('R'+str(i) for i in range(1,11))
PLAN=[dict(mode='best',rule=r,noise_seed=no,training_seed=tr) for r in RULES for no in (0,1,2) for tr in (0,1)]
PLAN += [dict(mode='best',rule=r,noise_seed=None,training_seed=tr) for r in ('A0','A100') for tr in (0,1)]
PLAN += [dict(mode='last500',rule='R1',noise_seed=no,training_seed=tr) for no in (0,1,2) for tr in (0,1)]
PLAN += [dict(mode='last500',rule=r,noise_seed=None,training_seed=tr) for r in ('A0','A100') for tr in (0,1)]

def require():assert os.environ.get('SLURM_JOB_ID'),'Numerical work requires Slurm'
def read(p):return json.loads(Path(p).read_text())
def sha(p,kind='sha256'):
 h=hashlib.new(kind)
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);assert ROOT.resolve() in p.resolve().parents;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvout(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(rows[0])
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save(p,x):
 import torch
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:torch.save(x,f)
def load(p):
 require();import torch
 return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def seal(p,extra=None):
 write(p/'COMPLETE.json',dict(extra or {},files={str(q.relative_to(p)):sha(q) for q in sorted(p.rglob('*')) if q.is_file()}))
def check(p):
 s=read(p/'COMPLETE.json')
 for n,h in s['files'].items():assert sha(p/n)==h,(str(p),n)
 return s
def files():
 paths=[HERE/n for n in ('AGENTS.md','TRAINING_PROVENANCE.json','training.py','experiment.py','report.py','job.sh')]
 paths += [ROOT/'src/eargtc/joint_explore/joint_purf_v10_v1'/n for n in ('model.py','training.py')]
 paths += [ROOT/'src/enhancerouter/adapter.py',ROOT/'src/eargtc/joint_five_router/adapter.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def registered():
 require();r=read(RUN/'REGISTRATION.json');assert r['files']==files(),'Source changed during experiment'
 assert Path(kernel.__file__).resolve()==HERE/'training.py'
 assert Path(sys.modules[kernel.UnifiedRepair.__module__].__file__).resolve()==ROOT/'src/eargtc/joint_explore/joint_purf_v10_v1/model.py'
 assert sha(SOURCE/'PUBLICATION_SHA256.json')==r['source_manifest_sha256']
 return r
def old_preflight():
 sys.path.insert(0,str(ROOT/'experiments/stage0_20261027'))
 from report_full_query import preflight
 preflight()
def register():
 require();assert not RUN.exists();old_preflight()
 manifest=read(SOURCE/'PUBLICATION_SHA256.json');assert sha(SOURCE/'RESULTS.json')==manifest['RESULTS.json']
 b=[r for r in read(SOURCE/'RESULTS.json')['rows'] if r['shot']==5 and r['variant']=='B0'];assert len(b)==25
 from enhancerouter.io import environment
 write(RUN/'REGISTRATION.json',dict(diagnostic_uses_truth=True,not_for_method_selection=True,eval1000_excluded_from_synthetic=True,
  shot=5,split_seeds=list(SEEDS),datasets=list(DS),point_seeds=[0],noise_seeds=[0,1,2],training_seed_ids=[0,1],
  rng_mapping='effective=split_seed+1000003*training_seed_id; id0 exactly legacy; original +101/+1101/+2101/+3101 batch offsets',
  rules=list(RULES),anchors=['A0','A100'],runs_per_cell=len(PLAN),runs=25*len(PLAN),plan=PLAN,
  config_policy='B0 frozen, keep=0; non-A0 B0 cross0 -> cross1; A0 cross0; last500 forces500 and no early stopping',
  pool_semantics='synthetic chosen-GNN => teach_g; chosen-LLM => teach_l; B0 replay retains original reversed legacy names',
  head_tail='full graph true-class counts; head>median, tail<=median; offline metric only',
  variance='hierarchical law of total variance split -> noise -> train; also crossed finite-ANOVA interactions; point seed0 only, no point-variance estimate',
  rounding='K=round(.8*N); ties impossible for integer .8*N; actual K/N recorded',
  confidence='max class probability from frozen expert log_softmax.exp(); margin uses B0 alpha/beta float64 mixture',
  bootstrap=dict(clusters='split seed',replicates=10000,rng_seed=20261031),
  source_manifest_sha256=sha(SOURCE/'PUBLICATION_SHA256.json'),source_results_sha256=sha(SOURCE/'RESULTS.json'),files=files(),B0_rows=b))
 write(RUN/'environment.json',environment());print('SYNTHETIC_REGISTERED',25*len(PLAN),flush=True)

class Sources:
 def __init__(self):self.checked={};self.cached={}
 def tensor(self,record,cache=False):
  p=Path(record['path']);kind=record.get('hash_algorithm','sha256');expected=record.get('hash',record.get('sha256'))
  k=(str(p.resolve()),kind,expected)
  if k not in self.checked:assert sha(p,kind)==expected,(str(p),'source hash');self.checked[k]=True
  if cache and k in self.cached:return self.cached[k]
  obj=load(p)
  if cache:self.cached[k]=obj
  return obj
def baseline_row(seed,ds):return next(r for r in read(RUN/'REGISTRATION.json')['B0_rows'] if (r['seed'],r['dataset'])==(seed,ds))
def raw_inputs(row,sources):
 import torch
 b=row['source']['B0_inputs'];up=sources.tensor(b['upstream']);e=sources.tensor(b['embeddings'],True);h=sources.tensor(b['hidden'],True)
 safe=sources.tensor(b['supervision']);return up,e,h['hidden'],safe
def cellpath(seed,ds):return RUN/'prepared'/ds/str(seed)
def run_id(plan):return plan['mode']+'__'+plan['rule']+'__n'+str(plan['noise_seed'] if plan['noise_seed'] is not None else 'none')+'__t'+str(plan['training_seed'])

def allocate(bank,rule,noise):
 import numpy as np
 n=len(bank['ids']);k=round(.8*n);rng=np.random.default_rng(0 if noise is None else noise)
 correct=np.ones(n,dtype=bool);details={}
 if rule=='A0':return np.zeros(n,dtype=bool),details
 if rule=='A100':return correct,details
 correct[:]=False
 if rule=='R1':correct[rng.permutation(n)[:k]]=True
 elif rule=='R6':
  correct[:]=True;remaining=n-k;full=[];partial=None;partial_ids=[];order=rng.permutation(bank['classes']).tolist()
  for c in order:
   if remaining==0:break
   at=np.flatnonzero(bank['truth']==c)
   if not len(at):continue
   if len(at)<=remaining:correct[at]=False;remaining-=len(at);full.append(int(c))
   else:
    selected=rng.permutation(at)[:remaining];correct[selected]=False;partial=int(c);partial_ids=bank['ids'][selected].tolist();remaining=0
  assert remaining==0
  details=dict(class_order=order,fully_wrong_classes=full,partially_wrong_class=partial,partially_wrong_ids=partial_ids,actual_wrong=int((~correct).sum()))
 else:
  mapping={'R2':('degree',True),'R3':('degree',False),'R4':('margin',True),'R5':('margin',False),
           'R7':('pL_confidence',True),'R8':('pL_confidence',False),'R9':('pG_confidence',True),'R10':('pG_confidence',False)}
  field,descending=mapping[rule];v=bank[field];tie=np.empty(n,dtype=np.int64);tie[rng.permutation(n)]=np.arange(n)
  rank=np.lexsort((tie,-v if descending else v));correct[rank[:k]]=True
 assert int(correct.sum())==k
 return correct,details

def pool_artifact(bank,rule,noise,dest,protected):
 import numpy as np,torch
 ids=bank['ids'] if rule!='A0' else np.empty(0,dtype=np.int64)
 mask,details=allocate(bank,rule,noise)
 if rule=='A0':mask=mask[:0]
 true_g=bank['gnn_pred']==bank['truth'];chosen_g=np.where(mask,true_g,~true_g) if rule!='A0' else np.empty(0,dtype=bool)
 labels=np.where(chosen_g,bank['gnn_pred'],bank['llm_pred']) if rule!='A0' else np.empty(0,dtype=np.int64)
 truth=bank['truth'] if rule!='A0' else np.empty(0,dtype=np.int64)
 assert not set(ids)&set(protected);assert np.all((labels==truth)==mask)
 assert len(ids)==len(set(ids));N=len(ids);correct=int(mask.sum())
 assert correct==(N if rule=='A100' else 0 if rule=='A0' else round(.8*N))
 records=[]
 for i,node in enumerate(ids):records.append(dict(node_id=int(node),truth=int(truth[i]),gnn_pred=int(bank['gnn_pred'][i]),llm_pred=int(bank['llm_pred'][i]),
  pseudo_label=int(labels[i]),chosen_expert='GNN' if chosen_g[i] else 'LLM',pool='teach_g' if chosen_g[i] else 'teach_l',correct=bool(mask[i]),
  degree=int(bank['degree'][i]),margin=float(bank['margin'][i]),pG_confidence=float(bank['pG_confidence'][i]),pL_confidence=float(bank['pL_confidence'][i])))
 pools={'teach_g':torch.as_tensor(ids[chosen_g].copy()),'teach_l':torch.as_tensor(ids[~chosen_g].copy()),'agreement':torch.empty(0,dtype=torch.long)}
 assert set(pools['teach_g'].tolist()).isdisjoint(pools['teach_l'].tolist())
 def avg(v,m):return float(np.mean(v[m])) if np.any(m) else None
 def hist(v):return np.bincount(v,minlength=bank['classes']).tolist()
 summary=dict(rule=rule,noise_seed=noise,N=N,K=correct,actual_precision=correct/N if N else None,
  teach_g_count=int(chosen_g.sum()),teach_l_count=int((~chosen_g).sum()),teach_g_precision=avg(mask,chosen_g),teach_l_precision=avg(mask,~chosen_g),
  true_class_histogram=hist(truth),pseudo_class_histogram=hist(labels),
  correct_mean_degree=avg(bank['degree'],mask) if N else None,wrong_mean_degree=avg(bank['degree'],~mask) if N else None,
  correct_mean_margin=avg(bank['margin'],mask) if N else None,wrong_mean_margin=avg(bank['margin'],~mask) if N else None,R6=details)
 active=[summary[k] for k in ('teach_g_precision','teach_l_precision') if summary[k] is not None]
 summary.update(active_pool_count=len(active),pseudo_loss_mass_multiplier=.5*len(active),equal_pool_precision=float(np.mean(active)) if active else None)
 dest.mkdir(parents=True);csvout(dest/'pool.csv',records,['node_id','truth','gnn_pred','llm_pred','pseudo_label','chosen_expert','pool','correct','degree','margin','pG_confidence','pL_confidence'])
 save(dest/'pool.pt',dict(ids=torch.as_tensor(ids.copy()),labels=torch.as_tensor(labels.copy()),pools=pools));write(dest/'POOL.json',summary);seal(dest,dict(all_checks_pass=True))
 return summary

def preflight(index):
 require();registered();old_preflight();import torch,numpy as np
 from enhancerouter.adapter import prepare_arrays
 from eargtc.data import _install_pyg_optional_extension_stubs
 import training as kernel
 torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
 ds=DS[index];out=RUN/'preflight'/ds;assert not out.exists();out.mkdir(parents=True)
 source=Sources();reg=read(RUN/'REGISTRATION.json');first=baseline_row(42,ds)
 oldcell=SOURCE/'cells'/f'{first["index"]:02d}'/'RESULTS.json';manifest=read(SOURCE/'PUBLICATION_SHA256.json')
 assert sha(oldcell)==manifest[str(oldcell.relative_to(SOURCE))]
 gr=read(oldcell)['inputs']['graph'];assert sha(gr['path'],gr['hash_algorithm'])==gr['hash'];_install_pyg_optional_extension_stubs()
 graph=torch.load(gr['path'],map_location='cpu',weights_only=False,mmap=True);y=graph.y.reshape(-1).long();n=len(y)
 edge=graph.edge_index.long();assert edge.shape[0]==2 and int(edge.min())>=0 and int(edge.max())<n
 a,b=edge[0],edge[1];u=torch.minimum(a,b);v=torch.maximum(a,b);nonself=u!=v;keys=torch.unique(u[nonself]*n+v[nonself]);u=keys//n;v=keys%n
 degree=torch.bincount(torch.cat((u,v)),minlength=n).numpy();classes=int(y.max())+1;freq=torch.bincount(y,minlength=classes).numpy();median=float(np.median(freq));head=np.flatnonzero(freq>median);tail=np.flatnonzero(freq<=median)
 del graph,edge,a,b,u,v,keys;gc.collect();receipts=[]
 for seed in SEEDS:
  row=baseline_row(seed,ds);dest=cellpath(seed,ds);assert not dest.exists();dest.mkdir(parents=True)
  up,e,h,safe=raw_inputs(row,source);sp=read(ROOT/f'runs/stage0-20261026-v2/splits/5/{seed}/{ds}.json')
  assert sp['num_nodes']==n and safe['support_ids'].tolist()==sp['support_ids'] and safe['valid_ids'].tolist()==sp['valid_ids']
  bpath=Path(row['source']['model_path']);assert sha(bpath/'COMPLETE.json')==row['source']['model_seal_sha256'];bs=read(bpath/'COMPLETE.json')
  assert sha(bpath/'router.pt')==bs['files']['router.pt'];router=load(bpath/'router.pt')
  f,s,p=prepare_arrays(sp,up,e,safe,router,ds)
  reference=source.tensor(row['source']['B0_inputs']['checkpoint']);assert reference['config']==row['config']
  assert torch.equal(s['selection_ids'],reference['validation_ids'])
  member=kernel.train(f,s,p,h,row['config'],seed,0,'best')
  assert member['checkpoints']['accuracy']==reference['checkpoints']['accuracy'],'B0 checkpoint selection changed'
  model=kernel.make_model(f,h,row['config']);model.load_state_dict(member['states']['accuracy'])
  ev=torch.as_tensor(sp['evaluation_1000_ids'],dtype=torch.long)
  expected_ev=read(ROOT/'runs/stage0-20261026-v2/EVALUATION_IDS.json')[f'{seed}/{ds}'];assert ev.tolist()==expected_ev
  pj,pr=kernel.predict(model,f,h,ev,repair=True);old=source.tensor(row['source']['B0_inputs']['predictions'])['probability'][ev]
  assert torch.equal(pj.argmax(1),old.argmax(1)),'B0 replay eval1000 labels must match exactly'
  a,b=(float(torch.nn.functional.softplus(reference['states']['accuracy'][k])) for k in ('raw_alpha','raw_beta'))
  assert (a,b)==(row['alpha'],row['beta'])
  g=up['gnn_logits'].argmax(1);l=up['llm_logits'].argmax(1);query=torch.as_tensor(sp['unlabeled_ids'],dtype=torch.long)
  protected=torch.unique(torch.cat((ev,safe['support_ids'],safe['valid_ids'])))
  eligible=(g[query]!=l[query])&((g[query]==y[query])^(l[query]==y[query]))&~torch.isin(query,protected)
  candidates=query[eligible].sort().values;N=min(1000,len(candidates));selected=np.sort(np.random.default_rng(0).choice(candidates.numpy(),size=N,replace=False))
  ids=torch.as_tensor(selected);pg=f['logp_g'][ids].exp().double();pl=f['logp_l'][ids].exp().double();pgl=(a*pg+b*pl)/(a+b);top=pgl.topk(2,dim=1).values
  bank=dict(ids=selected,truth=y[ids].numpy(),gnn_pred=g[ids].numpy(),llm_pred=l[ids].numpy(),degree=degree[selected],margin=(top[:,0]-top[:,1]).numpy(),
   pG_confidence=pg.max(1).values.numpy(),pL_confidence=pl.max(1).values.numpy(),classes=classes)
  summaries=[]
  for rule in (*RULES,'A0','A100'):
   for noise in ((None,) if rule.startswith('A') else (0,1,2)):
    summary=pool_artifact(bank,rule,noise,dest/'pools'/(rule+'__n'+str(noise)),protected.tolist());summaries.append(summary)
  for no in (0,1,2):
   for rule in RULES:assert torch.equal(load(dest/'pools'/(rule+'__n'+str(no))/'pool.pt')['ids'],ids)
  b0eval=dict(ids=ev,truth=y[ev],head_classes=torch.as_tensor(head),tail_classes=torch.as_tensor(tail),class_frequencies=torch.as_tensor(freq),class_frequency_median=median)
  save(dest/'safe.pt',dict(supervision=s,evaluation_ids=ev));save(dest/'evaluation.pt',b0eval)
  pgl_ev=(a*f['logp_g'][ev].exp().double()+b*f['logp_l'][ev].exp().double())/(a+b)
  baseline_metrics=evaluate_predictions(pj,pr,pgl_ev,pgl_ev,y[ev],head,tail,classes)
  c=dict(dataset=ds,split_seed=seed,B0=row,sources=row['source']['B0_inputs'],graph_source=gr,
   candidate_count=len(candidates),candidate_GNN_correct=int((g[candidates]==y[candidates]).sum()),candidate_LLM_correct=int((l[candidates]==y[candidates]).sum()),
   N=N,n_capped=len(candidates)<1000,K=round(.8*N),point_seed=0,selected_ids_sha256=hashlib.sha256(np.asarray(selected,dtype='<i8').tobytes()).hexdigest(),
   B0_alpha=a,B0_beta=b,B0_cross_zero=row['config']['cross']==0,
   B0_replay=dict(all_checks_pass=True,eval1000_labels_exact=True,states_bitwise_equal=all(torch.equal(member['states']['accuracy'][k],reference['states']['accuracy'][k]) for k in reference['states']['accuracy']),
    selected=member['checkpoints']['accuracy'],training_seed_id=0,effective_rng_seed=seed),B0_metrics=baseline_metrics,
   source_code=files(),registration_sha256=sha(RUN/'REGISTRATION.json'),class_frequencies=freq.tolist(),head_classes=head.tolist(),tail_classes=tail.tolist(),class_frequency_median=median,pool_summaries=summaries)
  write(dest/'CONTEXT.json',c);save(dest/'B0_replay.pt',member)
  records=[dict(node_id=int(x),truth=int(y[x]),gnn_pred=int(g[x]),llm_pred=int(l[x]),correct_expert='GNN' if g[x]==y[x] else 'LLM') for x in candidates.tolist()]
  csvout(dest/'candidates.csv',records,['node_id','truth','gnn_pred','llm_pred','correct_expert']);seal(dest,dict(all_checks_pass=True,B0_replay_exact=True))
  receipts.append(dict(dataset=ds,split_seed=seed,candidate_count=len(candidates),candidate_GNN_correct=c['candidate_GNN_correct'],candidate_LLM_correct=c['candidate_LLM_correct'],N=N,K=c['K'],n_capped=c['n_capped'],B0_cross=row['config']['cross'],B0_labels_exact=True,seal_sha256=sha(dest/'COMPLETE.json')))
  print('SYNTHETIC_PREFLIGHT_CELL_PASS',ds,seed,len(candidates),N,flush=True)
  del up,e,h,safe,router,f,s,p,reference,member,model,pj,pr,old;gc.collect()
 write(out/'RESULTS.json',dict(all_checks_pass=True,rows=receipts,source_code=files(),slurm_job=os.environ['SLURM_JOB_ID']));seal(out,dict(all_checks_pass=True,cells=5))

def preflight_gate():
 registered();rows=[]
 for ds in DS:
  p=RUN/'preflight'/ds;check(p);rows.extend(read(p/'RESULTS.json')['rows'])
 assert len(rows)==25 and all(r['B0_labels_exact'] for r in rows)
 write(RUN/'PREFLIGHT_PASS.json',dict(all_checks_pass=True,rows=rows,files=files()));csvout(RUN/'preflight.csv',rows)
 print('SYNTHETIC_PREFLIGHT_GATE_PASS',25,flush=True)

def evaluate_predictions(pj,pr,own,fixed,y,head,tail,classes):
 import torch,numpy as np
 j=pj.argmax(1);r=pr.argmax(1);o=own.argmax(1);b=fixed.argmax(1);wrong=b!=y;correct=~wrong;subset=r!=b
 def acc(v,mask=None):
  if mask is None:mask=torch.ones(len(y),dtype=torch.bool)
  return float((v[mask]==y[mask]).double().mean()) if bool(mask.any()) else None
 hist=torch.bincount(r,minlength=classes).double();prob=hist/hist.sum();nz=prob>0;entropy=float(-(prob[nz]*prob[nz].log()).sum())
 hmask=torch.isin(y,torch.as_tensor(head));tmask=torch.isin(y,torch.as_tensor(tail))
 out=dict(full_accuracy=acc(j),pR_accuracy=acc(r),pGL_own_accuracy=acc(o),pGL_B0_accuracy=acc(b),
  pR_contribution=acc(j)-acc(b),rescue_rate=acc(r,wrong),harm_rate=1-acc(r,correct) if bool(correct.any()) else None,
  reference_wrong_count=int(wrong.sum()),reference_correct_count=int(correct.sum()),
  S_count=int(subset.sum()),S_fraction=float(subset.double().mean()),S_pR_accuracy=acc(r,subset),S_pGL_accuracy=acc(b,subset),S_full_accuracy=acc(j,subset),
  head_pR_accuracy=acc(r,hmask),tail_pR_accuracy=acc(r,tmask),head_eval_count=int(hmask.sum()),tail_eval_count=int(tmask.sum()),
  prediction_entropy=entropy,prediction_normalized_entropy=entropy/np.log(classes) if classes>1 else 0.,prediction_class_count=int(nz.sum()),prediction_max_class_fraction=float(prob.max()),prediction_histogram=hist.long().tolist())
 return out

def train_cell(index):
 require();registered();assert read(RUN/'PREFLIGHT_PASS.json')['files']==files()
 if index>=5:assert read(RUN/'PILOT_PASS.json')['files']==files()
 import torch,numpy as np,training as kernel
 torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
 seed,ds=CELLS[index];prepared=cellpath(seed,ds);check(prepared);context=read(prepared/'CONTEXT.json');safe=load(prepared/'safe.pt');s=safe['supervision'];ev=safe['evaluation_ids']
 source=Sources();up,e,h,_=raw_inputs(context['B0'],source)
 base_f={'node_ids':torch.arange(up['gnn_logits'].shape[0]),'gnn_embeddings':e['embeddings'].float(),
  'logp_g':up['gnn_logits'].float().log_softmax(1),'logp_l':up['llm_logits'].float().log_softmax(1)}
 fixed=(context['B0_alpha']*base_f['logp_g'][ev].exp().double()+context['B0_beta']*base_f['logp_l'][ev].exp().double())/(context['B0_alpha']+context['B0_beta'])
 dest=RUN/'models'/f'{index:02d}';dest.mkdir(parents=True,exist_ok=True)
 for plan in PLAN:
  trial=dest/run_id(plan)
  if (trial/'COMPLETE.json').exists():check(trial);continue
  assert not trial.exists(),'Preserve interrupted run and diagnose before retry';trial.mkdir()
  pooldir=prepared/'pools'/(plan['rule']+'__n'+str(plan['noise_seed']));check(pooldir);pool=load(pooldir/'pool.pt');meta=read(pooldir/'POOL.json')
  assert not torch.isin(pool['ids'],torch.cat((ev,s['gold_ids'],s['selection_ids'],s['threshold_ids']))).any()
  f=dict(base_f,selected_pred=torch.zeros(len(base_f['node_ids']),dtype=torch.long));f['selected_pred'][pool['ids']]=pool['labels']
  cfg=copy.deepcopy(context['B0']['config']);cfg['keep']=0.;forced=False
  if plan['rule']=='A0':cfg['cross']=0.
  elif cfg['cross']==0:cfg['cross']=1.;forced=True
  if plan['mode']=='last500':cfg['max_steps']=500
  write(trial/'CONFIG.json',dict(plan=plan,split_seed=seed,dataset=ds,point_seed=None if plan['rule']=='A0' else 0,config=cfg,B0_config=context['B0']['config'],cross_forced=forced,cell_B0_cross_zero=context['B0_cross_zero'],pool_path=str(pooldir),pool_seal_sha256=sha(pooldir/'COMPLETE.json'),prepared_seal_sha256=sha(prepared/'COMPLETE.json'),files=files(),environment_sha256=sha(RUN/'environment.json')))
  member=kernel.train(f,s,pool['pools'],h,cfg,seed,plan['training_seed'],plan['mode']);selected='accuracy' if plan['mode']=='best' else 'last'
  model=kernel.make_model(f,h,cfg);model.load_state_dict(member['states'][selected]);pj,pr=kernel.predict(model,f,h,ev,repair=True)
  a,b=(float(x.detach()) for x in model.coefficients());own=(a*f['logp_g'][ev].exp().double()+b*f['logp_l'][ev].exp().double())/(a+b)
  evaluation=load(prepared/'evaluation.pt');metrics=evaluate_predictions(pj,pr,own,fixed,evaluation['truth'],context['head_classes'],context['tail_classes'],pj.shape[1])
  chosen=member['checkpoints'][selected];row=dict(run_id=f'{index:02d}/'+run_id(plan),index=index,dataset=ds,split_seed=seed,shot=5,rule=plan['rule'],selection_mode=plan['mode'],point_seed=None if plan['rule']=='A0' else 0,
   noise_seed=plan['noise_seed'],training_seed=plan['training_seed'],effective_rng_seed=member['training_seed'],
   candidate_count=context['candidate_count'],N=meta['N'],n_capped=context['n_capped'],K=meta['K'],actual_precision=meta['actual_precision'],cross_forced=forced,cell_B0_cross_zero=context['B0_cross_zero'],
   cross=cfg['cross'],aux=cfg['aux'],keep=cfg['keep'],selected_step=chosen['step'],selected_step0=chosen['step']==0,stop_step=member['history'][-1]['step'],
   alpha=a,beta=b,**metrics,**{k:meta[k] for k in ('teach_g_count','teach_l_count','teach_g_precision','teach_l_precision','correct_mean_degree','wrong_mean_degree','correct_mean_margin','wrong_mean_margin')},
   equal_pool_precision=meta['equal_pool_precision'],pseudo_loss_mass_multiplier=meta['pseudo_loss_mass_multiplier'],pool_path=str(trial/'pool.csv'),pool_sha256=sha(pooldir/'pool.csv'),prepared_context_sha256=sha(prepared/'CONTEXT.json'),checkpoint_path=str(trial/'checkpoint.pt'),config=cfg,pool_summary=meta,slurm_job=os.environ['SLURM_JOB_ID'])
  save(trial/'checkpoint.pt',member);write(trial/'RESULTS.json',row);shutil.copy2(pooldir/'pool.csv',trial/'pool.csv')
  preds=[];j=pj.argmax(1);r=pr.argmax(1);o=own.argmax(1);b0=fixed.argmax(1);y=evaluation['truth']
  for k,node in enumerate(ev.tolist()):preds.append(dict(eval_order=k,node_id=node,truth=int(y[k]),pR=int(r[k]),pGL_own=int(o[k]),pGL_B0=int(b0[k]),pJ=int(j[k]),in_S=bool(r[k]!=b0[k])))
  csvout(trial/'eval1000.csv',preds);seal(trial,dict(all_checks_pass=True,selected_step=chosen['step'],source_files=files()))
  print('SYNTHETIC_TRAIN_PASS',index,run_id(plan),metrics['full_accuracy'],chosen['step'],flush=True)
  del model,member,pj,pr,own,f;gc.collect()
 checks=[dict(run_id=run_id(plan),seal_sha256=sha(dest/run_id(plan)/'COMPLETE.json')) for plan in PLAN]
 write(dest/'CELL_COMPLETE.json',dict(all_checks_pass=True,runs=len(PLAN),results=checks,files=files(),slurm_job=os.environ['SLURM_JOB_ID']))

def pilot_gate():
 registered();proof=[]
 for index in range(5):
  p=RUN/'models'/f'{index:02d}';c=read(p/'CELL_COMPLETE.json');assert c['runs']==len(PLAN) and c['files']==files()
  for r in c['results']:assert sha(p/r['run_id']/'COMPLETE.json')==r['seal_sha256'];check(p/r['run_id'])
  proof.append(dict(index=index,sha256=sha(p/'CELL_COMPLETE.json')))
 write(RUN/'PILOT_PASS.json',dict(all_checks_pass=True,proof=proof,files=files()));print('SYNTHETIC_PILOT_GATE_PASS',flush=True)

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='register':register()
 elif mode=='preflight':preflight(int(os.environ.get('SLURM_ARRAY_TASK_ID',sys.argv[2] if len(sys.argv)>2 else -1)))
 elif mode=='preflight-gate':preflight_gate()
 elif mode=='cell':train_cell(int(os.environ.get('SLURM_ARRAY_TASK_ID',sys.argv[2] if len(sys.argv)>2 else -1)))
 elif mode=='pilot-gate':pilot_gate()
 elif mode=='report':
  import report;report.aggregate()
 else:raise ValueError(mode)
