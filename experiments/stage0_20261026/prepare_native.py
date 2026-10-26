from common import *
setup()
import torch,numpy as np,random,gzip,gc
from types import SimpleNamespace
assert torch.cuda.is_available()
torch.set_num_threads(1)
RUN.mkdir(parents=True,exist_ok=True)
assert not (RUN/'PREPARED.json').exists()
protocol=dict(stage=0,seeds=list(DEV),dev_seeds=list(DEV),held_out_seeds=[],shots=list(SHOTS),datasets=list(DS),cells=75,
 evaluation=dict(count=1000,key='seed/dataset',shared_across_shots=True,shared_across_seeds=False,historical42_44='reuse exact saved10shot Query1000 order',new45_46='native10shot Query; random.Random(seed).sample(sorted_query,1000); freeze once before fitting',overlap='zero with same-seed support/validation at every shot; cross-seed overlap permitted by user'),
 baseline=dict(historical42_44='reuse verified historical B0, no new calibration',new45_46='native per-shot fitting/calibration flow; preserve5shot branch-specific policy,3/10shot inherited reference fractions; validation-only decisions'),
 d1=dict(fractions=[.25,.5,.75,1.],unit='support parent with all6views',frozen='B0 full input transforms, normalization, affine, routing, rho, experts, prototypes, sources, Joint hyperparameters',parent_selection_seed='seed+913'),
 d2=dict(sizes=[50,200,1000],primary_N=200,oracle=True,exclude='own-seed fixed eval1000',scope='native source-only heads stay source-only; agreement scorer frozen'),
 d3=dict(per_class='k',pool='original validation; disclose underfilled classes; no Query replacement without registered justification'),
 statistics=dict(paired_cells=75,bootstrap_replicates=10000,bootstrap_seed=731901,L1_gate='CI lower>0;4/5 dataset means positive; no dataset<-1pp',L2_gate='mean>=0; CI not significantly negative'),
 stage1_gate='complete D1-D4, then D2b N200 L1 gate; no stage1 before stage0 completion',user_override='2026-10-26 seeds42-46 only; per-seed Query1000 shared acrossshots; authorized missing45/46 reconstruction')
write(RUN/'PROTOCOL.json',protocol)
idx=read(ROOT/'configs/inputs.json');sources={};evaluations={};audits=[];rows=[]
upstream=SOURCE.parent/'GNN-as-Judge';official={}
for name in ('train','valid','test'):
 p=upstream/f'dataset/ogbn_arxiv/split/time/{name}.csv.gz';sources[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p))
 with gzip.open(p,'rt') as f:official[name]=torch.tensor([int(x.strip()) for x in f if x.strip()])
class OfficialDataset:
 def __init__(self,**kwargs):pass
 def get_idx_split(self):return official
kernelpath=HERE/'native_split_kernel.py';kernel=compile(kernelpath.read_text(),str(kernelpath),'exec')
for ds in DS:
 g=graph(ds);n=g.num_nodes;base={k:getattr(g,k).reshape(-1).clone() for k in ('y','train_mask','val_mask','test_mask')};del g;gc.collect()
 gp=Path(idx['cells'][f'5/42/{ds}']['graph']);sources[str(gp)]=dict(md5=digest(gp,'md5'),sha256=digest(gp))
 for seed in DEV:
  generated={}
  for shot in SHOTS:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   gd=SimpleNamespace(**{k:v.to('cuda:0').clone() for k,v in base.items()},num_nodes=n)
   ns=dict(torch=torch,graph_data=gd,dataset_name=ds,shots=shot,device='cuda:0',PygNodePropPredDataset=OfficialDataset,T=SimpleNamespace(ToSparseTensor=lambda:None))
   exec(kernel,ns)
   values=[getattr(gd,k).nonzero().squeeze(-1).cpu().tolist() for k in ('train_mask','val_mask','test_mask')]
   sp=dict(dataset=ds,shots=shot,split_seed=seed,num_nodes=n,support_ids=values[0],valid_ids=values[1],unlabeled_ids=values[2],graph_path=str(gp),test_truth_exported=False)
   if seed<=44:
    p=Path(idx['cells'][f'{shot}/{seed}/{ds}']['inputs']['split']);old=read(p)
   elif shot==5:
    p=SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/splits/{ds}_5shot_seed{seed}.json';old=read(p)
   else:p=None;old=None
   if old is not None:
    assert all(sp[k]==old[k] for k in ('support_ids','valid_ids','unlabeled_ids')),(ds,seed,shot,'native replay mismatch')
    sources[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p));sp.update(reference_path=str(p),reference_sha256=digest(p),historical_split=True)
   else:sp.update(historical_split=False,origin='user-authorized native45/46 extension',kernel_sha256=digest(kernelpath))
   generated[shot]=sp;del gd,ns;torch.cuda.empty_cache()
  assert set(generated[3]['support_ids'])<=set(generated[5]['support_ids'])<=set(generated[10]['support_ids'])
  assert generated[3]['valid_ids']==generated[5]['valid_ids']==generated[10]['valid_ids']
  if seed<=44:
   ev=idx['standard_eval_ids'][f'{seed}/{ds}'];assert ev==read(idx['cells'][f'10/{seed}/{ds}']['inputs']['split'])['evaluation_1000_ids']
  else:ev=random.Random(seed).sample(generated[10]['unlabeled_ids'],1000)
  evaluations[f'{seed}/{ds}']=ev
  for shot,sp in generated.items():
   assert len(ev)==len(set(ev))==1000 and set(ev)<=set(sp['unlabeled_ids'])
   assert not set(ev)&(set(sp['support_ids'])|set(sp['valid_ids']))
   sp.update(evaluation_1000_ids=ev,evaluation_test_ids=ev,standard_eval_ids=ev,evaluation_full_ids=sp['unlabeled_ids'],evaluation_1000_provenance='exact saved historical10shot list' if seed<=44 else 'native10shot Query, historical random.Random(seed) algorithm')
   write(RUN/f'splits/{shot}/{seed}/{ds}.json',sp)
   rows.append(dict(dataset=ds,seed=seed,shot=shot,train_count=len(sp['support_ids']),valid_count=len(sp['valid_ids']),eval_train_overlap=0,eval_valid_overlap=0,eval_count=1000))
  audits.append(dict(dataset=ds,seed=seed,nested=True,same_validation=True,historical5shot_replay_exact=True,historical3_10_replay_exact=seed<=44))
  print('SPLIT_PASS',ds,seed,'counts',[len(generated[k]['support_ids']) for k in SHOTS],flush=True)
write(RUN/'EVALUATION_IDS.json',evaluations)
write(RUN/'SPLIT_AUDIT.json',dict(all_checks_pass=True,cells=75,rows=rows,nesting=audits,historical5shot_all5seeds_exact=True,historical42_44_allshots_exact=True,per_seed_eval_order_shared_across_shots=True,all_same_seed_overlaps_zero=True,source_files=sources,slurm_job=os.environ['SLURM_JOB_ID']))
old=read(ROOT/'runs/stage0-20261023-v1/SOURCE_FILES.json');old.update(sources)
write(RUN/'SOURCE_FILES.json',old);write(RUN/'environment.json',metadata())
write(RUN/'PREPARED.json',dict(cells=75,seeds=list(DEV),held_out_seeds=[],split_audit_sha256=digest(RUN/'SPLIT_AUDIT.json'),eval_ids_sha256=digest(RUN/'EVALUATION_IDS.json'),protocol_sha256=digest(RUN/'PROTOCOL.json'),all_checks_pass=True))
print('PREPARED75',flush=True)
