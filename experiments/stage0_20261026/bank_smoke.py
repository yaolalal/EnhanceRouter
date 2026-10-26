from common import *
setup()
import torch,frontend
from eargtc.models.gnn import MLPHead
original=RUN;testroot=RUN/f'sanity-20261026-{os.environ["SLURM_JOB_ID"]}';frontend.RUN=testroot
index=read(ROOT/'configs/inputs.json');checks=[]
for ds in DS:
 safe=load(index['cells'][f'5/42/{ds}']['inputs']['supervision']);mask=~torch.isin(safe['support_ids'],torch.tensor(read(RUN/'EVALUATION_IDS.json')[ds]));ids=safe['support_ids'][mask];targets=safe['support_targets'][mask]
 out={}
 for name in ENVIRONMENTS[1:4]:
  known={}
  for family in ['router_v10_multiseed_data_20260912_v1','router_v10_shots_v1_20260914']:
   p=SOURCE/f'runs/{family}/shared/text/{ds}/{name}.pt'
   if not p.exists():continue
   old=load(p)
   for j,n in enumerate(old['node_ids']):
    if int(n) in set(ids.tolist()):known[int(n)]=(old['l'][j],old['llm_prob'][j],old['changed_mask'][j])
  keep=torch.tensor([int(i) in known for i in ids]);assert keep.all(),(ds,name,'historical support missing')
  out[name]=dict(l=torch.stack([known[int(i)][0] for i in ids]),llm_prob=torch.stack([known[int(i)][1] for i in ids]),changed_mask=torch.stack([known[int(i)][2] for i in ids]))
 save(testroot/f'shared/text/{ds}.pt',dict(node_ids=ids,environments=out))
 sp=split(5,142,ds);up=load(folder(5,142,ds)/'standard/upstream.pt');state=load(folder(5,142,ds)/'standard/head.pt');cl=canonical(ds);g=graph(ds)
 head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval()
 assembled,_=frontend.view_rows(ds,ids,targets,sp,up,head,cl,g.edge_index)
 counts={b:len(assembled[b]['ids']) for b in ['agreement','disagreement']};assert sum(counts.values())==6*len(ids)
 checks.append(dict(dataset=ds,parents=len(ids),six_view_rows=counts,fixed_eval_overlap=0))
 print('BANK_SMOKE_PASS',checks[-1],flush=True)
write(RUN/f'BANK_SMOKE-20261026-{os.environ["SLURM_JOB_ID"]}.json',dict(checks=checks,all_checks_pass=True,scope='six-view construction API checks on historical support nodes outside the fixed evaluation set; not development performance',metadata=metadata()))
