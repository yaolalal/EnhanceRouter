from common import *
import argparse

def main(index):
 setup();import torch
 from tune_gcn_ssl_1000_eval import train_head,HeadTrial
 from eargtc.utils import set_seed
 shot,seed,ds=CELLS[index];sp=split(shot,seed,ds);out=folder(shot,seed,ds)/'small_stratified'
 if (out/'HEAD_COMPLETE.json').exists():return
 data=graph(ds);truth=data.y.reshape(-1).long();cl=canonical(ds);classes=cl['llm_logits'].shape[1];reserved=set(read(RUN/'EVALUATION_IDS.json')[ds])
 gen=torch.Generator().manual_seed(seed+1181);chosen=[];counts=[]
 for label in range(classes):
  nodes=torch.where((truth==label)&~data.train_mask)[0];nodes=torch.tensor([int(i) for i in nodes if int(i) not in reserved],dtype=torch.long)
  selected=nodes[torch.randperm(len(nodes),generator=gen)[:shot]];chosen+=selected.tolist();counts.append(len(selected))
 support=torch.tensor(sp['support_ids'],dtype=torch.long);valid=torch.tensor(sorted(chosen),dtype=torch.long)
 assert not(set(valid.tolist())&(set(support.tolist())|reserved))
 target=torch.full_like(truth,-1);target[support]=truth[support];target[valid]=truth[valid]
 trial=HeadTrial(**read(SOURCE/f'runs/gcn_ssl_v2_selected_20260623_v1/{ds}/mlp_metrics.json')['trial'])
 set_seed(seed)
 logits,stats,artifact=train_head(cl['g'],target,support,valid,classes,trial)
 save(out/'upstream.pt',dict(gnn_logits=logits.cpu().float(),llm_logits=cl['llm_logits']))
 save(out/'head.pt',{k:v.cpu() for k,v in artifact['state_dict'].items()})
 save(out/'supervision.pt',dict(support_ids=support,support_targets=truth[support],valid_ids=valid,valid_targets=truth[valid]))
 write(out/'head-training.json',dict(trial=trial.__dict__,stats=stats,history=artifact['history'],training_seed=seed,valid_ids=valid.tolist(),actual_per_class=counts,requested_per_class=shot,validation_sampling_seed=seed+1181,query_truth_used=False,metadata=metadata()))
 write(out/'HEAD_COMPLETE.json',dict(all_checks_pass=True,dataset=ds,shot=shot,seed=seed,validation_count=len(valid),actual_per_class=counts,underfilled_classes=[j for j,n in enumerate(counts) if n<shot],fixed_eval_overlap=0))
 print('D3_STRATIFIED_HEAD',shot,seed,ds,'validation',len(valid),'underfilled',sum(n<shot for n in counts),stats,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=lambda v:int(os.environ['SLURM_ARRAY_TASK_ID']) if v.startswith('$') else int(v));a=p.parse_args();main(a.index)
