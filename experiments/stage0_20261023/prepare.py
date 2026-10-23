from common import *

def main():
 setup();import numpy as np,torch,random
 RUN.mkdir(parents=True,exist_ok=True)
 protocol=dict(stage=0,dev_seeds=DEV,held_out_seeds=HELDOUT,historical_seeds=[42,43,44],shots=SHOTS,datasets=DS,
  held_out_policy='reserved; no training, selection or metric computation until final confirmation',
  evaluation=dict(count=1000,shared_across_shots_and_all_seeds=True,selection_seed=20261023,
   sampling='label-blind sample of intersection of historical 10shot Query across seeds42/43/44; avoid all official train candidates when feasible',
   reserve_before_new_support_and_validation=True,level1_population='agreement/disagreement subsets of this same fixed1000',
   level2_population='same fixed1000; historical Table1 replay kept separately'),
  baseline=dict(router_recipe='historical seed42 selected recipe at matching dataset/shot; scorer family and feature mode fixed',
   router_validation='native 3fold calibration, route grid and acceptance search refitted on each dev split',
   joint_catalog='unique historical winners from seeds42/43/44 for same dataset/shot, selected only on own validation',
   gold_normalization='historical recipe normalization retained before native new-validation calibration',
   experts='frozen encoder and LLM; GNN head retrained for each new support and for small-validation D3'),
  d1=dict(fractions=[.25,.5,.75,1.],unit='original target-support parent; all 6 views kept/dropped together',
   frozen='experts, prototypes, sources, feature modes, normalization, affine, rho and Joint configs',scope='router supervision amount; not total expert label budget'),
  d2=dict(sizes=[50,200,1000],primary_N=200,selection_seed=73019,
   random_pool='Query minus fixed evaluation1000',agreement_pool='top20percent by frozen B0 agreement scorer within Query minus fixed evaluation1000; cap2000',
   actual_N='min(requested, available); report requested and actual counts separately',
   oracle=True,truth_allowed_only_for='selected oracle parents and post-fit evaluation',source_only_heads='retain source-only scope; never silently pool target',
   views=list(ENVIRONMENTS),per_observation_weights='native unit weights; dataset masses unchanged'),
  d3=dict(validation_per_class='k; choose from original validation deterministically',
   refit='GNN checkpoint selection, native router calibration/route/rho, Joint checkpoint/config selection on reduced validation',
   min20_acceptance='retain original threshold; reject-all when validation has insufficient accepted rows'),
  d4=dict(variants=['confidence_router','confidence_preference'],acceptance='exact B0 accepted counts per branch',level2='both variants, identical frozen selected Joint hyperparameters'),
  statistics=dict(paired_unit='dataset,shot,dev_seed',cells=75,bootstrap_replicates=10000,bootstrap_seed=731901,
   CI='percentile 2.5/97.5 of paired cell means',dataset_average='3 shots x5 seeds, equal cell weight',
   L1_gate='95% CI lower>0; at least4/5 positive dataset deltas; no dataset mean<-1pp',
   L2_gate='mean accuracy delta>=0 and CI not significantly negative',
   stage1_decision='preregistered D2b N200 primary; other N descriptive; apply L1 gate'),
  graph_view_resolution='Stage0 uses exact paper B.4: clean+3 text+2 graph, no extra third graph perturbation',
  undefined_auroc='record null when only one class; never replace with a fabricated value')
 write(RUN/'PROTOCOL.json',protocol)
 idx=read(ROOT/'configs/inputs.json');allids={};union={};sources=set()
 for ds in DS:
  g=graph(ds);n=g.num_nodes;y=g.y.reshape(-1).long();sources.add(idx['cells'][f'5/42/{ds}']['graph'])
  historic=[read(idx['cells'][f'10/{s}/{ds}']['inputs']['split']) for s in [42,43,44]]
  eligible=set.intersection(*(set(x['unlabeled_ids']) for x in historic))
  candidate_train=set(torch.where(g.train_mask)[0].tolist())
  outside=eligible-candidate_train
  if len(outside)>=1000:eligible=outside
  ids=random.Random(20261023).sample(sorted(eligible),1000);allids[ds]=ids;reserved=set(ids)
  c=canonical(ds)['llm_logits'].shape[1]
  support_union=set();oracle_union=set()
  for seed in DEV:
   gen=torch.Generator().manual_seed(seed)
   support_order={}
   for label in range(c):
    candidates=torch.where((y==label)&g.train_mask)[0]
    candidates=torch.tensor([int(i) for i in candidates if int(i) not in reserved],dtype=torch.long)
    support_order[label]=candidates[torch.randperm(len(candidates),generator=gen)][:10].tolist()
   if ds=='arxiv':valid=historic[0]['valid_ids']
   else:
    candidates=torch.tensor([int(i) for i in torch.where(~g.train_mask)[0] if int(i) not in reserved],dtype=torch.long)
    valid=sorted(candidates[torch.randperm(len(candidates),generator=gen)[:500]].tolist())
   assert not set(valid)&reserved
   for shot in SHOTS:
    support=sorted(i for label in range(c) for i in support_order[label][:shot]);query=sorted(set(range(n))-set(support)-set(valid))
    sp=dict(dataset=ds,shots=shot,split_seed=seed,num_nodes=n,support_ids=support,valid_ids=valid,unlabeled_ids=query,
      evaluation_full_ids=query,evaluation_1000_ids=ids,evaluation_test_ids=ids,standard_eval_ids=ids,
      graph_path=idx['cells'][f'5/42/{ds}']['graph'],test_truth_exported=False)
    assert set(ids)<=set(query)
    write(RUN/f'splits/{shot}/{seed}/{ds}.json',sp)
    support_union.update(support)
    choices=[i for i in query if i not in reserved]
    rng=random.Random(73019+seed);rng.shuffle(choices);oracle_union.update(choices[:1000])
  # Text hidden states: all support parents, plus random-oracle parents only on datasets whose target scorers consume hL.
  wanted=support_union|(oracle_union if ds in ('cora','citeseer','ogbn-products') else set())
  union[ds]=sorted(wanted)
  write(RUN/f'shared/text-request-{ds}.json',dict(dataset=ds,node_ids=sorted(wanted),support_ids=sorted(support_union),random_oracle_ids=sorted(oracle_union)))
  print('SPLITS',ds,'eval',len(ids),'support union',len(support_union),'text request',len(wanted),flush=True)
  for shot in SHOTS:
   for s in [42,43,44]:
    rec=idx['cells'][f'{shot}/{s}/{ds}'];sources.update(rec['inputs'].values());sources.add(rec['checkpoint']);sources.add(rec['predictions'])
    for branch in rec['branches'].values():sources.update(branch.values())
  sources.update([SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/shared/graph/{ds}/{env}.pt' for env in ENVIRONMENTS[4:]])
  sources.add(SOURCE/f'runs/router_v7_text_environments_20260815_v2/{ds}/text_environments.pt')
  sources.add(SOURCE/f'runs/gcn_ssl_v2_selected_20260623_v1/{ds}/mlp_metrics.json')
  for env in ENVIRONMENTS[1:4]:
   sources.add(SOURCE/f'runs/router_v7_text_inputs_20260815_v3/{env}/datasets/{ds}.pt')
   sources.add(SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/shared/text/{ds}/{env}.pt')
   sources.add(SOURCE/f'runs/router_v10_shots_v1_20260914/shared/text/{ds}/{env}.pt')
  sources.add(next((SOURCE/f'runs/router_v7_text_logits_20260815_v2/text_truncate_050/{ds}/chunks').glob('*.jsonl.meta.json')))
 write(RUN/'EVALUATION_IDS.json',allids)
 weights=Path('/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct')
 for p in weights.iterdir():
  if p.is_file() and (p.suffix in ('.json','.safetensors','.model') or p.name in ('tokenizer.model',)):sources.add(p)
 frozen={}
 for j,p in enumerate(sorted(set(map(str,sources))),1):
  p=Path(p)
  if not p.is_file():continue
  stat=p.stat();frozen[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
  if j%30==0:print('SOURCE_HASH',j,len(sources),flush=True)
 write(RUN/'SOURCE_FILES.json',frozen)
 write(RUN/'environment.json',metadata())
 write(RUN/'PREPARED.json',dict(cells=75,source_files=len(frozen),dev_seeds=DEV,held_out_seeds=HELDOUT,held_out_evaluations=0,
  protocol_sha256=digest(RUN/'PROTOCOL.json'),eval_ids_sha256=digest(RUN/'EVALUATION_IDS.json')))

if __name__=='__main__':main()
