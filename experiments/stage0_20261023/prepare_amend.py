from common import *
setup()
import random
for ds in DS:
 req=read(RUN/f'shared/text-request-{ds}.json');wanted=set(req['support_ids']);test=set(read(RUN/'EVALUATION_IDS.json')[ds])
 for seed in DEV:
  sp=split(10,seed,ds);query=set(sp['unlabeled_ids'])-test
  nodes=list(range(sp['num_nodes']));random.Random(73019+seed).shuffle(nodes)
  selected=[i for i in nodes if i in query][:1000];wanted.update(selected)
  write(RUN/f'shared/random-oracle-v2-{ds}-{seed}.json',dict(node_ids=selected,seed=seed,shared_across_shots=True))
 write(RUN/f'shared/text-request-v2-{ds}.json',dict(node_ids=sorted(wanted),dataset=ds,support_ids=req['support_ids']))
 print('AMEND_REQUEST',ds,len(wanted),flush=True)
write(RUN/'PROTOCOL_AMENDMENT_01.json',dict(before_any_dev_router_fit_or_evaluation=True,heldout_untouched=True,
 reason='Actual legacy text cache is sparse, not full-node. Missing perturbed probabilities must also be freshly inferred. PubMed10 preference uses hL.',
 random_oracle_sampling='Uniform shuffle of all canonical node IDs using original preregistered seed73019+dev_seed; filter Query10 minus fixed1000; reuse exact parent order across shots',
 impact='label-blind coupling across shots reduces duplicate LLM inference; N and eligible unlabeled sampling law unchanged except reserving max10shot support for all shots',
 views='exact original clean+3text+2graph; frozen original fp16 LLM, no quantization; low-memory GPUs use weight offload only',metadata=metadata()))
