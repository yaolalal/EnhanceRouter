from common import *
setup()
import random
for ds in DS:
 support=set()
 for seed in DEV:
  sp=split(10,seed,ds);support.update(sp['support_ids']);reserved=set(evaluation_ids(seed,ds))
  query=set(sp['unlabeled_ids'])-reserved;nodes=list(range(sp['num_nodes']));random.Random(73019+seed).shuffle(nodes)
  selected=[i for i in nodes if i in query][:1000]
  write(RUN/f'shared/random-oracle-v2-{ds}-{seed}.json',dict(node_ids=selected,seed=seed,shared_across_shots=True,excludes_own_seed_eval1000=True))
 write(RUN/f'shared/text-request-v2-{ds}.json',dict(node_ids=sorted(support),support_ids=sorted(support),dataset=ds,scope='Support only for B0; D2 adds parents after frozen B0 selection'))
write(RUN/'CACHE_REQUESTS_READY.json',dict(seeds=list(DEV),held_out_seeds=[],query_labels_used=False))
