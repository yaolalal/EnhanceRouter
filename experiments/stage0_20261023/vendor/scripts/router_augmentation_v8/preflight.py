import os, json
from pathlib import Path
import torch
assert os.environ.get('SLURM_JOB_ID')
ROOT=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EnhanceRouter')
torch.set_num_threads(2)
records=[]
for d in ['cora','citeseer','pubmed','arxiv','ogbn-products']:
    safe=ROOT/'runs/joint_direct_numbered_router_export_20260813_v1'/d
    raw=torch.load(safe/'router_tensors.pt',map_location='cpu',weights_only=True,mmap=True)
    sup=torch.load(safe/'training_supervision.pt',map_location='cpu',weights_only=True)
    ids=sup['gold_support_ids']
    old=torch.load(ROOT/'runs/router_v7_graph_environments_20260815_v2'/d/'graph_environments.pt',map_location='cpu',weights_only=True,mmap=True)
    clean=old['environments']['clean']['query']; lookup={int(n):i for i,n in enumerate(clean['node_ids'])}; pos=torch.tensor([lookup[int(n)] for n in ids])
    hidden=torch.load(ROOT/'runs/router_v7_decision_llm_hidden_20260827_v1'/d/'hidden.pt',map_location='cpu',weights_only=True,mmap=True)
    ssl=torch.load(ROOT/'runs/gcn_ssl_v2_selected_20260623_v1'/d/'encoder_and_embeddings.pt',map_location='cpu',weights_only=True,mmap=True)
    record=dict(dataset=d,gold_count=len(ids),gold_ids_equal_old=torch.equal(ids,old['train_ids']),gnn_probability_max_diff=float((raw['gnn_logits'][ids].softmax(1)-clean['gnn_prob'][pos]).abs().max()),llm_probability_max_diff=float((raw['llm_logits'][ids].softmax(1)-clean['llm_prob'][pos]).abs().max()),gnn_embedding_max_diff=float((raw['gnn_embeddings'][ids]-ssl['embeddings'][ids]).abs().max()),hidden_metadata=hidden['metadata'],ssl_keys=list(ssl),full_gnn_equal_support=torch.equal(clean['gnn_pred'][pos],old['environments']['clean']['support']['gnn_pred']))
    records.append(record); print(json.dumps(record),flush=True)
(ROOT/'runs/router_augmentation_v8_20260911_v1/preflight.json').write_text(json.dumps(dict(slurm_job_id=os.environ['SLURM_JOB_ID'],records=records),indent=2))
