"""Reconstruct all75 B0 GNN heads and confirm frozen LLM outputs agree across split seeds."""
from d3_recovery import *
setup()
import torch
from eargtc.models.gnn import MLPHead
idx=read(ROOT/'configs/inputs.json');cache={};rows=[]
for sh,se,ds in CELLS:
 if ds not in cache:
  c=idx['cells'][f'5/42/{ds}'];cache[ds]=(load(c['inputs']['embeddings'])['embeddings'].float(),load(c['inputs']['upstream'])['llm_logits'].float())
 g,llm=cache[ds];state=head_state(sh,se,ds);expected=upstream_bank(sh,se,ds)
 head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(g),4096):parts.append(head(g[start:start+4096]))
 actual=torch.cat(parts);reference=expected['gnn_logits'].float()
 torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-4)
 assert torch.equal(actual.argmax(1),reference.argmax(1)),(sh,se,ds,'expert top1 mismatch')
 torch.testing.assert_close(expected['llm_logits'].float(),llm,rtol=0,atol=0)
 rows.append(dict(key=key(sh,se,ds),gnn_head_logits_match=True,gnn_top1_exact=True,LLM_logits_exact=True,max_gnn_abs=float((actual-reference).abs().max()),expert_model_seed=gnn_training_seed(ds)))
 print('EXPERT_RECONSTRUCTED',sh,se,ds,flush=True)
write(ROOT/'reports/fairrouter-baseline/EXPERT_AUDIT_20261027.json',dict(all_checks_pass=True,cells=75,rows=rows,evaluation_labels_used=False,source=metadata(),audit_code_sha256=digest(__file__)))
print('ALL75_EXPERTS_PASS',flush=True)
