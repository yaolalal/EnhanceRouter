from common import *
setup()
import torch
from eargtc.data import get_texts,_install_pyg_optional_extension_stubs
from eargtc.router_v7_decision.hidden import answer_hidden_and_logits
from eargtc.router_v7.text_pipeline import align_text_replay
from eargtc.numbered_prompts import build_numbered_prompt
from eargtc.prompts import DIRECT_PROMPTS
from transformers import AutoModelForCausalLM,AutoTokenizer
_install_pyg_optional_extension_stubs()
assert torch.cuda.device_count()>=2
modelpath='/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct'
tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side='left';tokenizer.truncation_side='left'
model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float16,device_map='balanced',max_memory={0:'9GiB',1:'9GiB','cpu':'0GiB'});model.eval()
ds='cora';node=read(RUN/'shared/text-request-v2-cora.json')['support_ids'][0] if 'support_ids' in read(RUN/'shared/text-request-v2-cora.json') else read(RUN/'shared/text-request-cora.json')['support_ids'][0]
assert node not in read(RUN/'EVALUATION_IDS.json')[ds]
ref=read(next((SOURCE/f'runs/router_v7_text_logits_20260815_v2/text_truncate_050/{ds}/chunks').glob('*.jsonl.meta.json')))
template=Path(ref['prompt_template_path']).read_text().rstrip() if ref.get('prompt_template_path') else None
texts=[]
for name in ['text_clean_replay','text_truncate_050']:
 data=torch.load(SOURCE/f'runs/router_v7_text_inputs_20260815_v3/{name}/datasets/{ds}.pt',map_location='cpu',weights_only=False)
 texts.append(get_texts(data)[node]);del data
prompts=[]
for t in texts:
 human=f'{t}\n{template}' if template is not None else build_numbered_prompt(text=t,base_prompt=DIRECT_PROMPTS[ds],label_names=ref['label_names'])
 prompts.append(tokenizer.apply_chat_template([{'role':'user','content':human}],tokenize=False,add_generation_prompt=True))
enc=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=int(ref['max_input_length']))
enc={k:v.to('cuda:0') for k,v in enc.items()};enc['position_ids']=(enc['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
with torch.inference_mode():h,num=answer_hidden_and_logits(model,enc,ref['number_token_ids'])
logits=torch.empty_like(num,dtype=torch.float32);logits[:,ref['number_to_label_ids']]=num.float();prob=logits.softmax(1).cpu()
aligned=align_text_replay(frozen_clean=canonical(ds)['llm_logits'][[node]].softmax(1),runtime_clean=prob[:1],runtime_shifted=prob[1:],changed_mask=torch.tensor([texts[0]!=texts[1]]))
assert tuple(h.shape)==(2,4096) and torch.isfinite(h).all() and torch.isfinite(aligned).all() and torch.allclose(aligned.sum(1),torch.ones(1),atol=1e-5)
result=dict(all_checks_pass=True,job=os.environ['SLURM_JOB_ID'],visible_gpus=torch.cuda.device_count(),device_map=model.hf_device_map,hidden_shape=list(h.shape),probability_shape=list(prob.shape),fixed_eval_overlap=0,scope='real fp16 two-GPU clean/perturbed forward and original label-order/alignment checks; not experiment accuracy',metadata=metadata())
write(RUN/f'LLM_GPU_SMOKE-{os.environ["SLURM_JOB_ID"]}.json',result)
print('LLM_GPU_SMOKE_PASS',json.dumps({k:v for k,v in result.items() if k!='metadata'}),flush=True)
