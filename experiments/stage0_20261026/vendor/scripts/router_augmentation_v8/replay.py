"""Recover missing frozen representations, with no expert or router training."""
import argparse, gc, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from eargtc.router_augmentation_v8.sources import *
from eargtc.router_augmentation_v8.data import ENVIRONMENTS


def replay_graph(dataset):
    from eargtc.data import load_graph_dataset
    from eargtc.models.gnn import GNNEncoder,MLPHead
    from eargtc.router_v7.perturbations import edge_dropout,ego_edge_mask,mixed_assignment
    source=gold_source(dataset);ids=source['ids'];meta=source['meta'];n=meta['node_count']
    graph_path=Path(meta['graph_path'])
    assert sha(graph_path)==meta['graph_path_sha256']
    data=load_graph_dataset(dataset,graph_path.parent.parent,torch.device('cpu'))
    x=data.x.float();edge=data.edge_index;del data
    cfg_path=ROOT/'runs/router_v5_structural_oof_fixed_20260729_v1'/dataset/'pipeline/config.json'
    cfg=read(cfg_path);ep=SSL/dataset/'encoder_and_embeddings.pt'; hp=SSL/dataset/'mlp_head.pt'
    payload=load(ep);head_payload=load(hp)
    encoder=GNNEncoder(x.shape[1],cfg['gnn_hidden_dim'],128,cfg['gnn_layers'],cfg['gnn_type'],cfg['dropout'])
    encoder.load_state_dict(payload['encoder_state']);encoder.eval()
    state=head_payload['head_state'];head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.)
    head.load_state_dict(state);head.eval()
    assigned=mixed_assignment(range(n),seed=303)
    selected=[i for i,c in assigned.items() if c=='graph']
    edges={'graph_drop_030':edge_dropout(edge,severity=.3,seed=101,num_nodes=n),
           'graph_ego_mixed_050':ego_edge_mask(edge,node_ids=selected,severity=.5,seed=303,num_nodes=n)}
    audits=[]
    for name,shift_edge in edges.items():
        with torch.inference_mode():emb=encoder(x,shift_edge);prob=head(emb).softmax(1)
        cached=cached_environment(dataset,name);pos=positions(cached['node_ids'],ids)
        mask=torch.tensor([int(i) in set(selected) for i in ids]) if 'mixed' in name else torch.ones(len(ids),dtype=torch.bool)
        difference=(prob[ids][mask]-cached['gnn_prob'][pos][mask]).abs()
        full_pred=prob.argmax(1)
        output_emb=emb[ids].clone();output_emb[~mask]=source['raw']['gnn_embeddings'][ids[~mask]]
        audit=dict(dataset=dataset,environment=name,assigned_gold=int(mask.sum()),cached_probability_max_diff=float(difference.max()) if difference.numel() else 0.,cached_argmax_mismatch=int((full_pred[ids][mask]!=cached['gnn_pred'][pos][mask]).sum()),embedding_changed_rows=int((output_emb!=source['raw']['gnn_embeddings'][ids]).any(1).sum()))
        if audit['cached_argmax_mismatch']:raise ValueError(f'GNN replay prediction drift: {audit}')
        save(RUN/'replay'/dataset/(name+'.pt'),dict(node_ids=ids,g=output_emb,edge_index=shift_edge,full_gnn_pred=full_pred,changed_mask=mask))
        audits.append(audit);print(json.dumps(audit),flush=True)
        del emb,prob;gc.collect()
    write(RUN/'replay'/dataset/'graph_audit.json',dict(slurm_job_id=os.environ['SLURM_JOB_ID'],records=audits,input_hashes={str(p):sha(p) for p in (graph_path,cfg_path,ep,hp)}))


def replay_text(datasets):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from eargtc.data import load_graph_dataset,get_texts
    from eargtc.numbered_prompts import build_numbered_prompt
    from eargtc.prompts import DIRECT_PROMPTS
    from eargtc.router_v7_decision.hidden import answer_hidden_and_logits
    from scripts.router_v7_decision.extract_llm_answer_hidden import _artifact_hash
    model_path=Path('/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left';tokenizer.truncation_side='left'
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16,device_map='auto')
    model.eval();device=model.get_input_embeddings().weight.device
    model_hash=_artifact_hash(model_path,['config.json','generation_config.json','model.safetensors.index.json'])
    tokenizer_hash=_artifact_hash(model_path,['tokenizer.json','tokenizer_config.json','special_tokens_map.json'])
    for dataset in datasets:
        source=gold_source(dataset);ids=source['ids']
        frozen=load(HIDDEN/dataset/'hidden.pt')
        assert frozen['metadata']['model_hash']==model_hash
        assert frozen['metadata']['tokenizer_hash']==tokenizer_hash
        clean_hidden=frozen['hidden'][ids].clone();del frozen
        audit=[];runtime_clean=None;clean_texts=None
        for name in ('text_clean_replay','text_truncate_050','text_mask_mixed_030','text_span_delete_030'):
            text_path=TEXT_INPUT/name/'datasets'/f'{dataset}.pt'
            data=load_graph_dataset(dataset,TEXT_INPUT/name,torch.device('cpu'))
            texts=get_texts(data);node_texts=[texts[int(i)] for i in ids];del data,texts
            if clean_texts is None:clean_texts=node_texts
            changed=torch.tensor([a!=b for a,b in zip(clean_texts,node_texts)])
            logit_root=CLEAN_LOGITS if name=='text_clean_replay' else TEXT_LOGITS
            chunk=next((logit_root/name/dataset/'chunks').glob('*.jsonl'))
            reference=read(str(chunk)+'.meta.json')
            template_path=reference.get('prompt_template_path')
            template=Path(template_path).read_text().rstrip() if template_path else None
            map_ids=reference['number_to_label_ids'];token_ids=reference['number_token_ids']
            hidden_rows=[];prob_rows=[]
            with torch.inference_mode():
                for i,text in enumerate(node_texts):
                    human=f'{text}\n{template}' if template is not None else build_numbered_prompt(text=text,base_prompt=DIRECT_PROMPTS[dataset],label_names=reference['label_names'])
                    prompt=tokenizer.apply_chat_template([{'role':'user','content':human}],tokenize=False,add_generation_prompt=True)
                    enc=tokenizer([prompt],return_tensors='pt',padding=True,truncation=True,max_length=reference['max_input_length'])
                    enc={k:v.to(device) for k,v in enc.items()}
                    h,numbered=answer_hidden_and_logits(model,enc,token_ids)
                    logits=torch.empty_like(numbered,dtype=torch.float32)
                    logits[:,map_ids]=numbered.float()
                    hidden_rows.append(h.cpu().half());prob_rows.append(logits.softmax(1).cpu())
                    if (i+1)%50==0:print(dataset,name,i+1,'/',len(ids),flush=True)
            hidden=torch.cat(hidden_rows);prob=torch.cat(prob_rows)
            wanted=set(ids.tolist());expected={}
            with chunk.open() as f:
                for line in f:
                    row=json.loads(line)
                    if row['node_id'] in wanted:expected[row['node_id']]=row['probabilities']
            ep=torch.tensor([expected[int(i)] for i in ids],dtype=torch.float32);ep=ep/ep.sum(1,keepdim=True)
            record=dict(environment=name,changed_text_rows=int(changed.sum()),raw_probability_max_diff=float((prob-ep).abs().max()),raw_argmax_mismatch=int((prob.argmax(1)!=ep.argmax(1)).sum()),text_path=str(text_path),text_sha256=sha(text_path),reference_path=str(chunk),reference_sha256=sha(chunk))
            if name=='text_clean_replay':
                runtime_clean=prob;record['clean_hidden_max_diff']=float((hidden-clean_hidden).abs().max())
            else:
                hidden[~changed]=clean_hidden[~changed]
            save(RUN/'replay'/dataset/(name+'.pt'),dict(node_ids=ids,l=hidden,runtime_prob=prob,changed_mask=changed))
            audit.append(record);print(json.dumps(record),flush=True)
        write(RUN/'replay'/dataset/'text_audit.json',dict(slurm_job_id=os.environ['SLURM_JOB_ID'],model_hash=model_hash,tokenizer_hash=tokenizer_hash,model_path=str(model_path),hidden_semantics='final attended prompt token from frozen LLM; changed rows freshly replayed, unchanged rows exact frozen clean; historical aligned probabilities retained separately',records=audit))
        gc.collect()


if __name__=='__main__':
    require_slurm();torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['graph','text']);p.add_argument('--datasets',nargs='+',default=list(DATASETS));args=p.parse_args()
    if args.kind=='graph':
        for d in args.datasets:replay_graph(d)
    else:replay_text(args.datasets)
