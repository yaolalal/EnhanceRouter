"""Shot/seed-specific six-view gold banks with frozen label-independent replay inputs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from eargtc.router_augmentation_v8.data import ENVIRONMENTS, organize
from eargtc.router_augmentation_v8.sources import positions, sha
from scripts.router_augmentation_v8.build import build70

RUN = REPO / 'runs/router_v10_shots_v1_20260914'
HIST = REPO / 'runs/router_v10_multiseed_data_20260912_v1'
OLD = REPO / 'runs/multiseed_reproduction_20260909_v1'
HIDDEN = REPO / 'runs/router_v7_decision_llm_hidden_20260827_v1'
TEXT_INPUT = REPO / 'runs/router_v7_text_inputs_20260815_v3'
TEXT_LOGITS = REPO / 'runs/router_v7_text_logits_20260815_v2'
FROZEN = REPO.parent / 'GNN-as-Judge/results/ablations/no_gnn_retrain_3_10shot/frozen_splits'
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
QUERY_KEYS = {'dataset', 'ids', 'tab', 'g', 'l', 'gnn_pred', 'llm_pred'}


def require():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical work requires Slurm')


def load(path):
    require()
    return torch.load(path, weights_only=True, map_location='cpu', mmap=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(RUN.resolve()):
        raise ValueError('New run outputs only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save(path, value):
    require()
    path = Path(path)
    if not path.resolve().is_relative_to(RUN.resolve()):
        raise ValueError('New run outputs only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(value, f)


class Inputs:
    def __init__(self):
        complete = read(HIST / 'COMPLETE.json')
        assert complete['all_checks_pass']
        seal_path = HIST / 'SOURCE_SEAL.json'
        assert sha(seal_path) == complete['files']['SOURCE_SEAL.json']
        self.hashes = {str(HIST / 'COMPLETE.json'): sha(HIST / 'COMPLETE.json'), str(seal_path): sha(seal_path)}
        self.seal = read(seal_path)['files']
        self.seal.update({str(HIST / p): h for p, h in complete['files'].items()})

    def bind(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        known = expected or self.seal.get(str(path))
        if known is not None and digest != known:
            raise ValueError('Source hash changed: ' + str(path))
        self.hashes[str(path)] = digest
        return path

    def load(self, path, expected=None):
        return load(self.bind(path, expected))


def assemble_training(dataset, environments, ids, targets, n):
    """Every parent contributes exactly one row per registered view."""
    if [e['environment'] for e in environments] != list(ENVIRONMENTS):
        raise ValueError('Exactly six ordered environments required')
    if any(not torch.equal(e['node_ids'], ids) for e in environments):
        raise ValueError('Every view must retain the exact gold support parents')
    return organize(dataset, environments, gold_ids=ids, gold_targets=targets, num_nodes=n)


def evaluation_bank(dataset, ids, tab70, g, l, pg, pl, *, branch, targets=None):
    """Only validation receives target/state; query schema excludes truth."""
    if branch not in ('agreement', 'disagreement'):
        raise ValueError('Unknown branch')
    mask = (pg == pl) == (branch == 'agreement')
    tab = tab70[mask]
    if branch == 'agreement':
        tab = torch.cat((tab[:, :27], tab[:, 54:]), 1)
    b = dict(dataset=dataset, ids=ids[mask], tab=tab.float().contiguous(),
             g=g[mask].float().contiguous(), l=l[mask].half().contiguous(),
             gnn_pred=pg[mask], llm_pred=pl[mask])
    if targets is not None:
        if branch == 'agreement':
            from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split
            halves = deterministic_calibration_threshold_split(b['ids'], dataset, 'agreement')
            b.update(target=pg[mask] == targets[mask], cal_mask=halves.calibration, policy_mask=halves.threshold)
        else:
            from eargtc.router_v8_learned.data import _historical_masks
            cal, pol = _historical_masks(b['ids'], dataset)
            b.update(state=torch.where(pg[mask] == targets[mask], 0, torch.where(pl[mask] == targets[mask], 1, 2)),
                     cal_mask=cal, policy_mask=pol)
    elif set(b) != QUERY_KEYS:
        raise ValueError('Query schema contains unknown fields')
    return b


def desired_support(dataset, inputs):
    wanted = set()
    manifest = read(inputs.bind(FROZEN / 'manifest.json'))
    records = {(r['dataset'], r['shots'], r['seed']): r for r in manifest['rows']}
    for shot in (3, 10):
        for seed in (42, 43, 44):
            s = read(inputs.bind(FROZEN / f'{dataset}_{shot}shot_seed{seed}.json', records[(dataset, shot, seed)]['frozen_sha256']))
            assert s['dataset'] == dataset and s['shots'] == shot and s['seed'] == seed
            wanted.update(s['train_ids'])
    return torch.tensor(sorted(wanted), dtype=torch.long)


def prepare_text(dataset):
    """Replay only nodes missing from the fixed historical union; no labels used."""
    require()
    inputs = Inputs()
    dest = RUN / 'shared/text' / dataset
    if (dest / 'COMPLETE.json').exists():
        verify_artifacts(dest)
        return
    ids = desired_support(dataset, inputs)
    cached = {name: inputs.load(HIST / 'shared/text' / dataset / (name + '.pt')) for name in ENVIRONMENTS[1:4]}
    available = set(cached[ENVIRONMENTS[1]]['node_ids'].tolist())
    assert all(set(v['node_ids'].tolist()) == available for v in cached.values())
    missing = torch.tensor([i for i in ids.tolist() if i not in available], dtype=torch.long)
    outputs = {}
    audit = dict(dataset=dataset, union_support_count=len(ids), cached_count=len(ids)-len(missing),
                 replayed_count=len(missing), no_labels_used=True, legacy_container_policy='text/IDs/fixed LLM probabilities only; label fields never accessed')
    print('TEXT_COVERAGE', json.dumps(audit), flush=True)
    if len(missing):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from eargtc.data import load_graph_dataset, get_texts
        from eargtc.numbered_prompts import build_numbered_prompt
        from eargtc.prompts import DIRECT_PROMPTS
        from eargtc.router_v7_decision.hidden import answer_hidden_and_logits
        from eargtc.router_v7.text_pipeline import align_text_replay
        from scripts.router_v7_decision.extract_llm_answer_hidden import _artifact_hash
        model_path = Path('/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct')
        hidden = inputs.load(HIDDEN / dataset / 'hidden.pt')
        clean_hidden = hidden['hidden'][missing].clone()
        hiddenmeta = hidden['metadata']
        # Clean LLM is a fixed zero-shot source. Discard the supervised GNN field immediately.
        up = OLD / 'seed42/upstream' / dataset
        cert = read(inputs.bind(up / 'COMPLETE.json'))
        frozen_logits = inputs.load(up / 'pipeline/predictions.pt', cert['files']['pipeline/predictions.pt'])['llm_logits']
        frozen_prob = frozen_logits[missing].float().softmax(1)
        model_hash = _artifact_hash(model_path, ['config.json', 'generation_config.json', 'model.safetensors.index.json'])
        tokenizer_hash = _artifact_hash(model_path, ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])
        assert hiddenmeta['model_hash'] == model_hash and hiddenmeta['tokenizer_hash'] == tokenizer_hash
        for name in ('config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'):
            if (model_path/name).exists(): inputs.bind(model_path/name)
        ref = read(inputs.bind(next((TEXT_LOGITS/'text_truncate_050'/dataset/'chunks').glob('*.jsonl.meta.json'))))
        template = inputs.bind(ref['prompt_template_path']).read_text().rstrip() if ref.get('prompt_template_path') else None
        protocol = dict(dataset=dataset,base_prompt=DIRECT_PROMPTS.get(dataset),label_names=ref['label_names'],prompt_template=template,use_chat_template=bool(ref['use_chat_template']),max_input_length=int(ref['max_input_length']),number_to_label_ids=ref['number_to_label_ids'])
        prompt_hash = hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert ref['use_chat_template'] is True and prompt_hash == hiddenmeta['prompt_hash']
        tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        tokenizer.padding_side='left';tokenizer.truncation_side='left'
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float16,device_map='auto')
        model.eval(); device=model.get_input_embeddings().weight.device
        clean_texts=None
        for name in ('text_clean_replay',)+ENVIRONMENTS[1:4]:
            inputs.bind(TEXT_INPUT/name/'datasets'/f'{dataset}.pt')
            data=load_graph_dataset(dataset,TEXT_INPUT/name,torch.device('cpu'))
            texts=get_texts(data);node_texts=[texts[int(i)] for i in missing]
            del data,texts
            if clean_texts is None: clean_texts=node_texts
            changed=torch.tensor([a!=b for a,b in zip(clean_texts,node_texts)])
            hs=[];ps=[]
            with torch.inference_mode():
                for i,text in enumerate(node_texts):
                    human=f'{text}\n{template}' if template is not None else build_numbered_prompt(text=text,base_prompt=DIRECT_PROMPTS[dataset],label_names=ref['label_names'])
                    prompt=tokenizer.apply_chat_template([{'role':'user','content':human}],tokenize=False,add_generation_prompt=True)
                    enc=tokenizer([prompt],return_tensors='pt',padding=True,truncation=True,max_length=ref['max_input_length'])
                    enc={k:v.to(device) for k,v in enc.items()}
                    h,num=answer_hidden_and_logits(model,enc,ref['number_token_ids'])
                    logits=torch.empty_like(num,dtype=torch.float32);logits[:,ref['number_to_label_ids']]=num.float()
                    hs.append(h.cpu().half());ps.append(logits.softmax(1).cpu())
                    if (i+1)%50==0: print(dataset,name,i+1,'/',len(missing),flush=True)
            h=torch.cat(hs);prob=torch.cat(ps)
            if name=='text_clean_replay':
                runtime_clean=prob
                continue
            h[~changed]=clean_hidden[~changed]
            aligned=align_text_replay(frozen_clean=frozen_prob,runtime_clean=runtime_clean,runtime_shifted=prob,changed_mask=changed)
            legacy=inputs.load(REPO/'runs/router_v7_text_environments_20260815_v2'/dataset/'text_environments.pt')['environments'][name]['query']
            # Whitelist only IDs and fixed LLM probabilities from the legacy container.
            legacy={k:legacy[k] for k in ('node_ids','llm_prob')}
            lookup={int(n):j for j,n in enumerate(legacy['node_ids'])}
            has=torch.tensor([int(i) in lookup for i in missing])
            if bool(has.any()):
                pos=torch.tensor([lookup[int(i)] for i in missing[has]])
                aligned[has]=legacy['llm_prob'][pos]
            aligned[~changed]=frozen_prob[~changed]
            outputs[name]=dict(node_ids=missing,l=h,llm_prob=aligned,changed_mask=changed)
        del model,hidden,frozen_logits
        torch.cuda.empty_cache();gc.collect()
        audit.update(model_hash=model_hash,tokenizer_hash=tokenizer_hash,prompt_hash=prompt_hash,gpu=torch.cuda.get_device_name())
    for name in ENVIRONMENTS[1:4]:
        old=cached[name]; fields=('l','llm_prob','changed_mask')
        both_ids=torch.cat((old['node_ids'],missing))
        pos=positions(both_ids,ids)
        value=dict(node_ids=ids)
        for k in fields:
            combined=torch.cat((old[k],outputs[name][k])) if len(missing) else old[k]
            value[k]=combined[pos]
        save(dest/(name+'.pt'),value)
    write(dest/'COMPLETE.json',dict(**audit,input_hashes=inputs.hashes,
          output_hashes={p.name:sha(p) for p in dest.glob('*.pt')},slurm_job_id=os.environ['SLURM_JOB_ID']))


def verify_artifacts(dest):
    meta=read(Path(dest)/'COMPLETE.json')
    for p,h in meta['input_hashes'].items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    for name,h in meta['output_hashes'].items():
        if sha(Path(dest)/name)!=h:raise ValueError('Output changed: '+name)
    return meta


def build(shot,seed,dataset):
    require()
    from eargtc.router_v8.prototypes import fit_class_prototypes
    from eargtc.models.gnn import MLPHead
    from eargtc.router_v7.perturbations import mixed_assignment
    from eargtc.multiseed_v8.data import _hidden
    root=RUN/f'shot{shot}';dest=root/f'seed{seed}/banks'/dataset
    if (dest/'manifest.json').exists():
        meta=read(dest/'manifest.json')
        assert meta['dataset']==dataset and meta['shots']==shot and meta['split_seed']==seed
        for name,h in meta['bank_hashes'].items():assert sha(dest/name)==h
        for path,h in meta['input_hashes'].items():assert sha(path)==h
        return meta
    if dest.exists():raise FileExistsError('Incomplete bank directory must be inspected')
    inputs=Inputs();up=root/f'seed{seed}/upstream'/dataset
    cert=read(inputs.bind(up/'COMPLETE.json'))
    assert cert['all_checks_pass'] and cert['test_truth_exported'] is False and cert['split_seed']==seed and cert['shot']==shot
    def upstream(name):return inputs.load(up/name,cert['files'][name])
    s=read(inputs.bind(root/'splits'/f'{dataset}_{shot}shot_seed{seed}.json',cert['split_sha256']))
    assert s['split_seed']==seed and s['dataset']==dataset
    sup=upstream('supervision.pt');raw=upstream('pipeline/predictions.pt');embedding=upstream('embeddings.pt');graph=upstream('graph.pt')
    assert set(sup)=={'support_ids','support_targets','valid_ids','valid_targets'}
    assert set(raw)=={'gnn_logits','llm_logits'} and set(embedding)=={'embeddings','node_ids'}
    assert set(graph)=={'edge_index','num_nodes'}
    n=s['num_nodes'];ids=sup['support_ids'];y=sup['support_targets'];g=embedding['embeddings'];edge=graph['edge_index']
    assert ids.tolist()==s['support_ids'] and sup['valid_ids'].tolist()==s['valid_ids']
    parts=[s[k] for k in ('support_ids','valid_ids','unlabeled_ids')]
    assert len(set(i for p in parts for i in p))==sum(map(len,parts))==n
    assert torch.equal(embedding['node_ids'],torch.arange(n)) and graph['num_nodes']==n
    pg=raw['gnn_logits'].float().softmax(1);pl=raw['llm_logits'].float().softmax(1);classes=pg.shape[1]
    hidden=_hidden(dataset,n,inputs.hashes)
    fitted=fit_class_prototypes(g,ids,y,num_classes=classes)
    historical_graph=inputs.load(OLD/'seed42/upstream'/dataset/'graph.pt')
    assert historical_graph['num_nodes']==n and torch.equal(historical_graph['edge_index'],edge)
    full_pred=pg.argmax(1)
    clean=build70(pg[ids],pl[ids],ids,edge,full_pred,g,fitted,n,classes)
    envs=[]
    textroot=RUN/'shared/text'/dataset
    textcert=verify_artifacts(textroot);inputs.bind(textroot/'COMPLETE.json')
    state=upstream('gnn_head.pt')
    head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.)
    head.load_state_dict(state);head.eval()
    with torch.inference_mode():check=head(g)
    torch.testing.assert_close(check,raw['gnn_logits'],atol=2e-5,rtol=2e-5)
    assigned=mixed_assignment(range(n),seed=303)
    for name in ENVIRONMENTS:
        eg=g[ids].clone();el=hidden[ids].clone();epg=pg[ids].clone();epl=pl[ids].clone();tab=clean.clone()
        if name.startswith('text'):
            replay=inputs.load(textroot/(name+'.pt'),textcert['output_hashes'][name+'.pt'])
            pos=positions(replay['node_ids'],ids);changed=replay['changed_mask'][pos]
            el=replay['l'][pos];epl=replay['llm_prob'][pos].clone();epl[~changed]=pl[ids][~changed]
            assert torch.equal(el[~changed],hidden[ids][~changed])
            tab=build70(epg,epl,ids,edge,full_pred,g,fitted,n,classes)
            tab[~changed]=clean[~changed]
        elif name.startswith('graph'):
            replay=inputs.load(HIST/'shared/graph'/dataset/(name+'.pt'))
            shifted=replay['g'];shift_edge=replay['edge_index']
            assert shifted.shape==g.shape
            changed=torch.tensor([assigned[int(i)]=='graph' for i in ids]) if 'mixed' in name else torch.ones(len(ids),dtype=torch.bool)
            with torch.inference_mode():prob=head(shifted).softmax(1)
            eg=shifted[ids].clone();eg[~changed]=g[ids][~changed]
            epg=prob[ids].clone();epg[~changed]=pg[ids][~changed]
            # Historical prototype semantics change only support-row embeddings; reference prototypes stay clean.
            proto_g=g.clone();proto_g[ids]=eg
            tab=build70(epg,epl,ids,shift_edge,prob.argmax(1),proto_g,fitted,n,classes)
            tab[~changed]=clean[~changed]
        envs.append(dict(environment=name,node_ids=ids,gnn_prob=epg,llm_prob=epl,tab70=tab,g=eg,l=el))
    assembled=assemble_training(dataset,envs,ids,y,n)
    import uuid
    final_dest=dest
    dest=dest.parent/('.'+dataset+'.views-'+uuid.uuid4().hex)
    dest.mkdir(parents=True)
    for branch in ('agreement','disagreement'):save(dest/(branch+'_train.pt'),assembled[branch])
    save(dest/'rows.pt',assembled['rows'])
    for role,key in (('validation','valid_ids'),('query','unlabeled_ids')):
        rows=torch.tensor(s[key],dtype=torch.long)
        tab=build70(pg[rows],pl[rows],rows,edge,full_pred,g,fitted,n,classes)
        for branch in ('agreement','disagreement'):
            b=evaluation_bank(dataset,rows,tab,g[rows],hidden[rows],full_pred[rows],pl.argmax(1)[rows],branch=branch,
                              targets=sup['valid_targets'] if role=='validation' else None)
            save(dest/f'{branch}_{role}.pt',b)
    meta=dict(schema='router_v10_shots_six_view_v1',dataset=dataset,shots=shot,split_seed=seed,
              num_nodes=n,num_classes=classes,gold_ids=ids.tolist(),support_ids=ids.tolist(),
              validation_ids=s['valid_ids'],query_ids=s['unlabeled_ids'],environments=list(ENVIRONMENTS),
              weight='one per observation; six views per support parent',
              clean_train_predictions='fresh same-shot same-seed full supervised head; historical gold-view semantics',
              prototype_reference='same-shot same-seed clean gold SSL prototypes, including parent self-reference',
              query_labels_read=False,training_validation_rows=0,training_query_rows=0,
              input_hashes=inputs.hashes,bank_hashes={p.name:sha(p) for p in dest.glob('*.pt')},
              slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(dest/'manifest.json',meta)
    dest.rename(final_dest)
    print('BANK_COMPLETE',shot,seed,dataset,flush=True)
    return meta


if __name__=='__main__':
    require();torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['text','build']);p.add_argument('--dataset',required=True,choices=DATASETS)
    p.add_argument('--shot',type=int,choices=[3,10]);p.add_argument('--seed',type=int,choices=[42,43,44]);a=p.parse_args()
    if a.stage=='text':prepare_text(a.dataset)
    else:
        if a.shot is None or a.seed is None:p.error('build requires shot and seed')
        build(a.shot,a.seed,a.dataset)
