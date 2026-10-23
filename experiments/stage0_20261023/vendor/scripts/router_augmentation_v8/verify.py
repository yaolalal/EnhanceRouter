"""Independent artifact audit. Only safe original gold targets are evaluated."""
import csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from eargtc.router_augmentation_v8.sources import *
from eargtc.router_augmentation_v8.data import ENVIRONMENTS,head_view,as_v6_bank
from eargtc.router_v8_gold_training.training import _validate
from eargtc.router_v8.evidence import PROBABILITY_CANDIDATE_NAMES,STRUCTURAL_CANDIDATE_NAMES,PROBABILITY_GLOBAL_NAMES,STRUCTURAL_GLOBAL_NAMES
from eargtc.router_v8.prototypes import GRAPH_CANDIDATE_NAMES,GRAPH_GLOBAL_NAMES


def verify_dataset(d):
    source=gold_source(d);ids=source['ids'];gold={int(i):int(y) for i,y in zip(ids,source['y'])}
    dest=RUN/'banks'/d;meta=read(dest/'manifest.json')
    for name,h in meta['output_hashes'].items():assert sha(dest/name)==h,name
    for name,h in meta['input_hashes'].items():assert sha(name)==h,name
    for root,file in [(GRAPH,'graph_environments.pt'),(TEXT,'text_environments.pt')]:
        manifest=read(root/d/'manifest.json');expected=next(r['sha256'] for r in manifest['files'] if r['path']==file)
        assert sha(root/d/file)==expected
    rows=load(dest/'rows.pt');probs=load(dest/'expert_probabilities.pt');n=len(rows['sample_id'])
    assert n==len(ids)*6
    assert torch.equal(probs['sample_id'],rows['sample_id'])
    assert rows['sample_id'].unique().numel()==n
    assert torch.equal(rows['weight'],torch.ones(n))
    assert set(rows['node_ids'].tolist())==set(gold)
    for ei,env in enumerate(ENVIRONMENTS):
        m=rows['environment_index']==ei;node=rows['node_ids'][m]
        assert torch.equal(node,ids)
        assert torch.equal(rows['sample_id'][m],ei*source['meta']['node_count']+ids)
        cached=cached_environment(d,env);pos=positions(cached['node_ids'],ids)
        assert torch.equal(rows['gnn_pred'][m],cached['gnn_pred'][pos])
        assert torch.equal(rows['llm_pred'][m],cached['llm_pred'][pos])
    for i in range(n):
        g,l,y=int(rows['gnn_pred'][i]),int(rows['llm_pred'][i]),gold[int(rows['node_ids'][i])]
        assert int(rows['gold_target'][i])==y
        if g==l:
            assert bool(rows['agreement_mask'][i]) and not bool(rows['disagreement_mask'][i])
            assert int(rows['agreement_target'][i])==int(g==y)
            assert int(rows['trust_target'][i])==int(rows['preference_target'][i])==-1
        else:
            assert bool(rows['disagreement_mask'][i]) and not bool(rows['agreement_mask'][i])
            assert int(rows['agreement_target'][i])==-1
            any_correct=(g==y or l==y)
            assert int(rows['trust_target'][i])==int(any_correct)
            assert bool(rows['preference_mask'][i])==any_correct
            assert int(rows['preference_target'][i])==(int(g==y) if any_correct else -1)
    head_counts={}
    for branch in ('agreement','disagreement'):
        bank=load(dest/(branch+'_train.pt'));mask=rows[branch+'_mask']
        assert torch.equal(bank['ids'],rows['sample_id'][mask])
        assert torch.equal(bank['node_ids'],rows['node_ids'][mask])
        native=as_v6_bank(bank,gold_ids=ids)
        if branch=='disagreement':_validate(native,True)
        for head in (('agreement',) if branch=='agreement' else ('trust','preference')):
            six=load(dest/(head+'_v6.pt'));seven=load(dest/(head+'_v7.pt'))
            expected=head_view(bank,head,'v6')
            for key,value in expected.items():assert torch.equal(six[key],value),(d,head,key)
            for mode in ('prob','full','compact','context','embed'):
                expected7=head_view(bank,head,'v7',mode)
                for key in ('sample_id','node_ids','target','weight'):assert torch.equal(seven[key],expected7[key])
                assert torch.equal(seven['features'][mode],expected7['x'])
                assert bool(torch.isfinite(seven['features'][mode]).all())
            assert set(six)=={'sample_id','node_ids','target','weight','tab','g','l'}
            assert torch.equal(six['weight'],torch.ones(len(six['target'])))
            head_counts[head]=dict(rows=len(six['target']),positive=int(six['target'].sum()),negative=int((six['target']==0).sum()))
    graph_audit=read(RUN/'replay'/d/'graph_audit.json');text_audit=read(RUN/'replay'/d/'text_audit.json')
    assert all(r['cached_argmax_mismatch']==0 for r in graph_audit['records'])
    prompt_audits=[]
    frozen_prompt_hash=load(HIDDEN/d/'hidden.pt')['metadata']['prompt_hash']
    from eargtc.prompts import DIRECT_PROMPTS
    import hashlib
    for record in text_audit['records']:
        assert sha(record['text_path'])==record['text_sha256']
        assert sha(record['reference_path'])==record['reference_sha256']
        metadata_path=Path(record['reference_path']+'.meta.json')
        reference=read(metadata_path)
        template_path=reference.get('prompt_template_path')
        template=Path(template_path).read_text().rstrip() if template_path else None
        protocol=dict(dataset=d,base_prompt=DIRECT_PROMPTS.get(d),label_names=reference['label_names'],prompt_template=template,use_chat_template=bool(reference['use_chat_template']),max_input_length=int(reference['max_input_length']),number_to_label_ids=reference['number_to_label_ids'])
        prompt_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert reference['use_chat_template'] is True
        assert prompt_hash==frozen_prompt_hash,(d,'prompt protocol differs')
        prompt_audits.append(dict(environment=record['environment'],metadata_path=str(metadata_path),metadata_sha256=sha(metadata_path),template_path=template_path,template_sha256=sha(template_path) if template_path else None,prompt_hash=prompt_hash))
    return dict(dataset=d,observations=n,heads=head_counts,clean_parity=meta['clean_parity'],text_runtime_diagnostics=text_audit['records'],prompt_audits=prompt_audits)


if __name__=='__main__':
    require_slurm();torch.set_num_threads(2)
    records=[verify_dataset(d) for d in DATASETS]
    total={head:{key:sum(r['heads'][head][key] for r in records) for key in ('rows','positive','negative')} for head in ('agreement','trust','preference')}
    report=dict(all_checks_pass=True,slurm_job_id=os.environ['SLURM_JOB_ID'],datasets=records,totals=total,total_observations=sum(r['observations'] for r in records),expert_or_router_training_performed=False,validation_query_nodes_in_training=0)
    write(RUN/'VERIFIED.json',report)
    candidate=PROBABILITY_CANDIDATE_NAMES+STRUCTURAL_CANDIDATE_NAMES+GRAPH_CANDIDATE_NAMES
    glob=PROBABILITY_GLOBAL_NAMES+STRUCTURAL_GLOBAL_NAMES+GRAPH_GLOBAL_NAMES
    schema=dict(v6=dict(disagreement_tab=[f'candidate_g:{n}' for n in candidate]+[f'candidate_l:{n}' for n in candidate]+[f'symmetric:{n}' for n in glob],agreement_tab=[f'candidate:{n}' for n in candidate]+[f'symmetric:{n}' for n in glob],g_width=128,l_width=4096),v7=dict(function='eargtc.router_v8_v7_explore.core_v2.raw_features',modes=['prob','full','compact','context','embed'],disagreement_widths=read(RUN/'banks/cora/manifest.json')['widths']['disagreement'],agreement_widths=read(RUN/'banks/cora/manifest.json')['widths']['agreement']),excluded_inputs=['gold_target','state','targets','node_id','sample_id','environment_index','weight'])
    write(RUN/'FEATURE_SCHEMA.json',schema)
    lines=['# Router-v8(v6/v7) augmentation alignment','',
           'Built from original gold parents only; six equally weighted observations per node: clean plus text truncate, mixed mask, span delete, graph dropout and mixed ego. Full frozen GNN predictions are used, as explicitly requested. No Router training or threshold/model selection was performed.','',
           '| Dataset | Observations | Agreement (+ / -) | Trust (+ / -) | Preference (GNN / LLM) |','|---|---:|---:|---:|---:|']
    for r in records:
        h=r['heads'];lines.append(f"| {r['dataset']} | {r['observations']} | {h['agreement']['positive']} / {h['agreement']['negative']} | {h['trust']['positive']} / {h['trust']['negative']} | {h['preference']['positive']} / {h['preference']['negative']} |")
    lines+=['',f"Total observations: {report['total_observations']}. Head counts: `{json.dumps(total)}`.",'',
            'Files per dataset: rows.csv/rows.pt record identity and all masked targets; expert_probabilities.pt holds historical environment distributions; agreement_train.pt and disagreement_train.pt hold tab/g/l banks; {agreement,trust,preference}_v6.pt contain eligible raw inputs and targets; corresponding _v7.pt files contain five raw feature modes and the same eligible targets. FEATURE_SCHEMA.json records names and widths.','',
            'Agreement target = consensus prediction equals gold. Disagreement trust = GNN or LLM correct. Preference = GNN correct, with neither-correct rows excluded. Outside a head mask the row-table target is -1, not a negative training example. Every eligible row has weight 1. Identical-valued environments are retained.','',
            'Identity: sample_id = environment_index * num_nodes + original_node_id (unique within each dataset); group cross-validation by (dataset, node_id), never by sample_id. Neither identity nor environment is a feature. Original train/validation/query membership is unchanged.','',
            'v6 clean tab/g/l and state/target are exactly equal to archived training banks. Augmented Build70/Build43 probabilities and candidate-dependent structural/prototype features are recomputed. The prototype reference remains the original clean gold support, matching v8; this preserves its historical gold self-inclusion. Graph rows use regenerated shifted GNN embeddings. Text rows use newly extracted final-prompt-token LLM hidden states. Unassigned mixed rows retain exact clean inputs.','',
            'Historical calibrated text probabilities and their class predictions are retained. Fresh hidden extraction can have software/hardware runtime drift; raw logit probability differences and argmax differences against the original replay are reported in replay/*/text_audit.json. The new hidden vectors are not claimed to be historical byte-identical reconstructions.','',
            'v7 views are prob/full/compact/context/embed from the historical core_v2 function. No learned scaling, pruning, rank normalization or model-specific transform is fitted here. These are to be fit on eligible training rows by a future experiment.','',
            'Use load_head(root, dataset, head, version, mode) from eargtc.router_augmentation_v8.data. Native v6 bank conversion as_v6_bank validates gold parent IDs and strips metadata not accepted by the old head trainer. Frozen historical experiment drivers still use their original gold-ID guards; do not overwrite or run those drivers on the new sample IDs. Future orchestration must audit original parent IDs using this dataset.','',
            'Only safe gold labels construct targets. Legacy environment containers physically include whole-graph labels; the adapter whitelists query probabilities/predictions/IDs and discards the other fields. No validation/query parents enter output. Text span-delete, previously a held-out operator, is explicitly included for gold-parent training in this namespace; it must not be described as an untouched operator test in a future experiment.','',
            'Historical experts and feature sources are external hash-bound dependencies. Existing Joint, Router-v8 and early router_v7 artifacts were not overwritten. Numerical construction and verification ran under Slurm.']
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
    code=list((ROOT/'eargtc/router_augmentation_v8').glob('*.py'))+list((ROOT/'scripts/router_augmentation_v8').glob('*'))+[ROOT/'tests/test_router_augmentation_v8.py']
    imports=[ROOT/'eargtc/router_v8/evidence.py',ROOT/'eargtc/router_v8/prototypes.py',ROOT/'eargtc/router_v8_v7_explore/core_v2.py',ROOT/'eargtc/router_v7/perturbations.py',ROOT/'eargtc/router_v7_decision/hidden.py']
    write(RUN/'CODE_SEAL.json',dict(files={str(p):sha(p) for p in code+imports if p.is_file()}))
    print(json.dumps(report),flush=True)
