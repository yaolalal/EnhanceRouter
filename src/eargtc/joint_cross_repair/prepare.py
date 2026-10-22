"""Build new safe exports exclusively from previously label-isolated Router data."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import torch

from .artifacts import (DATASETS, sha256, verify_files, read_json, write_json, save_tensor,
                        id_hash, validate_features, validate_supervision, validate_hidden)

ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROUTER = ROOT / 'runs/router_bc_improved_structural_oof_fixed_20260729_v1'
SAFE_SOURCE = ROOT / 'runs/joint_direct_numbered_router_export_20260813_v1'
HIDDEN_SOURCE = ROOT / 'runs/router_v7_decision_llm_hidden_20260827_v1'
SPLIT_SOURCE = ROOT / 'runs/router_bc_split_c_20260825_v3'
DEFAULT_EXPORT = ROOT / 'runs/joint_cross_repair_export_20260906_v2'


def split_id_hash(ids):
    return hashlib.sha256(''.join(f'{int(i)}\n' for i in ids).encode('utf-8')).hexdigest()


def historical_validation_split(dataset, validation_ids, agreement, historical):
    from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split
    selection, threshold, audit = [], [], {}
    for module in ['conflict', 'agreement']:
        same = agreement[validation_ids]
        ids = validation_ids[same if module == 'agreement' else ~same]
        split = deterministic_calibration_threshold_split(ids, dataset, module)
        cal, thr = ids[split.calibration], ids[split.threshold]
        expected = historical['modules'][module]['calibration']
        if split_id_hash(cal) != expected['calibration_ids_sha256'] or split_id_hash(thr) != expected['threshold_ids_sha256']:
            raise ValueError(f'historical validation split mismatch: {dataset}/{module}')
        selection.append(cal); threshold.append(thr)
        audit[module] = {'calibration_ids_sha256': split_id_hash(cal), 'threshold_ids_sha256': split_id_hash(thr),
                         'calibration_count':len(cal), 'threshold_count':len(thr)}
    return torch.cat(selection).sort().values, torch.cat(threshold).sort().values, audit


def prepare_dataset(dataset, output_root=DEFAULT_EXPORT, candidate_ids=None):
    from .policy import select_topk_policy, build_pools
    if dataset not in DATASETS:
        raise ValueError('unknown dataset')
    dest = Path(output_root).resolve() / dataset
    if dest.exists():
        raise FileExistsError(f'refusing to overwrite export: {dest}')
    source = SAFE_SOURCE / dataset
    complete_path = source / 'complete.json'
    complete = read_json(complete_path)
    if not complete.get('all_checks_pass') or complete.get('test_labels_exported') is not False:
        raise ValueError('historical export did not pass label isolation')
    needed = ['router_tensors.pt', 'router_nodes.jsonl', 'split_ids.json', 'training_supervision.pt',
              'replay_audit.json', 'safe_input_metadata.json']
    hashes = {str(source / name): complete['artifacts'][name]['sha256'] for name in needed}
    verify_files(hashes)
    hashes[str(complete_path)] = sha256(complete_path)
    replay = read_json(source / 'replay_audit.json')
    frozen_path = FROZEN_ROUTER / dataset / 'metrics.json'
    frozen = read_json(frozen_path)
    if not replay.get('all_checks_pass') or not replay.get('node_set_exact'):
        raise ValueError('Router replay did not pass')
    if 'router_v5_structural_oof_fixed_20260729_v1' not in frozen['provenance']['predictions']:
        raise ValueError('wrong structural OOF source')
    selected_checkpoint = frozen['scheme_b']['selected_checkpoint']
    if isinstance(selected_checkpoint, dict):
        selected_checkpoint = selected_checkpoint.get('name', selected_checkpoint.get('checkpoint'))
    if replay['selected_b_checkpoint'] != selected_checkpoint:
        raise ValueError(f'wrong frozen B checkpoint: {selected_checkpoint}')
    if replay['selected_c_checkpoint'] != frozen['scheme_c']['metrics']['validation_selection']['selected']:
        raise ValueError('wrong frozen C checkpoint')
    old = torch.load(source / 'router_tensors.pt', map_location='cpu', weights_only=True)
    raw_sup = torch.load(source / 'training_supervision.pt', map_location='cpu', weights_only=True)
    expected_sup = {'gold_support_ids', 'gold_support_targets', 'official_validation_ids', 'official_validation_targets'}
    if set(raw_sup) != expected_sup:
        raise ValueError('historical supervision schema changed')
    splits = read_json(source / 'split_ids.json')
    n, c = old['gnn_logits'].shape
    import json
    rows = [json.loads(line) for line in (source / 'router_nodes.jsonl').open() if line.strip()]
    from eargtc.joint_direct_numbered.router_replay import LABEL_FREE_NODE_KEYS
    if len(rows) != n or any(set(r) != LABEL_FREE_NODE_KEYS or r['node_id'] != i for i, r in enumerate(rows)):
        raise ValueError('historical Router row schema/IDs changed')
    features = {'node_ids':torch.arange(n), 'gnn_embeddings':old['gnn_embeddings'].float(),
                'logp_g':old['gnn_logits'].float().log_softmax(1), 'logp_l':old['llm_logits'].float().log_softmax(1),
                'choose_g':torch.tensor([r['scheme_b_route']=='gnn' for r in rows]),
                'agreement':torch.tensor([r['agreement'] for r in rows]),
                'selected_pred':torch.tensor([r['scheme_b_selected_pred'] for r in rows]),
                'trust':torch.tensor([r['scheme_c_selected_trust'] for r in rows],dtype=torch.float64)}
    validate_features(features)
    g, l = features['logp_g'].argmax(1), features['logp_l'].argmax(1)
    for i, row in enumerate(rows):
        if row['gnn_pred'] != int(g[i]) or row['llm_pred'] != int(l[i]):
            raise ValueError('expert prediction alignment mismatch')
        if row['scheme_c_checkpoint'] != replay['selected_c_checkpoint']:
            raise ValueError('trust checkpoint mismatch')
        key='scheme_c_'+row['scheme_c_checkpoint']+'_trust'
        if abs(row['scheme_c_selected_trust']-row[key])>1e-6:
            raise ValueError('selected trust does not match frozen C')
    gold, val, test = [torch.tensor(splits[k],dtype=torch.long) for k in ['train','validation','query']]
    if not torch.equal(gold,raw_sup['gold_support_ids']) or not torch.equal(val,raw_sup['official_validation_ids']):
        raise ValueError('safe labels do not match frozen split order')
    historical_path = SPLIT_SOURCE / dataset / 'metrics.json'
    selection, threshold, split_audit = historical_validation_split(dataset, val, features['agreement'], read_json(historical_path))
    val_lookup = {int(i):int(y) for i,y in zip(val,raw_sup['official_validation_targets'])}
    supervision = {'gold_ids':gold,'gold_targets':raw_sup['gold_support_targets'],
                   'selection_ids':selection,'selection_targets':torch.tensor([val_lookup[int(i)] for i in selection]),
                   'threshold_ids':threshold,'test_ids':test}
    validate_supervision(supervision,n,c)
    masks = {'conflict_gnn':~features['agreement'] & features['choose_g'],
             'conflict_llm':~features['agreement'] & ~features['choose_g'], 'agreement':features['agreement']}
    policies={}
    for name, mask in masks.items():
        ids=threshold[mask[threshold]]
        correct=torch.tensor([int(features['selected_pred'][i])==val_lookup[int(i)] for i in ids],dtype=torch.bool)
        policies[name]=select_topk_policy(features['trust'][ids],correct,ids)
        policies[name]['threshold_population_count']=len(ids)
    candidates = test.clone() if candidate_ids is None else torch.as_tensor(candidate_ids,dtype=torch.long)
    pools=build_pools(test,features['agreement'],features['choose_g'],features['trust'],policies,candidate_ids=candidates)
    hidden_path=HIDDEN_SOURCE/dataset/'hidden.pt'
    hidden=torch.load(hidden_path,map_location='cpu',weights_only=True,mmap=True)
    validate_hidden(hidden,n)
    if hidden['metadata']['dataset'] != dataset or hidden['metadata']['row_order_hash'] != id_hash(features['node_ids']):
        raise ValueError('hidden provenance/row hash mismatch')
    for p in [frozen_path,historical_path,hidden_path]:hashes[str(p.resolve())]=sha256(p)
    # Only split hashes are read from historical split-C; its fitted scores are not used.
    meta={'schema_version':1,'dataset':dataset,'node_count':n,'class_count':c,'gnn_width':int(features['gnn_embeddings'].shape[1]),
          'llm_width':4096,'llm_hidden_path':str(hidden_path.resolve()),'source_hashes':hashes,
          'router_root':str(FROZEN_ROUTER),'selected_b_checkpoint':replay['selected_b_checkpoint'],
          'selected_c_checkpoint':replay['selected_c_checkpoint'],'split_audit':split_audit,
          'protocol':'transductive_full_test_pseudo_candidates' if candidate_ids is None else 'transductive_test_subset_pseudo_candidates',
          'candidate_ids':candidates.tolist(),'pseudo_candidate_count':len(candidates),'test_labels_exported':False,
          'score':'frozen_20260729_scheme_c_selected_trust','policy_strata':list(masks),
          'gold_count':len(gold),'model_selection_count':len(selection),'threshold_count':len(threshold),'test_count':len(test),
          'pool_counts':{k:len(v) for k,v in pools.items()},'upstream_validation_reuse':True,
          'llm_cache_limitation':'Arxiv: 2/173 audit argmax mismatches; raw hidden reused, original expert logits retained.'}
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=dest.with_name(dest.name+f'.staging.{os.getpid()}');stage.mkdir()
    save_tensor(stage/'features.pt',features);save_tensor(stage/'supervision.pt',supervision);save_tensor(stage/'pools.pt',pools)
    write_json(stage/'policies.json',policies);write_json(stage/'metadata.json',meta)
    names=['features.pt','supervision.pt','pools.pt','policies.json','metadata.json']
    write_json(stage/'complete.json',{'schema_version':1,'dataset':dataset,'test_labels_exported':False,
                                   'artifacts':{name:sha256(stage/name) for name in names}})
    os.rename(stage,dest)
    return meta
