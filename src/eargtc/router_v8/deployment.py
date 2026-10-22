"""Target support scalar adaptation and disjoint validation calibration/policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .artifacts import write_json
from .experiment import DATASETS,load_supports,select_target,verify_code
from .training import fit_model,select_features,predict_logits
from .calibration import fit_heads,calibrated_logits,probabilities
from .metrics import evaluate,selective_metrics
from .policy import fit_risk_policy,apply_policy


def validation_halves(ids,dataset):
    ids=np.asarray(ids)
    if len(np.unique(ids))!=len(ids) or len(ids)<2:
        raise ValueError('validation requires at least two distinct conflict IDs')
    # Exact historical split-C conflict algorithm, independently implemented.
    hashes=[hashlib.sha256(f'{dataset}|conflict|{int(i)}'.encode()).digest() for i in ids]
    mask=np.array([bool(h[0]&1) for h in hashes])
    if mask.all() or not mask.any():
        order=sorted(range(len(ids)),key=lambda i:hashes[i]); mask[:]=False
        mask[order[:len(ids)//2]]=True
    return mask,~mask


def run_target(root,target):
    root=Path(root); destination=root/'targets'/target
    if (root/'selection'/'FROZEN.json').exists(): raise ValueError('already frozen')
    if (destination/'selection.json').exists(): raise FileExistsError('target already completed')
    pre=json.loads((root/'PREREGISTERED.json').read_text())
    verify_code(pre)
    selected=select_target(root,target)
    cfg=selected['best']['config']; group=pre['feature_groups'][cfg['evidence']]
    supports=load_supports(root)
    banks={d:select_features(b,group) for d,b in supports.items()}
    model,transform,_=fit_model([banks[d] for d in DATASETS if d!=target],cfg)
    support=transform.apply(banks[target]); c,r=predict_logits(model,support)
    z=support['state'].numpy()
    adaptation=fit_heads(c,r,z,bias_only=True,strength=20.)
    ca,ra=calibrated_logits(c,r,adaptation)
    support_diagnostic={'before':evaluate(probabilities(c,r),z),'after_bias':evaluate(probabilities(ca,ra),z),
        'warning':'after-bias is in-sample adaptation diagnostic, not selection or held-out performance'}
    vb=torch.load(root/'banks'/target/'validation.pt',map_location='cpu',weights_only=False)
    vb=transform.apply(select_features(vb,group))
    vc,vr=predict_logits(model,vb); vc,vr=calibrated_logits(vc,vr,adaptation)
    vz=vb['state'].numpy(); ids=vb['node_ids'].numpy()
    cal,policy=validation_halves(ids,target)
    calibration=fit_heads(vc[cal],vr[cal],vz[cal],strength=2.,min_per_class=5)
    cc,cr=calibrated_logits(vc,vr,calibration); q=probabilities(cc,cr)
    pq=q[policy]; pz=vz[policy]
    score=pq[:,:2].max(1); correct=pq[:,:2].argmax(1)==pz
    policies={str(risk):fit_risk_policy(score,correct,risk=risk) for risk in pre['secondary_risks']}
    validation_metrics=dict(before_calibration=evaluate(probabilities(vc[policy],vr[policy]),pz,ids[policy]),
        after_calibration=evaluate(pq,pz,ids[policy]),
        policies={risk:selective_metrics(pq,pz,apply_policy(pq,p)) for risk,p in policies.items()})
    qb=torch.load(root/'banks'/target/'query.pt',map_location='cpu',weights_only=False)
    if any(k in qb for k in ('state','labels','y','y_true','correctness')):
        raise ValueError('query bank must be label-free')
    query=transform.apply(select_features(qb,group))
    qc,qr=predict_logits(model,query)
    qa,qb_r=calibrated_logits(qc,qr,adaptation)
    final_c,final_r=calibrated_logits(qa,qb_r,calibration)
    final_q=probabilities(final_c,final_r)
    destination.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=cfg,state_dict=model.state_dict(),transform=transform.state_dict(),
        view=transform.view,group=group,source_fit_ids={d:supports[d]['node_ids'] for d in DATASETS if d!=target}),destination/'model.pt')
    torch.save(dict(node_ids=query['node_ids'],gnn_pred=query['gnn_pred'],llm_pred=query['llm_pred'],
        raw_trust_logit=torch.from_numpy(qc),raw_preference_logit=torch.from_numpy(qr),
        trust_logit=torch.from_numpy(final_c),preference_logit=torch.from_numpy(final_r),
        q=torch.from_numpy(final_q)),destination/'scores.pt')
    torch.save(dict(node_ids=vb['node_ids'],state=vb['state'],q=torch.from_numpy(q),calibration_mask=torch.from_numpy(cal),policy_mask=torch.from_numpy(policy)),destination/'validation_scores.pt')
    write_json(selected,destination/'source_selection.json')
    write_json(dict(target=target,config=cfg,adaptation=adaptation,calibration=calibration,policies=policies,
        calibration_ids=ids[cal].tolist(),policy_ids=ids[policy].tolist(),
        validation_split='sha256(dataset|conflict|node_id)[0]&1; exact historical split-C algorithm',
        support_diagnostic=support_diagnostic,validation_metrics=validation_metrics,
        negative_slopes={k:v['slope'] for k,v in calibration.items() if v['slope']<0},
        selected_by='source OOF LODO only; no target-validation architecture selection'),destination/'selection.json')
    print(json.dumps(dict(target=target,config=cfg,source_nll=selected['best']['macro_nll'],
        validation_policy=validation_metrics['after_calibration'],primary=validation_metrics['policies']['0.1'])),flush=True)
