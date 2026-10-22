"""Port of Original Router-v10 inference/refit orchestration; kernels are copied."""
import copy
import importlib
from fractions import Fraction
import hashlib
import json

import joblib
import numpy as np
from scipy.special import expit, logit
import torch

from .io import DATASETS
from .policy import apply_pair

FEATURES = {'dataset', 'ids', 'tab', 'g', 'l', 'gnn_pred', 'llm_pred'}
ENGINES = dict(v6='eargtc.router_aug_search.v6_engine', v7='eargtc.router_aug_search.v7_engine',
               expanded='eargtc.router_aug_search.v7_expanded', tune='enhancerouter.tune_engine')


def source_key(cfg, head, sources, banks):
    c = {k:v for k,v in cfg.items() if k not in ('id','target_steps')}
    h = hashlib.sha256(json.dumps(dict(head=head, config=c), sort_keys=True).encode())
    for d in sources:
        h.update(d.encode())
        for k in ('ids','node_ids','tab','g','l','state'):
            v = np.ascontiguousarray(np.asarray(banks[d][k]))
            h.update(json.dumps([k,str(v.dtype),list(v.shape)]).encode())
            h.update(v.tobytes(order='C'))
    return h.hexdigest()


def engine(name):
    module = importlib.import_module(ENGINES[name])
    if name == 'v6':
        module._source_key = source_key
    return module


def predict(wrapped, bank):
    return engine(wrapped['engine']).predict_candidate(wrapped['model'], bank)


def raw(wrapped, bank):
    z = predict(wrapped, bank)
    if 'gold_normalization' in wrapped:
        a = wrapped['gold_normalization']
        return a['slope'] * z + a['bias']
    if wrapped['engine'] == 'v6' and wrapped['model']['head'] == 'agreement':
        return logit(np.clip(z, 1e-6, 1-1e-6))
    m = wrapped['model']
    return (z-m['center'])/m['scale'] if 'center' in m else z


def predict_bundle(bundle, bank):
    if bundle.get('unavailable') or not len(bank['ids']):
        return dict(score=np.zeros(len(bank['ids'])), route=np.zeros(len(bank['ids']),dtype=np.int64),
                    q=np.tile([0.,0.,1.],(len(bank['ids']),1)))
    if bundle.get('tuning_schema') != 'crossfit_v1':
        if bundle['branch'] == 'agreement':
            return dict(score=predict(bundle['models']['agreement'], bank))
        z = {h:bundle['affine'][h]['slope']*predict(bundle['models'][h],bank)+bundle['affine'][h]['bias']
             for h in ('trust','preference')}
        p, t = expit(z['preference']), expit(z['trust'])
        route = (p < bundle['boundary']).astype(int)
        return dict(score=t*np.where(route==0,p,1-p), route=route,
                    q=np.column_stack((t*p,t*(1-p),1-t)))
    if bundle['branch'] == 'agreement':
        z = raw(bundle['models']['agreement'],bank); a=bundle['calibration']['agreement']
        return dict(score=expit(a['slope']*z+a['bias']))
    return apply_pair({h:raw(w,bank) for h,w in bundle['models'].items()},
                      bundle['calibration'],bundle['route_recipe'])


def top_mask(score, ids, fraction):
    score = np.asarray(score); ids = np.asarray(ids)
    assert np.isfinite(score).all() and len(ids)==len(np.unique(ids))
    ratio = Fraction(str(fraction)); assert 0 <= ratio <= 1
    k = (len(ids)*ratio.numerator+ratio.denominator-1)//ratio.denominator
    keep = np.zeros(len(ids),dtype=bool)
    keep[np.lexsort((ids,-score))[:k]] = True
    return keep


def rebuild(inputs, key, upstream, refit=True):
    shot, seed, ds = key.split('/'); cell=inputs.index['cells'][key]; split=inputs.split(key)
    n=split['num_nodes']; gp=upstream['gnn_logits'].argmax(1); lp=upstream['llm_logits'].argmax(1)
    agree=gp==lp; choose=torch.ones(n,dtype=torch.bool); accepted=torch.zeros(n,dtype=torch.bool)
    scores=torch.zeros(n,dtype=torch.float64); diagnostics=[]; models={}
    for branch in ('agreement','disagreement'):
        source=cell['branches'][branch]
        deployed=joblib.load(inputs.path(source['bundle']))
        fitted=copy.deepcopy(deployed)
        if refit:
            banks={}
            for d in DATASETS:
                other=f'{shot}/{seed}/{d}'; sp=inputs.split(other)
                b=inputs.tensor(inputs.index['cells'][other]['branches'][branch]['train'])
                parents=set(b['node_ids'].tolist())
                assert b['split']=='train' and parents<=set(sp['support_ids'])
                assert not parents & set(sp['valid_ids']+sp['unlabeled_ids'])
                assert np.array_equal(np.asarray(b['weight']),np.ones(len(b['ids'])))
                banks[d]=b
            for head,w in fitted.get('models',{}).items():
                w['model']=engine(w['engine']).fit_candidate(copy.deepcopy(w['model']['config']),head,banks,ds)
        models[branch]=fitted
        for role in ('validation','query'):
            b=inputs.tensor(source[role])
            assert set(b)<=FEATURES|{'split','state','target','cal_mask','policy_mask'}
            if role=='query':assert set(b)<=FEATURES|{'split'}
            ids=b['ids']; expected=split['valid_ids' if role=='validation' else 'unlabeled_ids']
            assert len(ids)==len(set(ids.tolist())) and set(ids.tolist())<=set(expected)
            assert bool(((b['gnn_pred']==b['llm_pred'])==(branch=='agreement')).all())
            b={k:b[k] for k in FEATURES}
            assert torch.equal(gp[ids],b['gnn_pred']) and torch.equal(lp[ids],b['llm_pred'])
            if role=='validation' and branch=='agreement':continue
            prediction=predict_bundle(fitted,b); reference=predict_bundle(deployed,b)
            errors={}
            for k in prediction:
                np.testing.assert_allclose(prediction[k],reference[k],rtol=0,atol=1e-10)
                errors[k]=float(np.max(np.abs(prediction[k]-reference[k]))) if len(ids) else 0.
            scores[ids]=torch.as_tensor(prediction['score'],dtype=torch.float64)
            if branch=='disagreement':choose[ids]=torch.as_tensor(prediction['route'])==0
            if role=='query':accepted[ids]=torch.from_numpy(top_mask(prediction['score'],ids,fitted['policy']['fraction']))
            diagnostics.append(dict(branch=branch,role=role,rows=len(ids),refit=refit,max_errors=errors))
    result=dict(node_ids=torch.arange(n),choose_g=choose,trust=torch.where(agree,torch.zeros_like(scores),scores).float(),
                accepted=accepted,selected_pred=torch.where(choose,gp,lp),agreement=agree,selected_score=scores)
    reference=inputs.tensor(cell['inputs']['router'])
    for k,v in result.items():
        if k in reference:
            torch.testing.assert_close(v,reference[k],rtol=0,atol=1e-10 if v.dtype==torch.float64 else 0)
    return result, models, diagnostics
