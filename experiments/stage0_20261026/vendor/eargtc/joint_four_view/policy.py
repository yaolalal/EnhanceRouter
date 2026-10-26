"""Finite label-free four-view policies and cross-validated policy selection.

Only statistics/selection functions consume validation targets. No gate is fitted
on gold in-sample predictions or on validation gradients.
"""
import hashlib
import itertools
import math
import torch

FAMILIES = ('base', 'static', 'confidence', 'agreement_guard')


def make_folds(node_ids, count=5):
    ids = torch.as_tensor(node_ids, dtype=torch.long)
    if ids.ndim != 1 or len(ids) < count or len(ids.unique()) != len(ids):
        raise ValueError('folds require distinct node IDs and at least one per fold')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f'joint-four-view-v3:{int(ids[i])}'.encode()).digest())
    result = torch.empty(len(ids), dtype=torch.long)
    result[torch.tensor(order)] = torch.arange(len(ids)) % count
    return result


def make_policies(pair_ids):
    records = []
    def add(family, **kw):
        records.append({'id':f'{family}/{len(records):05d}', 'family':family, **kw})
    # Coarse original-expert/Router controls, independent of previous validation picks.
    seen = set()
    for alpha, beta in itertools.product((0., .5, 1.), (0., .25, .5, 1.)):
        key = (alpha*beta, (1-alpha)*beta, 1-beta)
        if key in seen: continue
        seen.add(key)
        add('base', g_id=None, l_id=None, weights=[alpha,1-alpha,0.,0.],router_weight=1-beta)
    weights = [list(x/4 for x in w) for w in itertools.product(range(5), repeat=4) if sum(w)==4]
    priors = ([1,1,1,1],[2,2,1,1],[1,1,2,2],[1,0,1,0],[0,1,0,1])
    for g_id, l_id in pair_ids:
        for w, rho in itertools.product(weights, (0., .5)):
            add('static',g_id=g_id,l_id=l_id,weights=w,router_weight=rho)
        for prior, score, power, rho in itertools.product(priors, ('log_max','neg_entropy'), (1.,4.,16.), (0.,.5)):
            for family in ('confidence','agreement_guard'):
                add(family,g_id=g_id,l_id=l_id,prior=prior,score=score,power=power,router_weight=rho)
    return records


def _views(record, branches):
    return torch.stack([branches['g0'],branches['l0'],
        branches[record['g_id']] if record['g_id'] else branches['g0'],
        branches[record['l_id']] if record['l_id'] else branches['l0']])


def _scores(views):
    p = views.clamp_min(1e-12)
    return {'log_max':p.amax(2).log(),
            'neg_entropy':(views*p.log()).sum(2)/math.log(views.shape[2])}


def _apply(record, views, pb, agreement, scores=None):
    if 'weights' in record:
        w = views.new_tensor(record['weights'])[:,None,None]
    else:
        if scores is None: scores = _scores(views)
        prior = views.new_tensor(record['prior'])[:,None]
        logw = scores[record['score']]*record['power'] + prior.log()
        w = logw.softmax(0).unsqueeze(2)
    p = (w*views).sum(0)
    rho = record['router_weight']
    p = (1-rho)*p + rho*pb
    if record['family']=='agreement_guard': p = torch.where(agreement[:,None],pb,p)
    return p


def apply_policy(record, branches, pb, agreement):
    """Produce node probabilities using frozen predictions only."""
    return _apply(record,_views(record,branches),pb,agreement)


@torch.no_grad()
def collect_statistics(records, branches, pb, agreement, targets, folds, progress=None):
    """Store only per-fold sufficient statistics, not all full predictions."""
    n = len(targets)
    if pb.ndim!=2 or pb.shape[0]!=n or targets.shape!=(n,) or folds.shape!=(n,):
        raise ValueError('validation inputs differ')
    if set(folds.tolist()) != set(range(5)): raise ValueError('five nonempty folds required')
    for p in [pb,*branches.values()]:
        if p.shape!=pb.shape or not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(1),torch.ones(n),atol=2e-5):
            raise ValueError('invalid candidate probability')
    counts=torch.bincount(folds,minlength=5)
    correct=torch.zeros(len(records),5,dtype=torch.long)
    nll=torch.zeros(len(records),5,dtype=torch.float64)
    masks=[folds==k for k in range(5)]
    row=torch.arange(n);last=None;views=None;scores=None
    for i,r in enumerate(records):
        pair=(r['g_id'],r['l_id'])
        if pair!=last or views is None:
            views=_views(r,branches);scores=_scores(views);last=pair
        p=_apply(r,views,pb,agreement,scores)
        hit=p.argmax(1)==targets
        loss=-p[row,targets].double().clamp_min(1e-12).log()
        for k,mask in enumerate(masks):
            correct[i,k]=hit[mask].sum();nll[i,k]=loss[mask].sum()
        if progress and (i+1)%500==0:progress(i+1,len(records))
    return {'correct':correct,'nll_sum':nll,'counts':counts,'folds':folds}


def choose_from_statistics(records, correct, nll_sum, counts):
    if correct.shape != nll_sum.shape or correct.shape != (len(records),5) or counts.shape!=(5,) or bool((counts<=0).any()):
        raise ValueError('invalid fold statistics')
    if not bool(torch.isfinite(nll_sum).all()) or bool((correct<0).any()) or bool((correct>counts).any()):
        raise ValueError('invalid candidate scores')
    totals=correct.sum(1).tolist(); losses=nll_sum.sum(1).tolist();total=int(counts.sum())
    def best(indices, excluded=None):
        if excluded is None:
            return min(indices,key=lambda i:(-totals[i],losses[i],records[i]['id']))
        return min(indices,key=lambda i:(-(totals[i]-int(correct[i,excluded])),
            losses[i]-float(nll_sum[i,excluded]),records[i]['id']))
    def full_pick(i):
        return {'candidate':records[i], 'accuracy':totals[i]/total,'nll':losses[i]/total}
    families={}
    for family in FAMILIES:
        indices=[i for i,r in enumerate(records) if r['family']==family]
        if not indices:continue
        heldout=[best(indices,k) for k in range(5)]
        chosen=best(indices)
        fold_correct=[int(correct[i,k]) for k,i in enumerate(heldout)]
        fold_loss=[float(nll_sum[i,k]) for k,i in enumerate(heldout)]
        families[family]={**full_pick(chosen),'candidate_count':len(indices),
            'fold_candidate_ids':[records[i]['id'] for i in heldout],
            'fold_correct':fold_correct,'fold_nll_sum':fold_loss,
            'oof_accuracy':sum(fold_correct)/total,'oof_nll':sum(fold_loss)/total}
    winner=min(families,key=lambda f:(-families[f]['oof_accuracy'],families[f]['oof_nll'],FAMILIES.index(f)))
    direct=best(list(range(len(records))))
    return {'primary':{'family':winner,**families[winner]},
            'direct':{'family':records[direct]['family'],**full_pick(direct)},'families':families,
            'fold_counts':counts.tolist(), 'selection_population':'historical_validation_calibration_half',
            'scope':'selector-only CV conditional on frozen upstream; no validation gradients',
            'rule':'family by five-fold heldout accuracy then NLL; final candidate fit on all selection labels'}
