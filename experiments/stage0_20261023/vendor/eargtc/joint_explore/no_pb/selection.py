"""Fixed-V2 selection with beta=1 and no Router output candidate.

Uses the frozen candidate validation, shortlist, and tie-breaking helpers.
There is deliberately no p_B argument to this selector or its fusion function.
"""
from eargtc.joint_cross_repair_search.selection import (
    ALPHAS, _candidates, _shortlist, _score, _active, _trial_key, choose_independent,
)


def fuse(g, l, alpha):
    return float(alpha)*g + (1.-float(alpha))*l


def choose_pair(gs, ls, y):
    trials=[]
    for g in gs:
        for l in ls:
            for alpha in ALPHAS:
                accuracy,nll=_score(fuse(g['probability'],l['probability'],alpha),y)
                trials.append(dict(g_id=g['id'],l_id=l['id'],alpha=float(alpha),beta=1.,
                                   accuracy=accuracy,nll=nll,active_repairs=_active(g,l,alpha,1.)))
    return {**min(trials,key=_trial_key),'trial_count':len(trials)}


def select_without_pb(candidates, y):
    g,y=_candidates(candidates['g'],y,'g')
    l,_=_candidates(candidates['l'],y,'l')
    if g[0]['probability'].shape != l[0]['probability'].shape:
        raise ValueError('Branch shapes differ')
    checked={'g':g,'l':l}
    def subset(branch,family):
        def allowed(c):
            m=c['metadata']
            if m['base']: return True
            if family=='gold_only': return m['recipe']=='gold_only' and m['criterion']=='accuracy' and m['scale']==1.
            if family=='weighted': return m['criterion']=='accuracy' and m['scale']==1.
            return True
        return [c for c in checked[branch] if allowed(c)]
    selected={}
    for family in ('gold_only','weighted','scaled'):
        gs=choose_independent(subset('g',family),y)
        ls=choose_independent(subset('l',family),y)
        selected[family]=choose_pair([gs],[ls],y)
    gs,ls=_shortlist(g,y,4),_shortlist(l,y,4)
    selected['joint']={**choose_pair(gs,ls,y),'g_shortlist_ids':[c['id'] for c in gs],
                       'l_shortlist_ids':[c['id'] for c in ls]}
    base=choose_pair([c for c in g if c['metadata']['base']],
                     [c for c in l if c['metadata']['base']],y)
    alternatives=[{'family':k,**v} for k,v in selected.items()]
    alternatives.append({'family':'base_fusion',**base})
    winner=min(alternatives,key=lambda r:(-r['accuracy'],r['nll'],r['active_repairs'],r['family']))
    return dict(families=selected,base_fusion=base,optimized=winner,alternatives=alternatives,
                selection_population='historical_validation_calibration_half',seed_reduction='mean_probability')
