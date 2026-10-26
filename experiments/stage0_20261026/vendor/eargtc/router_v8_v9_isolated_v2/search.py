"""Independent single-version, single-branch recipe and threshold search."""
import numpy as np
from .core import dis_grid,dis_scores,threshold_trials,select

def recipe_grid(version,branch,source):
    if version not in ('v6','v7_strict') or branch not in ('disagreement','agreement'):raise ValueError('invalid identity')
    recipes={}
    if branch=='agreement':
        for i,cid in enumerate(source['order']['agreement']):recipes[cid]=dict(model=cid,complexity=float(i),control=i==0)
    else:
        if version=='v6':recipes['original']=dict(original=True,complexity=0.,control=True)
        for it,tid in enumerate(source['order']['trust']):
            for ip,pid in enumerate(source['order']['preference']):
                for cfg in dis_grid():
                    rid=f'{tid}|{pid}|{cfg["ts"]}|{cfg["tb"]}|{cfg["ps"]}|{cfg["pb"]}'
                    identity=cfg==dict(ts=1.,tb=0.,ps=1.,pb=0.)
                    recipes[rid]=dict(trust=tid,preference=pid,**cfg,complexity=float(it+ip+sum(abs(cfg[k]-v) for k,v in dict(ts=1.,tb=0.,ps=1.,pb=0.).items())),control=version=='v7_strict' and it==ip==0 and identity)
    return recipes

def score_recipe(branch,source,recipe):
    if branch=='agreement':return dict(score=np.asarray(source['agreement'][recipe['model']],float))
    if recipe.get('original'):return {k:np.asarray(source['original'][k]) for k in ('score','route','q')}
    return dis_scores(source['trust'][recipe['trust']],source['preference'][recipe['preference']],recipe)

def search_branch(version,branch,source,target,half):
    recipes=recipe_grid(version,branch,source);rows=[]
    for rid,recipe in recipes.items():
        output=score_recipe(branch,source,recipe)
        correct=output['route']==np.asarray(target) if branch=='disagreement' else np.asarray(target,bool)
        rows.extend(threshold_trials(rid,output['score'],correct,half,recipe['complexity']))
    selected=dict(threshold_only=select([r for r in rows if recipes[r['recipe']]['control']]),tuned=select(rows))
    return recipes,rows,selected
