"""Seed-matched Fixed V2 and beta=1 ablation; historical code stays read-only."""
from dataclasses import asdict,replace
import itertools
import time
import torch
from eargtc.joint_cross_repair.training import TrainConfig,blend,scheme_b_probability
from eargtc.joint_cross_repair_search.bank import RECIPES,train_member,member_logits
from eargtc.joint_cross_repair_search.experiment import candidate_bank
from eargtc.joint_cross_repair_search.selection import residual_probability,choose_independent,_shortlist
from eargtc.joint_five_router.experiment import select_candidates
from .protocol import RUN,REFERENCE,read,write,save,sha,check,cell,load,training_open,verify_code
from .training import score


def no_b_selection(candidates,y):
    choices=[]
    def trial(g,l,alpha,family):
        probability=alpha*g['probability']+(1-alpha)*l['probability']
        active=int(alpha>0 and g['metadata'].get('active_repair',False))+int(alpha<1 and l['metadata'].get('active_repair',False))
        return dict(g_id=g['id'],l_id=l['id'],alpha=alpha,beta=1.,family=family,active_repairs=active,**score(probability,y))
    def allowed(c,family):
        m=c['metadata']
        if m['base']:return True
        if family=='gold_only':return m['recipe']=='gold_only' and m['criterion']=='accuracy' and m['scale']==1.
        if family=='weighted':return m['criterion']=='accuracy' and m['scale']==1.
        return True
    for family in ('gold_only','weighted','scaled'):
        g=choose_independent([c for c in candidates['g'] if allowed(c,family)],y)
        l=choose_independent([c for c in candidates['l'] if allowed(c,family)],y)
        choices.extend(trial(g,l,a,family) for a in (0.,.5,1.))
    for g,l,a in itertools.product(_shortlist(candidates['g'],y,4),_shortlist(candidates['l'],y,4),(0.,.5,1.)):
        choices.append(trial(g,l,a,'joint'))
    choices.extend(trial(candidates['g'][0],candidates['l'][0],a,'base_fusion') for a in (0.,.5,1.))
    key=lambda r:(-r['accuracy'],r['nll'],r['active_repairs'],r['family'],abs(r['alpha']-.5),r['alpha'],r['g_id'],r['l_id'])
    return {'optimized':min(choices,key=key),'alternatives':choices,'p_b_used':False}


def run_fixed(seed,ds):
    training_open();verify_code();dest=cell(seed,ds)/'fixed'
    if (dest/'complete.json').exists():check(read(dest/'complete.json')['files']);return
    f,s,p,h,split,router=load(seed,ds)
    config=replace(TrainConfig(**read(REFERENCE/'joint/CONFIG.json')['train']),seed=seed)
    members={};start=time.monotonic()
    for recipe in RECIPES:
        path=dest/'bank'/f'{recipe.name}.pt';receipt=path.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);check({str(path):r['sha256']})
            if r['training_seed']!=seed or r['recipe']!=asdict(recipe) or r['config']!=asdict(config):raise ValueError('Fixed cached identity differs')
            member=torch.load(path,weights_only=True,map_location='cpu')
        else:
            member=train_member(f,s,p,h,recipe,config);save(path,member)
            write(receipt,dict(sha256=sha(path),training_seed=seed,recipe=asdict(recipe),config=asdict(config)))
        members[seed,recipe.name]=member
    candidates=candidate_bank(members,f,s,[seed])
    pb=scheme_b_probability(f['logp_g'].exp(),f['logp_l'].exp(),f['choose_g'],f['agreement'])
    selection=select_candidates(candidates,pb[s['selection_ids']],s['selection_targets'])
    no_b=no_b_selection(candidates,s['selection_targets'])
    catalog={c['id']:c for branch in candidates.values() for c in branch};cache={}
    def full(cid):
        if cid in cache:return cache[cid]
        m=catalog[cid]['metadata'];branch=m['branch'];base=f['logp_'+branch]
        if m['base']:value=base.exp()
        else:
            z=member_logits(members[seed,m['recipe']],branch,m['criterion'],f,h,f['node_ids'],config)
            value=residual_probability(base,z,m['scale'])
        torch.testing.assert_close(value[s['selection_ids']],catalog[cid]['probability'],atol=2e-6,rtol=0)
        cache[cid]=value;return value
    def assemble(pick):
        if pick['family']=='router':return pb
        return blend(full(pick['g_id']),full(pick['l_id']),pb,pick['alpha'],pick['beta'])
    pred=assemble(selection['optimized']);ablation=assemble(no_b['optimized'])
    # Seed-matched fresh gold parity is available for seeds42–44 in the old ensemble bank.
    parity=[]
    if seed in (42,43,44):
        donor_path=REFERENCE/f'seed{seed}/joint/r9_new/{ds}/bank/inner{seed}/gold_only.pt'
        persistence=read(REFERENCE/'PERSISTENCE.json')
        check({str(donor_path):persistence['copied_files'][str(donor_path.relative_to(REFERENCE))]})
        donor=torch.load(donor_path,weights_only=True,map_location='cpu')
        for b,c in itertools.product(('g','l'),('accuracy','nll')):
            a=members[seed,'gold_only']['validation_logits'][b][c].softmax(1)
            old=donor['validation_logits'][b][c].softmax(1)
            torch.testing.assert_close(a,old,atol=2e-6,rtol=0)
            if not torch.equal(a.argmax(1),old.argmax(1)):raise ValueError('Gold parity argmax differs')
            parity.append(b+'/'+c)
    save(dest/'predictions.pt',dict(node_ids=f['node_ids'],fixed=pred,no_b=ablation,validation_ids=s['selection_ids'],
                                   training_seed=seed,inner_ensemble=False))
    write(dest/'selection.json',dict(fixed=selection,no_b=no_b,training_seed=seed,inner_seeds=[],gold_parity=parity,
                                   train_seconds=time.monotonic()-start,recipe_count=9,test_truth_loaded=False))
    files={str(q):sha(q) for q in dest.rglob('*') if q.is_file() and '.tmp.' not in q.name}
    write(dest/'complete.json',dict(files=files,training_seed=seed,inner_seeds=[],all_checks_pass=True))
    print('FIXED COMPLETE',seed,ds,selection['optimized']['accuracy'],flush=True)


def verify_saved(seed,ds,f,s,h):
    """Reconstruct Fixed selection and full predictions before the global truth gate."""
    dest=cell(seed,ds)/'fixed';cfg=replace(TrainConfig(**read(REFERENCE/'joint/CONFIG.json')['train']),seed=seed)
    members={}
    for recipe in RECIPES:
        member=torch.load(dest/'bank'/f'{recipe.name}.pt',weights_only=True,map_location='cpu')
        if member['config']!=asdict(cfg) or member['recipe']!=asdict(recipe):raise ValueError('Fixed member identity failed')
        members[seed,recipe.name]=member
    candidates=candidate_bank(members,f,s,[seed]);catalog={c['id']:c for branch in candidates.values() for c in branch}
    pb=scheme_b_probability(f['logp_g'].exp(),f['logp_l'].exp(),f['choose_g'],f['agreement'])
    expected=read(dest/'selection.json')
    sel=select_candidates(candidates,pb[s['selection_ids']],s['selection_targets'])
    nob=no_b_selection(candidates,s['selection_targets'])
    if sel!=expected['fixed'] or nob!=expected['no_b']:raise ValueError('Fixed validation selection reconstruction differs')
    predictions=torch.load(dest/'predictions.pt',weights_only=True,map_location='cpu');cache={}
    if predictions['training_seed']!=seed or predictions['inner_ensemble'] is not False or not torch.equal(predictions['node_ids'],f['node_ids']):raise ValueError('Fixed export identity differs')
    def full(cid):
        if cid in cache:return cache[cid]
        m=catalog[cid]['metadata'];branch=m['branch'];base=f['logp_'+branch]
        if m['base']:out=base.exp()
        else:
            logits=member_logits(members[seed,m['recipe']],branch,m['criterion'],f,h,f['node_ids'],cfg)
            out=residual_probability(base,logits,m['scale'])
        cache[cid]=out;return out
    for name,selection in [('fixed',sel),('no_b',nob)]:
        pick=selection['optimized']
        out=pb if pick['family']=='router' else blend(full(pick['g_id']),full(pick['l_id']),pb,pick['alpha'],pick['beta'])
        saved=predictions[name]
        if not torch.isfinite(saved).all() or bool((saved<0).any()):raise ValueError('Invalid Fixed probabilities')
        torch.testing.assert_close(saved.sum(1),torch.ones(len(saved)),atol=2e-6,rtol=0)
        torch.testing.assert_close(saved,out,atol=2e-6,rtol=0)
        if not torch.equal(saved.argmax(1),out.argmax(1)):raise ValueError('Fixed argmax cannot be reconstructed')
        metrics=score(saved[s['selection_ids']],s['selection_targets'])
        if abs(metrics['accuracy']-pick['accuracy'])>1e-12 or abs(metrics['nll']-pick['nll'])>2e-6:raise ValueError('Fixed validation parity failed')
    return True
