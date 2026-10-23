"""Targets, observation identity and label-free feature views; no expert fitting."""
import numpy as np
import torch

ENVIRONMENTS = ('clean', 'text_truncate_050', 'text_mask_mixed_030',
                'text_span_delete_030', 'graph_drop_030', 'graph_ego_mixed_050')


def targets(gnn_pred, llm_pred, labels):
    g, l, y = [torch.as_tensor(x, dtype=torch.long).reshape(-1) for x in (gnn_pred,llm_pred,labels)]
    if not (g.shape == l.shape == y.shape) or bool((torch.cat((g,l,y)) < 0).any()):
        raise ValueError('invalid predictions/labels')
    agree = g == l
    gc, lc = g == y, l == y
    dis = ~agree
    decisive = dis & (gc | lc)
    missing = torch.full_like(g, -1)
    return dict(agreement_mask=agree, disagreement_mask=dis,
                agreement_target=torch.where(agree, gc.long(), missing),
                trust_target=torch.where(dis, (gc | lc).long(), missing),
                preference_target=torch.where(decisive, gc.long(), missing),
                preference_mask=decisive,
                state=torch.where(dis, torch.where(gc,0,torch.where(lc,1,2)), missing))


def organize(dataset, environments, *, gold_ids, gold_targets, num_nodes):
    gold_ids=torch.as_tensor(gold_ids,dtype=torch.long)
    gold_targets=torch.as_tensor(gold_targets,dtype=torch.long)
    if gold_ids.ndim!=1 or gold_ids.shape!=gold_targets.shape or gold_ids.unique().numel()!=len(gold_ids):
        raise ValueError('invalid gold support')
    gold={int(n):int(y) for n,y in zip(gold_ids,gold_targets)}
    seen=set(); pieces=[]; tabs=[]; gs=[]; ls=[]
    for env in environments:
        name=env['environment']; ids=torch.as_tensor(env['node_ids'],dtype=torch.long)
        if name not in ENVIRONMENTS:raise ValueError('unregistered environment')
        keys=[(name,int(n)) for n in ids]
        if len(set(keys))!=len(keys) or seen.intersection(keys):raise ValueError('duplicate node/environment')
        seen.update(keys)
        if not set(ids.tolist()) <= set(gold) or bool(((ids<0)|(ids>=num_nodes)).any()):
            raise ValueError('every augmented parent must be a gold node')
        n=len(ids); pg=env['gnn_prob'];pl=env['llm_prob']
        if pg.shape!=pl.shape or pg.ndim!=2 or pg.shape[0]!=n or pg.shape[1]<2:
            raise ValueError('invalid expert probability shape')
        for p in (pg,pl):
            if not bool(torch.isfinite(p).all()) or bool((p<0).any()) or not torch.allclose(p.sum(1),torch.ones(n),atol=1e-5,rtol=1e-5):
                raise ValueError('invalid expert probabilities')
        for k,w in [('tab70',70),('g',128),('l',4096)]:
            if env[k].shape!=(n,w) or not bool(torch.isfinite(env[k]).all()):raise ValueError('invalid '+k)
        g=pg.argmax(1);l=pl.argmax(1);y=torch.tensor([gold[int(i)] for i in ids])
        if bool((y>=pg.shape[1]).any()):raise ValueError('gold class outside probability width')
        env_id=ENVIRONMENTS.index(name)
        row=dict(node_ids=ids,sample_id=env_id*num_nodes+ids,environment_index=torch.full((n,),env_id,dtype=torch.long),gold_target=y,gnn_pred=g,llm_pred=l,weight=torch.ones(n),**targets(g,l,y))
        pieces.append(row);tabs.append(env['tab70'].float());gs.append(env['g'].float());ls.append(env['l'].half())
    if not pieces:raise ValueError('no observations')
    rows={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
    tab,g,l=torch.cat(tabs),torch.cat(gs),torch.cat(ls)
    result=dict(dataset=dataset,environment_names=list(ENVIRONMENTS),rows=rows)
    for branch in ('agreement','disagreement'):
        mask=rows[branch+'_mask']; view=tab[mask]
        if branch=='agreement':view=torch.cat((view[:,:27],view[:,54:]),1)
        bank=dict(dataset=dataset,split='train',ids=rows['sample_id'][mask],node_ids=rows['node_ids'][mask],environment_index=rows['environment_index'][mask],tab=view,g=g[mask],l=l[mask],gnn_pred=rows['gnn_pred'][mask],llm_pred=rows['llm_pred'][mask],weight=rows['weight'][mask])
        if branch=='agreement':bank['target']=rows['agreement_target'][mask].bool()
        else:
            for k in ('state','trust_target','preference_target','preference_mask'):bank[k]=rows[k][mask]
        result[branch]=bank
    return result


def head_view(bank, head, version='v6', mode='full'):
    """Return eligible inputs/targets with sample IDs and original parent IDs separate.

    v6 exposes the historical raw tab/g/l inputs. v7 materializes the selected
    raw_features mode, before learned standardization/pruning. Model fitting is
    deliberately not performed; IDs and environment are metadata, never features.
    """
    if head=='agreement':
        y=bank['target']; mask=torch.ones(len(y),dtype=torch.bool)
    elif head in ('trust','preference'):
        state=bank['state'];mask=state!=2 if head=='preference' else torch.ones(len(state),dtype=torch.bool)
        y=(state==0) if head=='preference' else (state!=2)
    else:raise ValueError('unknown head')
    output=dict(sample_id=bank['ids'][mask],node_ids=bank['node_ids'][mask],target=y[mask].float(),weight=bank['weight'][mask])
    feature_bank={k:bank[k][mask] for k in ('tab','g','l')}
    if version=='v6':
        output.update(feature_bank)
    elif version=='v7':
        from eargtc.router_v8_v7_explore.core_v2 import raw_features
        # Match historical v7: construct a branch view before selecting head rows.
        # Sparse embedding projection reductions can depend on batch shape.
        raw=raw_features({k:bank[k] for k in ('tab','g','l')},mode)
        output['x']=torch.from_numpy(raw.astype(np.float32))[mask]
    else:raise ValueError('unknown input version')
    return output


def as_v6_bank(bank, *, gold_ids):
    """Validate gold parents, then strip audit metadata for historical model APIs.

    The old experiment-level guard assumes node IDs are unique, unaugmented gold
    IDs. New orchestration must check parent IDs here instead of changing that
    frozen guard. Equal row weighting is the native trainer's default.
    """
    parents=bank['node_ids'].tolist()
    if bank.get('split')!='train' or not set(parents)<=set(torch.as_tensor(gold_ids).tolist()):
        raise ValueError('only original gold parents may enter training')
    if len(bank['ids'])!=len(parents) or bank['ids'].unique().numel()!=len(parents):
        raise ValueError('invalid sample identity')
    if not torch.equal(bank['weight'],torch.ones(len(parents))):
        raise ValueError('native v6 adapter requires equal row weights')
    keys=('dataset','split','ids','tab','g','l','gnn_pred','llm_pred', 'state' if 'state' in bank else 'target')
    return {k:bank[k] for k in keys}


def load_head(root, dataset, head, version='v6', mode='full'):
    """Load one verified eligible head view; numerical calls require Slurm."""
    from pathlib import Path
    from .sources import require_slurm,read,sha,load
    require_slurm()
    if head not in ('agreement','trust','preference') or version not in ('v6','v7'):
        raise ValueError('unknown head/version')
    folder=Path(root)/'banks'/dataset
    meta=read(folder/'manifest.json');name=f'{head}_{version}.pt'
    if meta['dataset']!=dataset or sha(folder/name)!=meta['output_hashes'][name]:
        raise ValueError('training view identity/hash mismatch')
    value=load(folder/name)
    if not set(value['node_ids'].tolist())<=set(meta['gold_ids']):
        raise ValueError('non-gold training parent')
    n=len(value['sample_id'])
    if value['sample_id'].unique().numel()!=n or not torch.equal(value['weight'],torch.ones(n)):
        raise ValueError('invalid sample identity/weights')
    if bool(((value['target']!=0)&(value['target']!=1)).any()):
        raise ValueError('masked targets entered head training')
    if version=='v7':
        if mode not in value['features']:raise ValueError('unknown v7 mode')
        return {k:v for k,v in value.items() if k!='features'}|{'x':value['features'][mode]}
    return value
