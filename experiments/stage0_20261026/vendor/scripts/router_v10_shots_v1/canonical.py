"""Exact historical label-free Joint handoff functions, isolated import context."""
from common import torch, require as slurm

def verify_router(query,disagreement,router,split):
    ids=query['node_ids'];did=disagreement['node_ids'];route=disagreement['route']
    if not torch.equal(ids,ids.sort().values):raise ValueError('Expected sorted deployment IDs')
    protected=torch.tensor(split['support_ids']+split['valid_ids'])
    agree=query['gnn_pred']==query['llm_pred']
    checks=[torch.equal(router['node_ids'],torch.arange(split['num_nodes'])),
            torch.equal(ids.sort().values,torch.tensor(split['unlabeled_ids']).sort().values),
            not bool(router['accepted'][protected].any()),
            torch.equal(router['accepted'][ids],query['accepted_mask']),
            torch.equal(torch.where(router['accepted'][ids],router['selected_pred'][ids],-1),query['pseudo_label']),
            router['selected_score'].dtype==query['selected_score'].dtype==torch.float64,
            torch.equal(router['selected_score'][ids],query['selected_score']),
            torch.equal(router['agreement'][ids],agree),
            torch.equal(did.sort().values,ids[~agree].sort().values),
            torch.equal(router['choose_g'][did],route==0)]
    pos=torch.searchsorted(ids,did)
    checks.extend([torch.equal(ids[pos],did),bool(((route==0)|(route==1)).all()),
                   torch.equal(router['selected_pred'][did],torch.where(route==0,query['gnn_pred'][pos],query['llm_pred'][pos])),
                   torch.equal(router['selected_pred'][ids[agree]],query['gnn_pred'][agree]),
                   torch.equal(router['trust'][ids],torch.where(agree,torch.zeros_like(query['selected_score']),query['selected_score']).float())])
    if not all(checks):raise ValueError('Frozen deployment handoff parity failed')

def canonical_router(query,disagreement,upstream,split,validation=None):
    """Convert frozen query decisions, and explicit validation routes, to safe full-node inputs."""
    slurm();n=split['num_nodes'];ids=query['node_ids']
    gp=upstream['gnn_logits'].argmax(1);lp=upstream['llm_logits'].argmax(1)
    if not torch.equal(gp[ids],query['gnn_pred']) or not torch.equal(lp[ids],query['llm_pred']):
        raise ValueError('Router experts differ from historical Joint experts')
    agree=gp==lp;choose=torch.ones(n,dtype=torch.bool)
    choose[disagreement['node_ids']]=disagreement['route']==0
    scores=torch.zeros(n,dtype=torch.float64);scores[ids]=query['selected_score']
    if validation is not None:
        v=validation['node_ids'];route=validation['route'];sc=validation['score']
        expected=torch.tensor(split['valid_ids']);expected=expected[~agree[expected]]
        if not torch.equal(v.sort().values,expected.sort().values) or not bool(((route==0)|(route==1)).all()):
            raise ValueError('Validation disagreement routing identity failed')
        choose[v]=route==0;scores[v]=sc
    accepted=torch.zeros(n,dtype=torch.bool);accepted[ids]=query['accepted_mask']
    r=dict(node_ids=torch.arange(n),choose_g=choose,trust=torch.where(agree,torch.zeros_like(scores),scores).float(),
        accepted=accepted,selected_pred=torch.where(choose,gp,lp),agreement=agree,selected_score=scores)
    verify_router(query,disagreement,r,split)
    return r
