"""Slurm-only semantic check with hand-calculated branch/coverage outcomes."""
from report_full_query import *

def main():
    preflight()
    import torch,numpy as np
    # IDs deliberately unsorted; one node has neither expert correct.
    g=np.array([0,0,0,0,0,0,0]);l=np.array([0,1,1,1,1,0,0])
    y=np.array([0,0,1,2,0,0,1])
    d=dict(ids=torch.tensor([3,1,4,2]),score=torch.tensor([.8,.4,.1,.9]),
           trust=torch.tensor([.1,.9,.7,.8]),preference=torch.tensor([.5,.9,.8,.1]),
           route=torch.tensor([0,0,1,1]),accepted=torch.tensor([False,True,False,True]))
    a=dict(ids=torch.tensor([6,5]),score=torch.tensor([.1,.2]),trust=torch.tensor([.1,.2]),
           preference=torch.tensor([float('nan'),float('nan')]),route=torch.tensor([0,0]),
           accepted=torch.tensor([False,True]))
    scores=dict(agreement=a,disagreement=d);q=np.arange(1,7)
    validate_bank(scores,q,g,l)
    r=router_metrics(scores,q,g,l,y)
    assert r['query_count']==6 and r['disagreement_count']==4 and r['agreement_count']==2
    assert r['trust_auroc']==r['preference_auroc']==r['A_correctness_auroc']==1.
    assert r['D_k_20']==1 and r['D_Prec_20']==1. and r['D_Prec_100']==.5
    assert r['A_Prec_20']==1. and r['A_Prec_100']==.5 and r['deployed_precision']==1.
    sub=np.array([1,3,4,5,6]);r=router_metrics(scores,sub,g,l,y)
    assert r['query_count']==5 and r['disagreement_count']==3 and r['D_Prec_20']==0.
    assert r['preference_auroc'] is None and r['D_Prec_100']==1/3
    # Membership at the array boundary and an empty branch must remain valid.
    r=router_metrics(scores,np.array([5,6]),g,l,y)
    assert r['disagreement_count']==0 and r['D_k_20']==0 and r['D_Prec_20'] is None
    assert r['trust_auroc'] is None and r['preference_auroc'] is None
    print('FULL_QUERY_SEMANTIC_CHECK_PASS',flush=True)

if __name__=='__main__':main()
