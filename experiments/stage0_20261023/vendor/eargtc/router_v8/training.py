"""Source-only transforms and reproducible independent model training."""
import copy
import numpy as np
import torch
from .model import TrustPreference, event_probabilities
from .losses import factorized_loss, loss_weights


class EvidenceTransform:
    def __init__(self,view,state):
        self.view,self.state=view,state

    @classmethod
    def fit(cls,banks,view):
        state={}
        for prefix in ('candidate','global'):
            xs=[torch.cat((b['candidate_g'],b['candidate_l'])) if prefix=='candidate'
                else b['symmetric_global'] for b in banks]
            if not xs or any(len(x)==0 for x in xs):
                raise ValueError('transform requires nonempty source domains')
            if view=='raw':
                # Equal source-domain mass; both candidates share identical transform.
                mean=torch.stack([x.mean(0) for x in xs]).mean(0)
                var=torch.stack([((x-mean)**2).mean(0) for x in xs]).mean(0)
                state[prefix+'_mean']=mean; state[prefix+'_scale']=var.sqrt().clamp_min(.01)
            elif view=='rank':
                for i,x in enumerate(xs):
                    state[f'{prefix}_reference_{i}']=x.sort(dim=0).values.contiguous()
            else:
                raise ValueError(view)
        return cls(view,state)

    def state_dict(self):
        return {k:v.clone() for k,v in self.state.items()}

    def _apply(self,x,prefix):
        if self.view=='raw':
            return ((x-self.state[prefix+'_mean'])/self.state[prefix+'_scale']).clamp(-12,12)
        refs=[v for k,v in self.state.items() if k.startswith(prefix+'_reference_')]
        outputs=[]
        for ref in refs:
            cols=[]
            for j in range(x.shape[1]):
                col=ref[:,j].contiguous(); val=x[:,j].contiguous()
                lo=torch.searchsorted(col,val,right=False); hi=torch.searchsorted(col,val,right=True)
                cols.append((lo+hi).float()/(2*len(col)))
            outputs.append(torch.stack(cols,dim=1))
        return (torch.stack(outputs).mean(0)-.5)*2

    def apply(self,bank):
        return {**bank,'candidate_g':self._apply(bank['candidate_g'],'candidate'),
                'candidate_l':self._apply(bank['candidate_l'],'candidate'),
                'symmetric_global':self._apply(bank['symmetric_global'],'global')}


def select_features(bank,group):
    return {**bank,'candidate_g':bank['candidate_g'][:,group['candidate']],
            'candidate_l':bank['candidate_l'][:,group['candidate']],
            'symmetric_global':bank['symmetric_global'][:,group['global']]}


def predict_logits(model,bank,batch_size=8192):
    model.eval(); device=next(model.parameters()).device
    cs,rs=[],[]
    with torch.no_grad():
        for start in range(0,len(bank['node_ids']),batch_size):
            x=[bank[k][start:start+batch_size].to(device) for k in ('candidate_g','candidate_l','symmetric_global')]
            c,r=model(*x); cs.append(c.cpu()); rs.append(r.cpu())
    return (torch.cat(cs).numpy(),torch.cat(rs).numpy()) if cs else (np.zeros(0),np.zeros(0))


def fit_model(banks,config,*,epochs=None,snapshots=None,device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    transform=EvidenceTransform.fit(banks,config['view'])
    transformed=[transform.apply(b) for b in banks]
    xs=[torch.cat([b[k] for b in transformed]).to(device) for k in ('candidate_g','candidate_l','symmetric_global')]
    state=torch.cat([b['state'] for b in transformed]).long().to(device)
    domain=torch.cat([torch.full((len(b['state']),),i,dtype=torch.long) for i,b in enumerate(banks)]).to(device)
    model=TrustPreference(xs[0].shape[1],xs[2].shape[1],width=config['width'],
                          dropout=config['dropout'],architecture=config.get('architecture','mlp')).to(device)
    weights=loss_weights(state,domain,config['treatment'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.003),weight_decay=config['weight_decay'])
    endpoint_epochs=sorted(snapshots or [epochs or config.get('epochs',240)])
    saved={}
    for epoch in range(1,max(endpoint_epochs)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        c,r=model(*xs)
        loss=factorized_loss(c,r,state,weights=weights)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite source loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if epoch in endpoint_epochs:
            saved[epoch]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.eval()
    return model,transform,saved
