"""Independent candidate-shared, exchange-equivariant Router V8."""
import torch
from torch import nn


def event_probabilities(trust_logit, preference_logit):
    c, r = trust_logit.sigmoid(), preference_logit.sigmoid()
    return torch.stack((c*r, c*(1-r), 1-c), dim=-1)


class TrustPreference(nn.Module):
    def __init__(self, candidate_dim, global_dim, width=64, dropout=.1, architecture='mlp'):
        super().__init__()
        if architecture == 'linear':
            self.encoder = nn.Identity()
            h = candidate_dim
            self.trust = nn.Linear(2*h+global_dim, 1)
        elif architecture == 'mlp':
            h = width // 2
            self.encoder = nn.Sequential(nn.Linear(candidate_dim,width), nn.LayerNorm(width),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(width,h), nn.GELU())
            self.trust = nn.Sequential(nn.Linear(2*h+global_dim,width),nn.GELU(),
                                       nn.Dropout(dropout),nn.Linear(width,1))
        else:
            raise ValueError(architecture)
        self.scorer = nn.Linear(h,1,bias=False)

    def forward(self, candidate_g, candidate_l, symmetric_global):
        g, l = self.encoder(candidate_g), self.encoder(candidate_l)
        preference = self.scorer(g-l).squeeze(-1)
        trust = self.trust(torch.cat((g+l,(g-l).abs(),symmetric_global),dim=-1)).squeeze(-1)
        return trust, preference
