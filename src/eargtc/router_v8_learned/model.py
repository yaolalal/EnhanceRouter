"""Independent binary heads with trainable raw-embedding bottlenecks.

Only tab/g/l are inputs. ``state`` is consumed exclusively by training.py.
Fitted artifacts contain CPU tensors and can be restored without training data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PROBABILITY_INDICES = tuple(range(16)) + tuple(range(27, 43)) + tuple(range(54, 63))
FEATURE_NAMES = ([f'candidate_g_{i:02d}' for i in range(27)]
                 + [f'candidate_l_{i:02d}' for i in range(27)]
                 + [f'symmetric_global_{i:02d}' for i in range(16)])


def neural_grid():
    configs = []
    for features in ('prob', 'all', 'gated'):
        for embeddings in ('none', 'gnn', 'llm', 'both'):
            for augmented in (False, True):
                recipe = 'aug' if augmented else 'plain'
                configs.append(dict(
                    id=f'neural_{features}_{embeddings}_{recipe}',
                    feature_mode=features, embedding_mode=embeddings,
                    width=64 if augmented else 32, dropout=.3 if augmented else .1,
                    weight_decay=.1 if augmented else .01,
                    mixup=.2 if augmented else 0., ranking=.2 if augmented else 0.,
                    consistency=.1 if augmented else 0., lr=.001,
                    source_steps=200, adapt_steps=80, seed=42,
                    gate_penalty=.001, batch_size=256))
    return configs


class LearnedHead(nn.Module):
    def __init__(self, config, transform):
        super().__init__()
        self.register_buffer('feature_mask', transform['mask'].float().clone())
        self.gate_logits = (nn.Parameter(torch.zeros(70))
                            if config['feature_mode'] == 'gated' else None)
        width = int(config['width'])
        bottleneck = width // 2
        mode = config['embedding_mode']
        self.gnn = (nn.Sequential(nn.LayerNorm(transform['g_dim']),
                                 nn.Linear(transform['g_dim'], bottleneck), nn.GELU())
                    if mode in ('gnn', 'both') else None)
        self.llm = (nn.Sequential(nn.LayerNorm(transform['l_dim']),
                                 nn.Linear(transform['l_dim'], bottleneck), nn.GELU())
                    if mode in ('llm', 'both') else None)
        inputs = 70 + bottleneck * (int(self.gnn is not None) + int(self.llm is not None))
        self.fusion = nn.Sequential(nn.Linear(inputs, width), nn.GELU(),
                                    nn.Dropout(config['dropout']), nn.Linear(width, 1))

    def forward(self, tab, g, l):
        gates = self.feature_mask
        if self.gate_logits is not None:
            gates = gates * self.gate_logits.sigmoid()
        pieces = [tab * gates]
        if self.gnn is not None:
            pieces.append(self.gnn(g))
        if self.llm is not None:
            pieces.append(self.llm(l))
        return self.fusion(torch.cat(pieces, dim=1)).squeeze(-1)


def _restore(fitted, device):
    model = LearnedHead(fitted['config'], fitted['transform']).to(device)
    model.load_state_dict(fitted['state_dict'])
    return model


def _features(bank, index, transform, device):
    # Keep full banks on CPU; only bounded minibatches move onto the accelerator.
    values = [torch.as_tensor(bank[key])[index].to(device=device, dtype=torch.float32)
              for key in ('tab', 'g', 'l')]
    mean = transform['mean'].to(device)
    scale = transform['scale'].to(device)
    values[0] = ((values[0] - mean) / scale).clamp(-20, 20)
    return values


def predict_head(fitted, bank, device='cpu', batch_size=1024):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    model = _restore(fitted, device)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(bank['ids']), batch_size):
            x = _features(bank, slice(start, start + batch_size), fitted['transform'], device)
            outputs.append(model(*x).cpu().double().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)


def feature_importance(fitted):
    state = fitted['state_dict']
    mask = fitted['transform']['mask'].bool()
    gates = (state['gate_logits'].sigmoid() if 'gate_logits' in state else torch.ones(70))
    weights = {}
    for family, key in (('gnn_projection', 'gnn.1.weight'), ('llm_projection', 'llm.1.weight')):
        weights[family] = float(state[key].norm()) if key in state else 0.
    return dict(feature_names=list(FEATURE_NAMES), mask=mask.tolist(),
                gates=(gates * mask).tolist(), projection_weight_norms=weights,
                gradient_norms=dict(fitted['training_audit'].get('gradient_norms', {})))


def fit_head(banks, head, config, device='cpu', initial=None, steps=None, unlabeled=None):
    from .training import train_head
    return train_head(banks, head, config, device, initial, steps, unlabeled)
