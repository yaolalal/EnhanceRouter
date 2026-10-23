from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNNEncoder


@dataclass
class SSLStats:
    final_loss: float
    final_feature_loss: float
    final_edge_loss: float


class GraphSSLModel(nn.Module):
    def __init__(self, encoder: GNNEncoder, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, feature_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.feature_decoder(z)


def _negative_edges(num_nodes: int, edge_index: torch.Tensor, num_samples: int) -> torch.Tensor:
    device = edge_index.device
    src = torch.randint(0, num_nodes, (num_samples,), device=device)
    dst = torch.randint(0, num_nodes, (num_samples,), device=device)
    return torch.stack([src, dst], dim=0)


def _edge_logits(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def pretrain_ssl_gnn(
    data,
    hidden_dim: int,
    embedding_dim: int,
    n_layers: int,
    gnn_type: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    feature_mask_rate: float = 0.2,
    edge_loss_weight: float = 0.2,
):
    encoder = GNNEncoder(
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        gnn_type=gnn_type,
        dropout=0.3,
    ).to(data.x.device)
    model = GraphSSLModel(encoder, embedding_dim, data.x.size(1)).to(data.x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_edges = data.edge_index
    max_pos = min(pos_edges.size(1), 20000)
    final_feature_loss = torch.tensor(0.0, device=data.x.device)
    final_edge_loss = torch.tensor(0.0, device=data.x.device)

    for _ in range(max(epochs, 0)):
        model.train()
        optimizer.zero_grad()
        x_corrupt = data.x.clone()
        if feature_mask_rate > 0:
            mask = torch.rand_like(x_corrupt) < feature_mask_rate
            x_corrupt = x_corrupt.masked_fill(mask, 0.0)
        z = model.encode(x_corrupt, data.edge_index)
        recon = model.decode_features(z)
        final_feature_loss = F.mse_loss(recon, data.x)

        perm = torch.randperm(pos_edges.size(1), device=data.x.device)[:max_pos]
        pos = pos_edges[:, perm]
        neg = _negative_edges(data.num_nodes, pos_edges, pos.size(1))
        edge_logits = torch.cat([_edge_logits(z, pos), _edge_logits(z, neg)])
        edge_labels = torch.cat(
            [torch.ones(pos.size(1), device=data.x.device), torch.zeros(neg.size(1), device=data.x.device)]
        )
        final_edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)
        loss = final_feature_loss + edge_loss_weight * final_edge_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    stats = SSLStats(
        final_loss=float((final_feature_loss + edge_loss_weight * final_edge_loss).detach().cpu().item()),
        final_feature_loss=float(final_feature_loss.detach().cpu().item()),
        final_edge_loss=float(final_edge_loss.detach().cpu().item()),
    )
    return model.encoder, z.detach(), stats
