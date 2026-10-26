"""A single classifier: two independent reductions, concatenation, logits."""
import torch
from torch import nn
from .protocol import slurm


class Fusion(nn.Module):
    def __init__(self, graph_dim, text_dim, classes, graph_width=64, text_width=64, hidden=64, dropout=.1):
        super().__init__()
        slurm()
        if not 0 < graph_width < graph_dim or not 0 < text_width < text_dim:
            raise ValueError('Projection must reduce both input dimensions')
        if classes < 2 or hidden < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid classifier dimensions/dropout')
        self.graph_dim, self.text_dim = graph_dim, text_dim
        def projection(dim,width):
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.LayerNorm(width))
        self.graph = projection(graph_dim,graph_width)
        self.text = projection(text_dim,text_width)
        self.classifier = nn.Sequential(nn.Linear(graph_width + text_width, hidden), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, graph, text):
        if graph.ndim != 2 or text.ndim != 2 or graph.shape[0] != text.shape[0]:
            raise ValueError('Embeddings must be aligned matrices')
        if graph.shape[1] != self.graph_dim or text.shape[1] != self.text_dim:
            raise ValueError('Embedding widths changed')
        return self.classifier(torch.cat((self.graph(graph.detach().float()),
                                          self.text(text.detach().float())), dim=1))
