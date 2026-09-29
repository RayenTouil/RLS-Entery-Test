# FR : Attention multi-têtes écrite avec les opérations de base de PyTorch.
# EN: Multi-head attention written with basic PyTorch operations.

import math

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # FR : Les têtes partagent la dimension en parts égales : 128 / 4 = 32.
        # EN: Heads split the dimension equally: 128 / 4 = 32.
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # FR : Q cherche une information ; K sert à comparer ; V contient ce qui sera mélangé.
        # EN: Q asks for information; K is used for matching; V holds the content to mix.
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        # FR : (B,T,D) devient (B,H,T,D/H), avec H têtes.
        # EN: (B,T,D) becomes (B,H,T,D/H), with H heads.
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))
        # FR : Chaque query est comparée à chaque key : scores (B,H,T,T).
        # EN: Each query is compared with each key: scores have shape (B,H,T,T).
        # FR : sqrt(d_head) évite que les scores grossissent trop avec la dimension.
        # EN: sqrt(d_head) keeps scores from growing too much with the head dimension.
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # FR : True autorise la lecture ; -inf donne un poids nul après softmax.
        # EN: True allows attention; -inf gives zero weight after softmax.
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        # FR : Normalise sur les clés : une distribution de poids pour chaque query.
        # EN: Normalizes over keys: one weight distribution for each query.
        weights = torch.softmax(scores, dim=-1)
        # FR : Calcule une moyenne pondérée des values pour chaque tête.
        # EN: Computes a weighted average of the values for each head.
        context = weights @ value
        # FR : Réunit les têtes en (B,T,D), puis applique la projection de sortie.
        # EN: Joins the heads into (B,T,D), then applies the output projection.
        context = context.transpose(1, 2).reshape(x.shape)
        return self.output(context)
