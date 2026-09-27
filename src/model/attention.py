import math

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ value
        context = context.transpose(1, 2).reshape(x.shape)
        return self.output(context)
