from torch import nn

from src.model.attention import Attention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, n_heads)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask):
        x = x + self.attention(self.norm_attention(x), mask)
        return x + self.mlp(self.norm_mlp(x))
