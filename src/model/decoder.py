# FR : Un bloc de décodeur : attention, MLP et connexions résiduelles.
# EN: One decoder block: attention, an MLP and residual connections.

from torch import nn

from src.model.attention import Attention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, n_heads)
        self.norm_mlp = nn.LayerNorm(d_model)
        # FR : Le MLP transforme chaque position séparément : D -> 4D -> D.
        # EN: The MLP transforms each position separately: D -> 4D -> D.
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask):
        # FR : Pre-norm : normalise avant l’attention, puis conserve l’entrée par addition.
        # EN: Pre-norm: normalizes before attention, then keeps the input through addition.
        x = x + self.attention(self.norm_attention(x), mask)
        # FR : Deuxième connexion résiduelle, cette fois autour du MLP.
        # EN: The second residual connection wraps the MLP.
        return x + self.mlp(self.norm_mlp(x))
