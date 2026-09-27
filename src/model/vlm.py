import torch
from torch import nn

from src.model.decoder import DecoderBlock
from src.model.encoder import Encoder


class TinyVLM(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.n_visual = 16
        self.encoder = Encoder()
        self.adapter = nn.Linear(128, d_model)
        self.letters = nn.Embedding(27, d_model)
        self.positions = nn.Embedding(self.n_visual + 45, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 27)

    def visual_tokens(self, images):
        features = self.encoder(images)
        return self.adapter(features.flatten(2).transpose(1, 2))

    def decode_tokens(self, visual, input_ids):
        x = torch.cat((visual, self.letters(input_ids)), dim=1)
        length = x.shape[1]
        x = x + self.positions(torch.arange(length, device=x.device))
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        mask[:self.n_visual, :self.n_visual] = True
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x[:, self.n_visual - 1:]))

    def forward(self, images, input_ids):
        return self.decode_tokens(self.visual_tokens(images), input_ids)
