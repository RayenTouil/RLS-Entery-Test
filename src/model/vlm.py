# FR : Assemble le CNN et le décodeur pour prédire la prochaine lettre.
# EN: Combines the CNN and decoder to predict the next letter.

import torch
from torch import nn

from src.model.decoder import DecoderBlock
from src.model.encoder import Encoder


class TinyVLM(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        # FR : La grille 4x4 du CNN fournit 16 tokens visuels.
        # EN: The CNN 4x4 grid provides 16 visual tokens.
        self.n_visual = 16
        self.encoder = Encoder()
        # FR : Projette les caractéristiques visuelles dans la dimension du décodeur.
        # EN: Projects visual features into the decoder dimension.
        self.adapter = nn.Linear(128, d_model)
        # FR : Chaque id de lettre reçoit un vecteur appris.
        # EN: Each letter id gets a learned vector.
        self.letters = nn.Embedding(27, d_model)
        # FR : 16 positions visuelles + au plus 45 positions de lettres.
        # EN: 16 visual positions + at most 45 letter positions.
        self.positions = nn.Embedding(self.n_visual + 45, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 27)

    def visual_tokens(self, images):
        features = self.encoder(images)
        # FR : (B,128,4,4) -> (B,128,16) -> (B,16,128) -> (B,16,D).
        # EN: (B,128,4,4) -> (B,128,16) -> (B,16,128) -> (B,16,D).
        return self.adapter(features.flatten(2).transpose(1, 2))

    def decode_tokens(self, visual, input_ids):
        # FR : Place l’image avant les lettres : (B,16+T,D).
        # EN: Places the image before the letters: (B,16+T,D).
        x = torch.cat((visual, self.letters(input_ids)), dim=1)
        length = x.shape[1]
        # FR : Ajoute une position apprise pour distinguer l’ordre des tokens.
        # EN: Adds a learned position to distinguish token order.
        x = x + self.positions(torch.arange(length, device=x.device))
        # FR : Le triangle inférieur empêche les lettres de lire le futur.
        # EN: The lower triangle prevents letters from reading the future.
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        # FR : Les tokens visuels se voient tous, mais ne voient aucune lettre.
        # EN: Visual tokens see each other, but cannot see any letters.
        mask[:self.n_visual, :self.n_visual] = True
        for block in self.blocks:
            x = block(x, mask)
        # FR : Le dernier visuel prédit la première lettre : T entrées donnent T+1 sorties.
        # EN: The last visual token predicts the first letter: T inputs give T+1 outputs.
        return self.head(self.norm(x[:, self.n_visual - 1:]))

    def forward(self, images, input_ids):
        return self.decode_tokens(self.visual_tokens(images), input_ids)
