# FR : CNN simple : transforme les pixels en une grille de caractéristiques.
# EN: Simple CNN: turns pixels into a grid of features.

from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # FR : Chaque convolution de stride 2 divise la hauteur et la largeur par deux.
        # EN: Each stride-2 convolution halves the height and width.
        self.layers = nn.Sequential(
            # FR : (B,3,64,64) devient (B,32,32,32).
            # EN: (B,3,64,64) becomes (B,32,32,32).
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            # FR : La deuxième grille a la forme (B,64,16,16).
            # EN: The second grid has shape (B,64,16,16).
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            # FR : La troisième grille a la forme (B,128,8,8).
            # EN: The third grid has shape (B,128,8,8).
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            # FR : Moyenne chaque zone 2x2 : sortie (B,128,4,4), donc 16 positions.
            # EN: Averages each 2x2 area: output (B,128,4,4), giving 16 positions.
            nn.AvgPool2d(2),
        )

    def forward(self, images):
        return self.layers(images)
