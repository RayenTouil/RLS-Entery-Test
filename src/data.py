# FR : Charge les images et construit des lots image/texte.
# EN: Loads the images and builds image/text batches.

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import Tokenizer


class ShapeScenes(Dataset):
    def __init__(self, data_dir="data", split="train", blind=False):
        # FR : Charge les tenseurs enregistrés par le générateur officiel.
        # EN: Loads the tensors saved by the official generator.
        data = torch.load(Path(data_dir) / f"{split}.pt", weights_only=True)
        self.images = data["images"]
        self.words = data["words"]
        # FR : Encode les mots une seule fois lors du chargement.
        # EN: Encodes the words once when loading the dataset.
        self.ids = [Tokenizer().encode(word) for word in self.words]
        self.blind = blind

    # FR : Indique au DataLoader combien d’exemples sont disponibles.
    # EN: Tells the DataLoader how many samples are available.
    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        # FR : Convertit les pixels uint8 [0,255] en float32 [0,1].
        # EN: Converts uint8 pixels [0,255] to float32 [0,1].
        image = self.images[index].float().div(255)
        # FR : Pour E1, le modèle reçoit uniquement des images nulles.
        # EN: For E1, the model only receives zero images.
        if self.blind:
            image = torch.zeros_like(image)
        return image, self.ids[index]


# FR : Empile les images en (B,3,64,64) et complète les mots à droite.
# EN: Stacks images into (B,3,64,64) and pads words on the right.
def collate(samples):
    images, sequences = zip(*samples)
    inputs, targets = Tokenizer().batch(sequences)
    return torch.stack(images), inputs, targets


def make_loader(config, split, shuffle=False, seed=None):
    dataset = ShapeScenes(config["data_dir"], split, config["blind"])
    # FR : Fixe l’ordre aléatoire des lots ; num_workers=0 garde le chargement simple.
    # EN: Fixes the random batch order; num_workers=0 keeps loading simple.
    generator = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=collate, generator=generator)
