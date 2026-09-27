from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import Tokenizer


class ShapeScenes(Dataset):
    def __init__(self, data_dir="data", split="train", blind=False):
        data = torch.load(Path(data_dir) / f"{split}.pt", weights_only=True)
        self.images = data["images"]
        self.words = data["words"]
        self.ids = [Tokenizer().encode(word) for word in self.words]
        self.blind = blind

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        image = self.images[index].float().div(255)
        if self.blind:
            image = torch.zeros_like(image)
        return image, self.ids[index]


def collate(samples):
    images, sequences = zip(*samples)
    inputs, targets = Tokenizer().batch(sequences)
    return torch.stack(images), inputs, targets


def make_loader(config, split, shuffle=False, seed=None):
    dataset = ShapeScenes(config["data_dir"], split, config["blind"])
    generator = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=collate, generator=generator)
