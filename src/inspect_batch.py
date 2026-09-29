# FR : Affiche les dimensions du pipeline sur un vrai lot.
# EN: Prints pipeline shapes for a real batch.

import torch

from src.data import make_loader
from src.model.vlm import TinyVLM
from src.tokenizer import Tokenizer
from src.utils import read_config, setup


def main():
    config = read_config("configs/baseline.yaml")
    setup(config["seed"], config["threads"])
    images, inputs, targets = next(iter(make_loader(config, "train")))
    model = TinyVLM(**config["model"]).eval()
    # FR : Inspecte un forward sans calculer de gradients ni entraîner le modèle.
    # EN: Inspects a forward pass without gradients or training.
    with torch.no_grad():
        features = model.encoder(images)
        visual = model.visual_tokens(images)
        logits = model(images, inputs)
    print("image", tuple(images.shape))
    print("CNN", tuple(features.shape))
    print("visual tokens", tuple(visual.shape))
    print("input letters", tuple(inputs.shape))
    print("logits", tuple(logits.shape))
    print("targets", tuple(targets.shape))
    # FR : Ignore le padding avant de retrouver le mot du premier exemple.
    # EN: Removes padding before decoding the first sample word.
    print("decoded sample", Tokenizer().decode(targets[0][targets[0] != -100]))
    print("parameters", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
