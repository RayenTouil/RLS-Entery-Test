# FR : Affiche une prédiction et sauvegarde l’image correspondante.
# EN: Prints one prediction and saves the corresponding image.

import argparse

import torch
from PIL import Image

from src.data import ShapeScenes
from src.generate import generate
from src.model.vlm import TinyVLM
from src.utils import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "test_heldout"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--image", default="tmp/example.png")
    args = parser.parse_args()
    # FR : Recharge les poids existants sans entraîner un nouveau modèle.
    # EN: Loads existing weights without training a new model.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    setup(config["seed"], config["threads"])
    data = ShapeScenes(config["data_dir"], args.split, config["blind"])
    model = TinyVLM(**config["model"]).to(args.device)
    model.load_state_dict(saved["model"])
    # FR : Ajoute une dimension de batch : une image devient un lot de taille 1.
    # EN: Adds a batch dimension: one image becomes a batch of size 1.
    prediction = generate(model, data[args.index][0][None].to(args.device))[0]
    print("reference:", data.words[args.index])
    print("prediction:", prediction)
    from pathlib import Path
    path = Path(args.image)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FR : PIL attend (hauteur, largeur, canaux), contrairement à PyTorch.
    # EN: PIL expects (height, width, channels), unlike PyTorch.
    image = data.images[args.index].permute(1, 2, 0).numpy()
    Image.fromarray(image).resize((384, 384)).save(path)
    print("image:", path)


if __name__ == "__main__":
    main()
