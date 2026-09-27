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
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    setup(config["seed"], config["threads"])
    data = ShapeScenes(config["data_dir"], args.split, config["blind"])
    model = TinyVLM(**config["model"]).to(args.device)
    model.load_state_dict(saved["model"])
    prediction = generate(model, data[args.index][0][None].to(args.device))[0]
    print("reference:", data.words[args.index])
    print("prediction:", prediction)
    from pathlib import Path
    path = Path(args.image)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = data.images[args.index].permute(1, 2, 0).numpy()
    Image.fromarray(image).resize((384, 384)).save(path)
    print("image:", path)


if __name__ == "__main__":
    main()
