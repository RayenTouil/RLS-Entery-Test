"""ShapeScenes dataset generator -- RLS Entrance Challenge.

Generates 64x64 RGB scenes of 1-2 colored shapes and the one-word
description of each scene: size + color + shape [+ relation + size +
color + shape]. See the Project Brief, section 4, for the full spec.

Provided by RLS. Run with the fixed seed and do not change the splits.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    def tqdm(iterable, **kwargs):
        return iterable

IMAGE_SIZE = 64
MARGIN = 3  # minimum distance from an object's edge to the image border
MIN_GAP = 4  # minimum distance between the edges of two objects

SIZE_NAMES = ["small", "large"]
# Half-extent in pixels. Two large objects must fit side by side with a gap:
# 2 * (12 + 12) + MIN_GAP <= IMAGE_SIZE - 2 * MARGIN.
SIZES = {"small": 7, "large": 12}

COLOR_NAMES = ["red", "green", "blue", "yellow"]
COLORS = {
    "red": (214, 39, 40),
    "green": (44, 160, 44),
    "blue": (31, 119, 180),
    "yellow": (240, 200, 30),
}

SHAPES = ["circle", "square", "triangle", "cross"]
RELATIONS = ["leftof", "rightof", "above", "below"]

BACKGROUND = (250, 250, 248)
NOISE_SIGMA = 6.0

# Color-shape pairs withheld from train/val/test and reserved for
# test-heldout, so the model never sees these exact combinations during
# training even though each color and each shape appear separately.
HELD_OUT_PAIRS = [("green", "triangle"), ("yellow", "cross"), ("blue", "square")]
HELD_OUT_SET = set(HELD_OUT_PAIRS)

SEED = 42
SPLIT_SIZES = {"train": 20000, "val": 2000, "test": 2000, "test_heldout": 2000}
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "test_heldout": 3}


def sample_object_attrs(rng, forbid_heldout, force_pair=None):
    if force_pair is not None:
        color, shape = force_pair
    else:
        while True:
            color = str(rng.choice(COLOR_NAMES))
            shape = str(rng.choice(SHAPES))
            if not forbid_heldout or (color, shape) not in HELD_OUT_SET:
                break
    size = str(rng.choice(SIZE_NAMES))
    return {"size": size, "color": color, "shape": shape}


def sample_single_position(rng, half_extent):
    lo, hi = half_extent + MARGIN, IMAGE_SIZE - half_extent - MARGIN
    return rng.uniform(lo, hi), rng.uniform(lo, hi)


def sample_axis_pair(rng, half_first, half_second):
    # Centers of two objects along one axis, the first one before the second,
    # with at least MIN_GAP between their edges and both inside the margins.
    lo = half_first + MARGIN
    hi = IMAGE_SIZE - half_second - MARGIN
    first = rng.uniform(lo, hi - half_first - half_second - MIN_GAP)
    second = rng.uniform(first + half_first + half_second + MIN_GAP, hi)
    return first, second


def sample_pair_positions(rng, relation, half1, half2):
    # The relation holds by construction: the objects are separated along the
    # relation's axis, so they never overlap. Along the other axis each object
    # is placed independently.
    y1 = rng.uniform(half1 + MARGIN, IMAGE_SIZE - half1 - MARGIN)
    y2 = rng.uniform(half2 + MARGIN, IMAGE_SIZE - half2 - MARGIN)
    x1 = rng.uniform(half1 + MARGIN, IMAGE_SIZE - half1 - MARGIN)
    x2 = rng.uniform(half2 + MARGIN, IMAGE_SIZE - half2 - MARGIN)
    if relation == "leftof":
        x1, x2 = sample_axis_pair(rng, half1, half2)
    elif relation == "rightof":
        x2, x1 = sample_axis_pair(rng, half2, half1)
    elif relation == "above":
        y1, y2 = sample_axis_pair(rng, half1, half2)
    elif relation == "below":
        y2, y1 = sample_axis_pair(rng, half2, half1)
    else:
        raise ValueError(f"unknown relation {relation!r}")
    return (x1, y1), (x2, y2)


def draw_shape(draw, obj):
    cx, cy = obj["cx"], obj["cy"]
    h = SIZES[obj["size"]]
    color = COLORS[obj["color"]]
    shape = obj["shape"]
    if shape == "circle":
        draw.ellipse([cx - h, cy - h, cx + h, cy + h], fill=color)
    elif shape == "square":
        draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - h), (cx - h, cy + h), (cx + h, cy + h)], fill=color)
    elif shape == "cross":
        bar = h * 0.55
        draw.rectangle([cx - h, cy - bar / 2, cx + h, cy + bar / 2], fill=color)
        draw.rectangle([cx - bar / 2, cy - h, cx + bar / 2, cy + h], fill=color)
    else:
        raise ValueError(f"unknown shape {shape!r}")


def render(rng, objects):
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    for obj in objects:
        draw_shape(draw, obj)
    arr = np.asarray(img, dtype=np.int16)
    arr = arr + rng.normal(0.0, NOISE_SIGMA, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def describe(obj):
    return obj["size"] + obj["color"] + obj["shape"]


def sample_scene(rng, heldout):
    n_objects = 1 if rng.random() < 0.5 else 2

    if n_objects == 1:
        force = HELD_OUT_PAIRS[rng.integers(len(HELD_OUT_PAIRS))] if heldout else None
        obj = sample_object_attrs(rng, forbid_heldout=not heldout, force_pair=force)
        obj["cx"], obj["cy"] = sample_single_position(rng, SIZES[obj["size"]])
        objects, relation = [obj], None
    else:
        relation = str(rng.choice(RELATIONS))
        forced_slot = int(rng.integers(0, 2)) if heldout else -1
        obj1 = sample_object_attrs(
            rng,
            forbid_heldout=forced_slot != 0,
            force_pair=HELD_OUT_PAIRS[rng.integers(len(HELD_OUT_PAIRS))] if forced_slot == 0 else None,
        )
        obj2 = sample_object_attrs(
            rng,
            forbid_heldout=forced_slot != 1,
            force_pair=HELD_OUT_PAIRS[rng.integers(len(HELD_OUT_PAIRS))] if forced_slot == 1 else None,
        )
        (obj1["cx"], obj1["cy"]), (obj2["cx"], obj2["cy"]) = sample_pair_positions(
            rng, relation, SIZES[obj1["size"]], SIZES[obj2["size"]]
        )
        objects = [obj1, obj2]

    image = render(rng, objects)
    word = describe(objects[0]) if relation is None else describe(objects[0]) + relation + describe(objects[1])
    assert word.isalpha() and word.islower()
    assert 13 <= len(word) <= 45
    return image, word


def build_split(name, n, seed, heldout):
    split_id = SPLIT_IDS[name]
    images = np.empty((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    words = []
    for i in tqdm(range(n), desc=name):
        rng = np.random.default_rng([seed, split_id, i])
        image, word = sample_scene(rng, heldout=heldout)
        images[i] = image
        words.append(word)
    images = images.transpose(0, 3, 1, 2).copy()  # (N, H, W, C) -> (N, C, H, W)
    return images, words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-n", type=int, default=SPLIT_SIZES["train"])
    parser.add_argument("--val-n", type=int, default=SPLIT_SIZES["val"])
    parser.add_argument("--test-n", type=int, default=SPLIT_SIZES["test"])
    parser.add_argument("--heldout-n", type=int, default=SPLIT_SIZES["test_heldout"])
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate a handful of samples per split for a quick smoke test. "
        "Never use this for an actual submission -- it changes the splits.",
    )
    args = parser.parse_args()

    counts = {"train": args.train_n, "val": args.val_n, "test": args.test_n, "test_heldout": args.heldout_n}
    if args.debug:
        counts = {k: min(v, 200) for k, v in counts.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, n in counts.items():
        t0 = time.time()
        images, words = build_split(name, n, args.seed, heldout=(name == "test_heldout"))
        torch.save({"images": torch.from_numpy(images), "words": words}, args.out_dir / f"{name}.pt")
        print(f"{name}: {n} samples in {time.time() - t0:.1f}s -> {args.out_dir / f'{name}.pt'}")

    meta = {
        "seed": args.seed,
        "image_size": IMAGE_SIZE,
        "colors": COLOR_NAMES,
        "shapes": SHAPES,
        "sizes": SIZE_NAMES,
        "relations": RELATIONS,
        "held_out_pairs": HELD_OUT_PAIRS,
        "vocabulary": list("abcdefghijklmnopqrstuvwxyz") + ["<eos>"],
        "splits": counts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {args.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
