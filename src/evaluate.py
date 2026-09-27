import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

from src.data import make_loader
from src.generate import generate
from src.model.vlm import TinyVLM
from src.tokenizer import Tokenizer
from src.utils import hardware, setup, write_json


SIZES = ("small", "large")
COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle", "cross")
RELATIONS = ("leftof", "rightof", "above", "below")


def parse_object(text):
    result = {}
    for field, choices in (("size", SIZES), ("color", COLORS)):
        value = next((choice for choice in choices if text.startswith(choice)), None)
        result[field] = value
        if value is not None:
            text = text[len(value):]
    result["shape"] = text if text in SHAPES else None
    return result


def parse_word(word):
    starts = [match.start() for match in re.finditer("small|large", word)]
    second = next((position for position in starts if position > 0), None)
    first_text = word if second is None else word[:second]
    relation = next((value for value in RELATIONS if first_text.endswith(value)), None)
    if relation is not None:
        first_text = first_text[:-len(relation)]
    first = parse_object(first_text)
    other = parse_object(word[second:]) if second is not None else {}
    result = {f"{key}1": value for key, value in first.items()}
    result.update({f"{key}2": other.get(key) for key in ("size", "color", "shape")})
    result["relation"] = relation
    result["object_count"] = 2 if second is not None else 1
    return result


def metrics(predictions, references):
    correct, total = defaultdict(int), defaultdict(int)
    positional_letters = 0
    letter_total = 0
    for prediction, reference in zip(predictions, references):
        predicted, expected = parse_word(prediction), parse_word(reference)
        correct["exact_match"] += prediction == reference
        total["exact_match"] += 1
        correct["object_count"] += predicted["object_count"] == expected["object_count"]
        total["object_count"] += 1
        for field in ("size1", "color1", "shape1", "size2", "color2", "shape2", "relation"):
            if expected[field] is None:
                continue
            correct[field] += predicted[field] == expected[field]
            total[field] += 1
            if field != "relation":
                group = field[:-1]
                correct[group] += predicted[field] == expected[field]
                total[group] += 1
        positional_letters += sum(a == b for a, b in zip(prediction, reference))
        letter_total += max(len(prediction), len(reference))
    scores = {name: correct[name] / count for name, count in total.items()}
    scores["letter_accuracy"] = positional_letters / max(1, letter_total)
    return {"metrics": scores, "denominators": dict(total), "correct": dict(correct),
            "letter_denominator": letter_total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test", "test_heldout"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--zero-images", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    setup(config["seed"], config["threads"])
    config = dict(config, blind=config["blind"] or args.zero_images)
    model = TinyVLM(**config["model"]).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    teacher_correct, teacher_total = 0, 0
    predictions, references = [], []
    for images, inputs, targets in make_loader(config, args.split):
        with torch.inference_mode():
            predicted_ids = model(images.to(args.device), inputs.to(args.device)).argmax(-1).cpu()
        valid_letters = (targets >= 0) & (targets < 26)
        teacher_correct += ((predicted_ids == targets) & valid_letters).sum().item()
        teacher_total += valid_letters.sum().item()
        predictions.extend(generate(model, images.to(args.device)))
        references.extend(Tokenizer().decode(row[row != -100]) for row in targets)
    result = metrics(predictions, references)
    result["metrics"]["teacher_forced_letter_accuracy"] = teacher_correct / teacher_total
    result["teacher_forced_letter_denominator"] = teacher_total
    result.update({"split": args.split, "seed": config["seed"], "n_seeds": 1, "config": config,
                   "checkpoint": str(args.checkpoint), "checkpoint_epoch": saved["completed"],
                   "hardware": hardware(args.device)})
    suffix = "_zero_images" if args.zero_images else ""
    output = Path(args.output or f'{config["output_dir"]}/results_{args.split}{suffix}.json')
    write_json(output, result)
    write_json(output.with_name(output.stem + "_predictions.json"),
               [{"reference": ref, "prediction": pred} for ref, pred in zip(references, predictions)])
    print(result["metrics"])


if __name__ == "__main__":
    main()
