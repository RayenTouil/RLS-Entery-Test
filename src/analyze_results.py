import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluate import parse_word
from src.utils import write_json


def reversed_equivalent(prediction, reference):
    predicted, expected = parse_word(prediction), parse_word(reference)
    inverse = {"leftof": "rightof", "rightof": "leftof", "above": "below", "below": "above"}
    if expected["relation"] is None or predicted["relation"] != inverse[expected["relation"]]:
        return False
    return all(predicted[f"{field}{slot}"] == expected[f"{field}{3 - slot}"]
               for field in ("size", "color", "shape") for slot in (1, 2))


def analyze(rows):
    groups = {}
    for count in (1, 2):
        group = [row for row in rows if parse_word(row["reference"])["object_count"] == count]
        groups[str(count)] = {"count": len(group),
                             "exact_match": sum(r["prediction"] == r["reference"] for r in group) / len(group)}
    errors = [row for row in rows if row["prediction"] != row["reference"]]
    reversed_errors = [row for row in errors if reversed_equivalent(row["prediction"], row["reference"])]
    shape_errors = Counter()
    for row in rows:
        expected, predicted = parse_word(row["reference"]), parse_word(row["prediction"])
        for slot in (1, 2):
            field = f"shape{slot}"
            if expected[field] is not None and predicted[field] != expected[field]:
                shape_errors[f'{expected[field]} -> {predicted[field]}'] += 1
    return {"by_object_count": groups, "strict_errors": len(errors),
            "errors_equivalent_after_reversing_objects": len(reversed_errors),
            "other_errors": len(errors) - len(reversed_errors),
            "reversed_examples": reversed_errors[:5],
            "other_examples": [row for row in errors if not reversed_equivalent(row["prediction"], row["reference"])][:5],
            "shape_confusions": dict(shape_errors.most_common())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    path = Path(args.predictions)
    results = analyze(json.loads(path.read_text(encoding="utf-8")))
    write_json(path.with_name(path.stem.replace("_predictions", "_analysis") + ".json"), results)
    print(results)


if __name__ == "__main__":
    main()
