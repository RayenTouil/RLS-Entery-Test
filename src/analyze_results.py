# FR : Examine les erreurs sans changer le score officiel.
# EN: Inspects errors without changing the official score.

import argparse
import json
from collections import Counter
from pathlib import Path

from src.evaluate import parse_word
from src.utils import write_json


# FR : A leftof B équivaut à B rightof A si tous les attributs sont aussi inversés.
# EN: A leftof B equals B rightof A when all object attributes are also reversed.
def reversed_equivalent(prediction, reference):
    predicted, expected = parse_word(prediction), parse_word(reference)
    inverse = {"leftof": "rightof", "rightof": "leftof", "above": "below", "below": "above"}
    if expected["relation"] is None or predicted["relation"] != inverse[expected["relation"]]:
        return False
    return all(predicted[f"{field}{slot}"] == expected[f"{field}{3 - slot}"]
               for field in ("size", "color", "shape") for slot in (1, 2))


def analyze(rows):
    groups = {}
    # FR : Sépare les scènes à un objet et à deux objets pour comparer la difficulté.
    # EN: Separates one-object and two-object scenes to compare difficulty.
    for count in (1, 2):
        group = [row for row in rows if parse_word(row["reference"])["object_count"] == count]
        groups[str(count)] = {"count": len(group),
                             "exact_match": sum(r["prediction"] == r["reference"] for r in group) / len(group)}
    errors = [row for row in rows if row["prediction"] != row["reference"]]
    # FR : Compte les reformulations équivalentes parmi les erreurs exact-match.
    # EN: Counts equivalent rewordings among strict exact-match errors.
    reversed_errors = [row for row in errors if reversed_equivalent(row["prediction"], row["reference"])]
    # FR : Ces confusions sont ordonnées : une permutation d’objets peut y apparaître.
    # EN: These confusions use object order, so swapped objects can appear here.
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
    # FR : Lit les prédictions déjà produites ; aucun entraînement supplémentaire.
    # EN: Reads existing predictions; no extra training is needed.
    results = analyze(json.loads(path.read_text(encoding="utf-8")))
    write_json(path.with_name(path.stem.replace("_predictions", "_analysis") + ".json"), results)
    print(results)


if __name__ == "__main__":
    main()
