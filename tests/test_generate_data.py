"""Tests for the RLS-provided ShapeScenes generator (generate_data.py).

Covers the dataset spec from the Project Brief (section 4): fixed splits and
seed, word grammar and length bounds, the 27-token vocabulary, reproducibility,
held-out color-shape pairs, and that the rendered pixels agree with the label.
"""

import json
import re
import sys

import numpy as np
import pytest
import torch

import generate_data as gd

SPEC_SPLIT_SIZES = {"train": 20000, "val": 2000, "test": 2000, "test_heldout": 2000}
N = 300  # samples per split in the content tests; keeps the suite fast

_OBJECT = "({})({})({})".format("|".join(gd.SIZE_NAMES), "|".join(gd.COLOR_NAMES), "|".join(gd.SHAPES))
_RELATION = "|".join(gd.RELATIONS)
_WORD_RE = re.compile(rf"{_OBJECT}(?:({_RELATION}){_OBJECT})?")


def parse_word(word):
    """Split a word into ([(size, color, shape), ...], relation or None)."""
    match = _WORD_RE.fullmatch(word)
    assert match is not None, f"word does not follow the ShapeScenes grammar: {word!r}"
    groups = match.groups()
    objects = [groups[0:3]]
    if groups[3] is not None:
        objects.append(groups[4:7])
    return objects, groups[3]


def pairs_in(word):
    objects, _ = parse_word(word)
    return [(color, shape) for _, color, shape in objects]


def all_possible_words():
    objects = [s + c + sh for s in gd.SIZE_NAMES for c in gd.COLOR_NAMES for sh in gd.SHAPES]
    pairs = [a + r + b for a in objects for r in gd.RELATIONS for b in objects]
    return objects + pairs


def locate(image_chw, color, tol=60):
    """Mask of the pixels close to `color`, plus its bounding box (y0, y1, x0, x1)."""
    hwc = image_chw.transpose(1, 2, 0).astype(np.int32)
    mask = np.sqrt(((hwc - np.array(color)) ** 2).sum(axis=-1)) < tol
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask, None
    return mask, (ys.min(), ys.max(), xs.min(), xs.max())


def classify_shape(mask, box):
    """Recover the shape of a solid-color mask from its bounding box: a flat base, else a diagonal probe."""
    y0, y1, x0, x1 = box
    if mask[y1, x0] and mask[y1, x1]:  # both bottom corners filled: square or triangle
        ratio = mask.sum() / ((y1 - y0 + 1) * (x1 - x0 + 1))
        return "square" if ratio > 0.8 else "triangle"
    center_y, center_x, half = (y0 + y1) / 2, (x0 + x1) / 2, (x1 - x0) / 2
    diagonal_filled = mask[int(round(center_y + 0.55 * half)), int(round(center_x + 0.55 * half))]
    return "circle" if diagonal_filled else "cross"


@pytest.fixture(scope="module")
def splits():
    return {
        name: gd.build_split(name, N, gd.SEED, heldout=(name == "test_heldout"))
        for name in gd.SPLIT_IDS
    }


# --- spec constants ---------------------------------------------------------


def test_split_sizes_seed_and_image_size_match_the_brief():
    assert gd.SPLIT_SIZES == SPEC_SPLIT_SIZES
    assert gd.SEED == 42
    assert gd.IMAGE_SIZE == 64


def test_every_possible_word_is_lowercase_letters_within_the_length_bounds():
    words = all_possible_words()
    assert all(re.fullmatch(r"[a-z]+", w) for w in words)
    assert min(len(w) for w in words) == 13
    assert max(len(w) for w in words) == 45


def test_held_out_pairs_are_valid_and_leave_every_color_and_shape_available():
    assert len(set(gd.HELD_OUT_PAIRS)) == len(gd.HELD_OUT_PAIRS)
    all_pairs = {(c, s) for c in gd.COLOR_NAMES for s in gd.SHAPES}
    assert set(gd.HELD_OUT_PAIRS) < all_pairs
    remaining = all_pairs - set(gd.HELD_OUT_PAIRS)
    assert {c for c, _ in remaining} == set(gd.COLOR_NAMES)
    assert {s for _, s in remaining} == set(gd.SHAPES)


# --- output format ----------------------------------------------------------


@pytest.mark.parametrize("name", list(gd.SPLIT_IDS))
def test_images_and_words_have_the_expected_format(splits, name):
    images, words = splits[name]
    assert images.shape == (N, 3, gd.IMAGE_SIZE, gd.IMAGE_SIZE)
    assert images.dtype == np.uint8
    assert images.flags["C_CONTIGUOUS"]
    assert len(words) == N
    for word in words:
        assert 13 <= len(word) <= 45
        parse_word(word)


def test_scenes_contain_one_and_two_objects(splits):
    counts = {len(parse_word(w)[0]) for w in splits["train"][1]}
    assert counts == {1, 2}


def test_images_are_not_blank_and_are_noisy(splits):
    images, _ = splits["train"]
    assert all(img.std() > 10 for img in images)
    background = images[:, :, 0, 0].astype(np.int32)  # top-left pixel is always background
    assert background.std() > 0


# --- reproducibility --------------------------------------------------------


def test_generation_is_deterministic():
    first = gd.build_split("val", 20, gd.SEED, heldout=False)
    second = gd.build_split("val", 20, gd.SEED, heldout=False)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_a_different_seed_gives_different_data():
    first = gd.build_split("val", 20, 1, heldout=False)
    second = gd.build_split("val", 20, 2, heldout=False)
    assert first[1] != second[1]
    assert not np.array_equal(first[0], second[0])


def test_sample_i_does_not_depend_on_how_many_samples_are_requested():
    few = gd.build_split("train", 5, gd.SEED, heldout=False)
    many = gd.build_split("train", 10, gd.SEED, heldout=False)
    assert few[1] == many[1][:5]
    assert np.array_equal(few[0], many[0][:5])


def test_splits_do_not_share_a_random_stream(splits):
    assert not np.array_equal(splits["train"][0][:20], splits["val"][0][:20])
    assert not np.array_equal(splits["val"][0][:20], splits["test"][0][:20])


# --- held-out color-shape pairs ---------------------------------------------


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_held_out_pairs_never_appear_outside_test_heldout(splits, name):
    held_out = set(gd.HELD_OUT_PAIRS)
    for word in splits[name][1]:
        assert not held_out & set(pairs_in(word)), word


def test_every_test_heldout_scene_contains_a_held_out_pair(splits):
    held_out = set(gd.HELD_OUT_PAIRS)
    seen = set()
    for word in splits["test_heldout"][1]:
        found = held_out & set(pairs_in(word))
        assert found, word
        seen |= found
    assert seen == held_out


def test_held_out_colors_and_shapes_are_seen_separately_in_training(splits):
    train_pairs = {p for w in splits["train"][1] for p in pairs_in(w)}
    train_colors = {c for c, _ in train_pairs}
    train_shapes = {s for _, s in train_pairs}
    for color, shape in gd.HELD_OUT_PAIRS:
        assert color in train_colors
        assert shape in train_shapes


# --- placement geometry -----------------------------------------------------


@pytest.mark.parametrize("relation", gd.RELATIONS)
def test_two_objects_are_inside_the_canvas_and_separated_along_the_relation(relation):
    rng = np.random.default_rng(0)
    for _ in range(2000):
        h1, h2 = (gd.SIZES[str(rng.choice(gd.SIZE_NAMES))] for _ in range(2))
        (x1, y1), (x2, y2) = gd.sample_pair_positions(rng, relation, h1, h2)
        for x, y, h in ((x1, y1, h1), (x2, y2, h2)):
            assert gd.MARGIN + h <= x <= gd.IMAGE_SIZE - gd.MARGIN - h
            assert gd.MARGIN + h <= y <= gd.IMAGE_SIZE - gd.MARGIN - h
        if relation == "leftof":
            gap = (x2 - h2) - (x1 + h1)
        elif relation == "rightof":
            gap = (x1 - h1) - (x2 + h2)
        elif relation == "above":
            gap = (y2 - h2) - (y1 + h1)
        else:
            gap = (y1 - h1) - (y2 + h2)
        assert gap >= gd.MIN_GAP - 1e-9  # separated along the relation axis, so never overlapping


def test_two_large_objects_still_fit_side_by_side():
    assert 2 * (2 * gd.SIZES["large"]) + gd.MIN_GAP <= gd.IMAGE_SIZE - 2 * gd.MARGIN


def test_single_object_is_inside_the_canvas():
    rng = np.random.default_rng(0)
    for h in gd.SIZES.values():
        for _ in range(200):
            x, y = gd.sample_single_position(rng, h)
            assert gd.MARGIN + h <= x <= gd.IMAGE_SIZE - gd.MARGIN - h
            assert gd.MARGIN + h <= y <= gd.IMAGE_SIZE - gd.MARGIN - h


# --- pixels agree with the label --------------------------------------------


@pytest.mark.parametrize("size", gd.SIZE_NAMES)
@pytest.mark.parametrize("color", gd.COLOR_NAMES)
@pytest.mark.parametrize("shape", gd.SHAPES)
def test_a_rendered_object_has_the_labelled_shape_color_and_size(shape, color, size):
    obj = {"size": size, "color": color, "shape": shape, "cx": 32.0, "cy": 32.0}
    image = gd.render(np.random.default_rng(0), [obj]).transpose(2, 0, 1)
    mask, box = locate(image, gd.COLORS[color])
    assert box is not None
    y0, y1, x0, x1 = box
    half = gd.SIZES[size]
    assert abs((x1 - x0 + 1) - (2 * half + 1)) <= 1
    assert abs((y1 - y0 + 1) - (2 * half + 1)) <= 1
    assert abs((x0 + x1) / 2 - 32) <= 1 and abs((y0 + y1) / 2 - 32) <= 1
    assert classify_shape(mask, box) == shape


@pytest.mark.parametrize("name", ["train", "test_heldout"])
def test_generated_images_match_their_words(splits, name):
    images, words = splits[name]
    checked_relations = 0
    for image, word in zip(images, words):
        objects, relation = parse_word(word)
        colors = [c for _, c, _ in objects]
        if len(set(colors)) < len(colors):
            continue  # same-colored objects cannot be told apart by color
        found = []
        for size, color, shape in objects:
            mask, box = locate(image, gd.COLORS[color])
            assert box is not None, f"{color} object not visible in {word!r}"
            assert classify_shape(mask, box) == shape, word
            assert abs((box[3] - box[2] + 1) - (2 * gd.SIZES[size] + 1)) <= 1, word
            found.append(((box[2] + box[3]) / 2, (box[0] + box[1]) / 2))
        if relation is not None:
            (x1, y1), (x2, y2) = found
            expected = {"leftof": x1 < x2, "rightof": x1 > x2, "above": y1 < y2, "below": y1 > y2}
            assert expected[relation], word
            checked_relations += 1
    assert checked_relations > 0


# --- command line / files on disk -------------------------------------------


def run_main(monkeypatch, out_dir, *args):
    monkeypatch.setattr(sys, "argv", ["generate_data.py", "--out-dir", str(out_dir), *args])
    gd.main()


def test_main_writes_every_split_and_the_metadata(tmp_path, monkeypatch):
    counts = {"train": 12, "val": 6, "test": 5, "test_heldout": 4}
    run_main(
        monkeypatch, tmp_path,
        "--train-n", "12", "--val-n", "6", "--test-n", "5", "--heldout-n", "4",
    )
    for name, n in counts.items():
        data = torch.load(tmp_path / f"{name}.pt", weights_only=True)
        images, words = gd.build_split(name, n, gd.SEED, heldout=(name == "test_heldout"))
        assert data["images"].dtype == torch.uint8
        assert tuple(data["images"].shape) == (n, 3, gd.IMAGE_SIZE, gd.IMAGE_SIZE)
        assert data["words"] == words
        assert np.array_equal(data["images"].numpy(), images)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["seed"] == gd.SEED
    assert meta["splits"] == counts
    assert meta["image_size"] == gd.IMAGE_SIZE
    assert meta["held_out_pairs"] == [list(p) for p in gd.HELD_OUT_PAIRS]
    assert meta["vocabulary"] == [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["<eos>"]
    assert len(meta["vocabulary"]) == 27


def test_debug_flag_caps_every_split_at_200(tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, "--debug")
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["splits"] == {name: 200 for name in SPEC_SPLIT_SIZES}
