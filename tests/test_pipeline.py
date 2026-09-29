# FR : Tests ajoutés pour l’alignement des cibles, le padding et la génération.
# EN: Extra tests for target alignment, padding and generation.

import torch
from torch import nn

from src.evaluate import metrics, parse_word
from src.generate import generate
from src.model.vlm import TinyVLM
from src.tokenizer import Tokenizer


# FR : Vérifie les ids, eos et les positions ignorées.
# EN: Checks ids, eos and ignored positions.
def test_tokenizer_and_padding():
    tokenizer = Tokenizer()
    inputs, targets = tokenizer.batch([tokenizer.encode("red"), tokenizer.encode("blue")])
    assert inputs.tolist() == [[17, 4, 3, 26], [1, 11, 20, 4]]
    assert targets.tolist() == [[17, 4, 3, 26, -100], [1, 11, 20, 4, 26]]
    assert tokenizer.decode(tokenizer.encode("largeredcircle")) == "largeredcircle"


# FR : Modifier les logits du padding ne doit pas modifier la loss.
# EN: Changing padding logits must not change the loss.
def test_padding_does_not_change_loss():
    logits = torch.randn(2, 5, 27)
    targets = torch.tensor([[1, 2, 26, -100, -100], [3, 4, 5, 6, 26]])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    before = criterion(logits.flatten(0, 1), targets.flatten())
    logits[0, 3:] = torch.randn(2, 27) * 100
    after = criterion(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(before, after)


# FR : Changer les lettres fournies ne doit pas révéler la première cible au modèle.
# EN: Changing input letters must not reveal the first target to the model.
def test_first_letter_has_no_access_to_ground_truth():
    model = TinyVLM().eval()
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        first = model(images, torch.zeros(2, 5, dtype=torch.long))
        second = model(images, torch.ones(2, 5, dtype=torch.long))
        empty = model(images, torch.empty(2, 0, dtype=torch.long))
    torch.testing.assert_close(first[:, 0], second[:, 0])
    torch.testing.assert_close(first[:, 0], empty[:, 0], atol=1e-6, rtol=1e-5)
    assert first.shape == (2, 6, 27)


# FR : Une faute de forme ne doit pas effacer la taille et la couleur reconnues.
# EN: A misspelled shape must not erase recognized size and color.
def test_misspelling_preserves_other_attributes():
    parsed = parse_word("largeredcirleleftofsmallbluesquare")
    assert parsed["size1"] == "large"
    assert parsed["color1"] == "red"
    assert parsed["shape1"] is None
    assert parsed["relation"] == "leftof"
    assert parsed["shape2"] == "square"
    assert parse_word("")["shape1"] is None
    assert parse_word("nonsense")["color1"] is None


# FR : Les objets et relations manquants doivent compter comme faux.
# EN: Missing objects and relations must count as incorrect.
def test_attribute_denominators_and_no_spelling_correction():
    scores = metrics(["largeredcirle", "smallbluecross"],
                     ["largeredcircle", "smallbluecrossleftoflargeredsquare"])
    assert scores["metrics"]["exact_match"] == 0
    assert scores["denominators"]["relation"] == 1
    assert scores["metrics"]["shape1"] == 0.5
    assert scores["metrics"]["color"] == 2 / 3
    assert scores["metrics"]["shape2"] == 0


# FR : Force des logits simples pour tester les deux conditions d’arrêt.
# EN: Forces simple logits to test both stopping conditions.
def test_generation_stops_at_eos_and_at_length_limit():
    model = TinyVLM().eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[26] = 10
    assert generate(model, torch.zeros(2, 3, 64, 64)) == ["", ""]
    with torch.no_grad():
        model.head.bias[26] = 0
        model.head.bias[0] = 10
    assert generate(model, torch.zeros(1, 3, 64, 64), max_letters=4) == ["aaaa"]
