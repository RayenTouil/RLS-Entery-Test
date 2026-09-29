# FR : Teste les descriptions avec objets et relation inversés.
# EN: Tests descriptions with reversed objects and relation.

from src.analyze_results import reversed_equivalent


# FR : Vérifie les attributs et la relation, pas seulement l’ordre des noms.
# EN: Checks the attributes and relation, not just the order of names.
def test_relation_and_both_objects_must_be_reversed():
    reference = "largeredcircleleftofsmallbluecross"
    assert reversed_equivalent("smallbluecrossrightoflargeredcircle", reference)
    assert not reversed_equivalent("smallbluecrossleftoflargeredcircle", reference)
    assert not reversed_equivalent("smallbluecrossrightoflargeredcirle", reference)
    assert not reversed_equivalent("largeredcircle", reference)
