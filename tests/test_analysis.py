from src.analyze_results import reversed_equivalent


def test_relation_and_both_objects_must_be_reversed():
    reference = "largeredcircleleftofsmallbluecross"
    assert reversed_equivalent("smallbluecrossrightoflargeredcircle", reference)
    assert not reversed_equivalent("smallbluecrossleftoflargeredcircle", reference)
    assert not reversed_equivalent("smallbluecrossrightoflargeredcirle", reference)
    assert not reversed_equivalent("largeredcircle", reference)
