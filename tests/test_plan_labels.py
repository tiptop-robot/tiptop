"""Unit tests for cuTAMP action label parsing. Run with: pixi run test-unit"""

import pytest


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Pick(crackers_in_wrapper, grasp1, q1)", "crackers_in_wrapper"),
        ("Pick(green_block,g,q)", "green_block"),  # no spaces
        ("Pick(Rubik's_cube, grasp1, q1)", "Rubik's_cube"),  # apostrophe not truncated
        ("Pick( Rubik's_cube , grasp1, q1)", "Rubik's_cube"),  # surrounding whitespace stripped
        ("PlaceNear(mug, apple, p1, q1)", "mug"),  # other operators
    ],
)
def test_parse_grasped_object(label, expected):
    # Local import to avoid slow transitive imports affecting other tests
    from tiptop.scripts.viz_tiptop_run import parse_grasped_object

    assert parse_grasped_object(label) == expected


@pytest.mark.parametrize("label", ["not a label", "Pick()", "Pick(only_one_arg)"])
def test_parse_grasped_object_raises_on_unparseable(label):
    # Local import to avoid slow transitive imports affecting other tests
    from tiptop.scripts.viz_tiptop_run import parse_grasped_object

    with pytest.raises(ValueError, match="Could not parse object name"):
        parse_grasped_object(label)
