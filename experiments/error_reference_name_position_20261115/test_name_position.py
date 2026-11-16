"""Define the name-by-line signature before reading the aligned results."""

import pytest
from name_position import classify


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["A", "A", "A", "A"], "all_same"),
        (["A", "B", "B", "A"], "name_aligned"),
        (["A", "B", "A", "B"], "line_aligned"),
        (["A", "B", "C", "A"], "mixed"),
        (["other:x", "B", "B", "other:x"], "name_aligned"),
    ],
)
def test_invariance_categories(answers, expected):
    assert classify(answers) == expected


def test_reject_incomplete_table():
    with pytest.raises(ValueError):
        classify(["A", "B"])
