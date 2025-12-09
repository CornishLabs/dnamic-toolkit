import pytest

from dnamic_toolkit.physcalc import add_two_numbers


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 2, 3),
        (0, 0, 0),
        (-5, 7, 2),
        (10, -3, 7),
    ],
)
def test_add_two_numbers(a, b, expected):
    assert add_two_numbers(a, b) == expected
