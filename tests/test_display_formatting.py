import pytest

from dnamic_toolkit.display.formatting import format_uncertainty


def test_parentheses_uncertainty_formatting():
    assert format_uncertainty(0.031, 0.0099) == "0.03(1)"
    assert format_uncertainty(0.0342, 0.0123) == "0.034(12)"
    assert format_uncertainty(12.34, 1.23) == "12.3(12)"
    assert format_uncertainty("2.25", "0.1") == "2.3(1)"


def test_plus_minus_uncertainty_formatting():
    assert (
        format_uncertainty(0.031, 0.0099, style="plus_minus")
        == "0.03 +/- 0.01"
    )
    assert format_uncertainty(0.031, 0.0099, style="latex") == "0.03 \\pm 0.01"


def test_scientific_uncertainty_formatting():
    assert format_uncertainty(12345, 67) == "1.235(7)e4"
    assert format_uncertainty(1.2345e-5, 5.6e-7) == "1.23(6)e-5"
    assert (
        format_uncertainty(1.2345e-5, 5.6e-7, style="latex")
        == "(1.23 \\pm 0.06) \\times 10^{-5}"
    )
    assert (
        format_uncertainty(
            12345,
            67,
            style="plus_minus",
            notation="scientific",
        )
        == "(1.235 +/- 0.007)e4"
    )


def test_fixed_notation_can_be_forced():
    assert format_uncertainty(12345, 67, notation="fixed") == "12350(70)"


def test_uncertainty_rounding_can_increase_magnitude():
    assert format_uncertainty(12.34, 9.9) == "10(10)"


def test_negative_zero_is_not_displayed():
    assert format_uncertainty(-0.001, 0.01) == "0.00(1)"


@pytest.mark.parametrize("uncertainty", [0, -1, float("nan"), float("inf")])
def test_invalid_uncertainties_are_rejected(uncertainty):
    with pytest.raises(ValueError):
        format_uncertainty(1.0, uncertainty)


@pytest.mark.parametrize("value", ["bad", float("nan"), float("inf")])
def test_invalid_values_are_rejected(value):
    with pytest.raises(ValueError):
        format_uncertainty(value, 0.1)


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        format_uncertainty(1.0, 0.1, style="pm")
    with pytest.raises(ValueError):
        format_uncertainty(1.0, 0.1, notation="SCI")
