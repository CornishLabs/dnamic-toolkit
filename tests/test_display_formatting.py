import pytest

from dnamic_toolkit.display.formatting import format_uncertainty


def test_parentheses_uncertainty_formatting():
    assert format_uncertainty(0.031, 0.0099) == "0.03(1)"
    assert format_uncertainty(0.0342, 0.0123) == "0.034(12)"
    assert format_uncertainty(12.34, 1.23) == "12.3(12)"


def test_plus_minus_uncertainty_formatting():
    assert (
        format_uncertainty(0.031, 0.0099, style="plus_minus")
        == "0.03 +/- 0.01"
    )
    assert format_uncertainty(0.031, 0.0099, style="latex") == "0.03 \\pm 0.01"


def test_scientific_uncertainty_formatting():
    assert format_uncertainty(1.2345e-5, 5.6e-7) == "1.23(6)e-5"
    assert (
        format_uncertainty(1.2345e-5, 5.6e-7, style="latex")
        == "(1.23 \\pm 0.06) \\times 10^{-5}"
    )


def test_negative_zero_is_not_displayed():
    assert format_uncertainty(-0.001, 0.01) == "0.00(1)"


@pytest.mark.parametrize("uncertainty", [0, -1, float("nan"), float("inf")])
def test_invalid_uncertainties_are_rejected(uncertainty):
    with pytest.raises(ValueError):
        format_uncertainty(1.0, uncertainty)


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        format_uncertainty(1.0, 0.1, style="pm")
    with pytest.raises(ValueError):
        format_uncertainty(1.0, 0.1, notation="SCI")
