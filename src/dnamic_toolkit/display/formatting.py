"""Formatting helpers for values with uncertainties."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal, TypeAlias

__all__ = ["format_uncertainty"]

Number: TypeAlias = Decimal | int | float | str
Style: TypeAlias = Literal["parentheses", "plus_minus", "latex"]
Notation: TypeAlias = Literal["auto", "fixed", "scientific"]


def format_uncertainty(
    value: Number,
    uncertainty: Number,
    *,
    style: Style = "parentheses",
    notation: Notation = "auto",
) -> str:
    """Format a value and its one-standard-error uncertainty."""

    if style not in ("parentheses", "plus_minus", "latex"):
        raise ValueError(
            "style must be 'parentheses', 'plus_minus', or 'latex'"
        )
    if notation not in ("auto", "fixed", "scientific"):
        raise ValueError(
            "notation must be 'auto', 'fixed', or 'scientific'"
        )

    value_decimal = _to_decimal(value, "value")
    uncertainty_decimal = _to_decimal(uncertainty, "uncertainty")

    if uncertainty_decimal <= 0:
        raise ValueError("uncertainty must be positive")

    rounded_uncertainty, decimal_place = _round_uncertainty(
        uncertainty_decimal
    )
    rounded_value = _quantize(value_decimal, decimal_place)

    magnitude = max(abs(rounded_value), rounded_uncertainty)
    use_scientific = (
        notation == "scientific"
        or (
            notation == "auto"
            and (
                magnitude.adjusted() >= 4
                or magnitude.adjusted() <= -3
            )
        )
    )

    if use_scientific:
        scale_exponent = (
            rounded_value.adjusted()
            if not rounded_value.is_zero()
            else rounded_uncertainty.adjusted()
        )
    else:
        scale_exponent = 0

    scale = Decimal(1).scaleb(scale_exponent)
    scaled_place = decimal_place - scale_exponent

    value_text = format(
        _quantize(rounded_value / scale, scaled_place),
        "f",
    )
    uncertainty_text = format(
        _quantize(rounded_uncertainty / scale, scaled_place),
        "f",
    )

    if style == "parentheses":
        digits = uncertainty_text.replace(".", "").lstrip("0") or "0"
        result = f"{value_text}({digits})"
        return (
            f"{result}e{scale_exponent}"
            if scale_exponent
            else result
        )

    operator = r"\pm" if style == "latex" else "+/-"
    result = f"{value_text} {operator} {uncertainty_text}"

    if scale_exponent == 0:
        return result
    if style == "latex":
        return rf"({result}) \times 10^{{{scale_exponent}}}"
    return f"({result})e{scale_exponent}"


def _to_decimal(value: Number, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error

    if not result.is_finite():
        raise ValueError(f"{name} must be finite")

    return result


def _round_uncertainty(
    uncertainty: Decimal,
) -> tuple[Decimal, int]:
    # Base-10 exponent of the first significant digit.
    # For example, 0.0123 has an adjusted exponent of -2.
    original_exponent = uncertainty.adjusted()

    # Decimal digits without the sign or decimal point.
    # For example, Decimal("0.0123") produces (1, 2, 3).
    digits = uncertainty.as_tuple().digits

    # Keep two significant digits when the leading digit is 1 or 2,
    # provided the uncertainty contains another non-zero digit.
    # Otherwise, keep one significant digit.
    significant_digits = (
        2
        if digits[0] in (1, 2) and any(digits[1:])
        else 1
    )

    # Convert the desired number of significant digits into the decimal
    # exponent expected by quantize().
    # Example: 0.0123 with two significant digits rounds at 1e-3.
    decimal_place = original_exponent - significant_digits + 1
    rounded = _quantize(uncertainty, decimal_place)

    # Rounding may increase the order of magnitude, such as 9.9 -> 10.
    # Recalculate the decimal place so the rounded result still has the
    # intended number of significant digits.
    if rounded.adjusted() > original_exponent:
        decimal_place = rounded.adjusted() - significant_digits + 1
        rounded = _quantize(uncertainty, decimal_place)

    return rounded, decimal_place


def _quantize(value: Decimal, exponent: int) -> Decimal:
    """Round value to exponent place 
        >>> _quantize(Decimal('8.453'),1)
        Decimal('1E+1')
        >>> _quantize(Decimal('8.453'),0) 
        Decimal('8')
        >>> _quantize(Decimal('8.453'),-1) 
        Decimal('8.5')
        >>> _quantize(Decimal('8.453'),-2)  
        Decimal('8.45')
    """
    quantum = Decimal(1).scaleb(exponent)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    return abs(rounded) if rounded.is_zero() else rounded