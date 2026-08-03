"""Small mathematical models shared by apparatus calibrations."""

from dataclasses import dataclass

import numpy as np


def rounded_two_slope(
    x,
    offset,
    slope_low,
    slope_change,
    transition,
    width,
):
    """Evaluate a smoothly rounded transition between two linear gradients.

    Far below ``transition`` the gradient is ``slope_low``; far above it the
    gradient is ``slope_low + slope_change``. ``np.logaddexp`` evaluates the
    soft transition without overflowing for points far from the transition.

    A scalar input produces a Python ``float`` for convenient interactive use. An
    array-like input produces a NumPy array for fitting and plotting.
    """
    x_array = np.asarray(x, dtype=float)
    z = (x_array - transition) / width
    result = (
        offset
        + slope_low * x_array
        + slope_change * width * np.logaddexp(0.0, z)
    )

    if result.ndim == 0:
        return float(result)
    return result


@dataclass(frozen=True)
class RoundedTwoSlopeCalibration:
    """One immutable, traceable set of rounded-two-slope fit coefficients."""

    identifier: str
    source_rid: int
    measured_on: str
    offset: float
    slope_low: float
    slope_change: float
    transition: float
    width: float
    valid_input_range: tuple[float, float]
    input_unit: str
    output_unit: str
    notes: str = ""

    def evaluate(self, x):
        """Evaluate this calibration for a scalar or array-like input."""
        return rounded_two_slope(
            x,
            self.offset,
            self.slope_low,
            self.slope_change,
            self.transition,
            self.width,
        )
