"""Binomial statistics helpers shared by online and offline image analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.special import betaincinv

DEFAULT_JEFFREYS_LEVEL = 0.6827


def jeffreys_probability_interval(
    num_successes,
    num_shots,
    *,
    level: float = DEFAULT_JEFFREYS_LEVEL,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return observed probability and modified Jeffreys interval bounds.

    The plotted estimate is the observed fraction ``k / n``. The interval is
    the equal-tailed Jeffreys posterior interval, with endpoints clipped to
    0 or 1 when all observations are failures or successes. Zero-shot inputs
    return NaN because there is no observed proportion to plot.
    """

    successes, shots = _validate_binomial_counts(num_successes, num_shots)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie between 0 and 1")

    probability = np.full(successes.shape, np.nan, dtype=float)
    low = np.full(successes.shape, np.nan, dtype=float)
    high = np.full(successes.shape, np.nan, dtype=float)

    has_shots = shots > 0
    probability[has_shots] = successes[has_shots] / shots[has_shots]

    posterior_alpha = successes[has_shots] + 0.5
    posterior_beta = shots[has_shots] - successes[has_shots] + 0.5
    tail = 0.5 * (1.0 - float(level))
    low[has_shots] = betaincinv(posterior_alpha, posterior_beta, tail)
    high[has_shots] = betaincinv(
        posterior_alpha,
        posterior_beta,
        1.0 - tail,
    )

    # Modified Jeffreys intervals include the observed boundary exactly.
    low[has_shots & (successes == 0)] = 0.0
    high[has_shots & (successes == shots)] = 1.0

    return (
        _as_scalar_if_scalar(probability),
        _as_scalar_if_scalar(low),
        _as_scalar_if_scalar(high),
    )


def jeffreys_probability_errors(
    num_successes,
    num_shots,
    *,
    level: float = DEFAULT_JEFFREYS_LEVEL,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return probability and asymmetric error-bar lengths for plotting."""

    probability, low, high = jeffreys_probability_interval(
        num_successes,
        num_shots,
        level=level,
    )
    return probability, probability - low, high - probability


def _as_scalar_if_scalar(value: np.ndarray) -> np.ndarray | float:
    if value.shape == ():
        return float(value)
    return value


def _validate_binomial_counts(
    num_successes,
    num_shots,
) -> tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(num_successes, dtype=float)
    shots = np.asarray(num_shots, dtype=float)
    try:
        successes, shots = np.broadcast_arrays(successes, shots)
    except ValueError as error:
        raise ValueError(
            "num_successes and num_shots must be broadcastable"
        ) from error

    if np.any(~np.isfinite(successes)) or np.any(~np.isfinite(shots)):
        raise ValueError("num_successes and num_shots must be finite")
    if np.any(shots < 0):
        raise ValueError("num_shots must be non-negative")
    if np.any(successes < 0) or np.any(successes > shots):
        raise ValueError("num_successes must lie between 0 and num_shots")

    return successes, shots


def aggregate_binomial_chunk_statistics(
    x_values: Sequence[float],
    num_successes: Sequence[float | int],
    num_shots: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse chunk statistics into probability and Jeffreys errors by x.

    Returns ``x, probability, lower_error, upper_error, total_shots``.
    """

    successes_by_x = defaultdict(float)
    shots_by_x = defaultdict(int)
    for x_value, successes_chunk, shots_chunk in zip(
        x_values,
        num_successes,
        num_shots,
        strict=True,
    ):
        if int(shots_chunk) < 0:
            raise ValueError("num_shots entries must be non-negative")
        key = float(x_value)
        successes_by_x[key] += float(successes_chunk)
        shots_by_x[key] += int(shots_chunk)

    unique_x = np.asarray(sorted(shots_by_x), dtype=float)
    total_successes = np.asarray(
        [successes_by_x[float(x_value)] for x_value in unique_x],
        dtype=float,
    )
    total_shots = np.asarray(
        [shots_by_x[float(x_value)] for x_value in unique_x],
        dtype=int,
    )
    (
        probabilities,
        probability_error_low,
        probability_error_high,
    ) = jeffreys_probability_errors(total_successes, total_shots)

    return (
        unique_x,
        probabilities,
        probability_error_low,
        probability_error_high,
        total_shots,
    )
