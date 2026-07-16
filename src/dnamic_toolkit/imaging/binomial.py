"""Binomial statistics helpers shared by online and offline image analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.special import betaincinv


def estimate_probability_from_counts(
    num_successes: float,
    num_shots: int,
) -> tuple[float, float]:
    """Return a binomial probability estimate and conservative standard error."""

    if num_shots < 0:
        raise ValueError("num_shots must be non-negative")
    if num_shots == 0:
        return 0.0, float("inf")

    probability = float(num_successes) / float(num_shots)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("num_successes must lie between 0 and num_shots")

    error = math.sqrt(max(probability * (1.0 - probability), 0.0) / num_shots)
    error = max(error, 0.5 / num_shots)
    return probability, error


def estimate_probability_array(
    num_successes: np.ndarray,
    num_shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version of :func:`estimate_probability_from_counts`."""

    successes = np.asarray(num_successes, dtype=float)
    if num_shots < 0:
        raise ValueError("num_shots must be non-negative")
    if num_shots == 0:
        return (
            np.zeros(successes.shape, dtype=float),
            np.full(successes.shape, float("inf"), dtype=float),
        )

    probability = successes / float(num_shots)
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("num_successes must lie between 0 and num_shots")

    error = np.sqrt(np.maximum(probability * (1.0 - probability), 0.0) / num_shots)
    error = np.maximum(error, 0.5 / num_shots)
    return probability, error


def jeffreys_median_ci(
    num_successes,
    num_shots,
    *,
    level: float = 0.6827,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Jeffreys posterior median and equal-tailed credible interval.

    This is useful for sparse loading data where normal error bars behave badly
    near probabilities of 0 or 1.
    """

    successes = np.asarray(num_successes, dtype=float)
    shots = np.asarray(num_shots, dtype=float)
    if np.any(shots < 0):
        raise ValueError("num_shots must be non-negative")
    if np.any(successes < 0) or np.any(successes > shots):
        raise ValueError("num_successes must lie between 0 and num_shots")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie between 0 and 1")

    alpha = successes + 0.5
    beta = shots - successes + 0.5
    tail = 0.5 * (1.0 - float(level))
    median = betaincinv(alpha, beta, 0.5)
    low = betaincinv(alpha, beta, tail)
    high = betaincinv(alpha, beta, 1.0 - tail)
    return median, low, high


def aggregate_binomial_chunk_statistics(
    x_values: Sequence[float],
    num_successes: Sequence[float | int],
    num_shots: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated chunk-level statistics into one estimate per x value."""

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
    probabilities = np.empty(len(unique_x), dtype=float)
    probability_errors = np.empty(len(unique_x), dtype=float)
    total_shots = np.empty(len(unique_x), dtype=int)

    for index, x_value in enumerate(unique_x):
        total = shots_by_x[float(x_value)]
        probability, probability_error = estimate_probability_from_counts(
            successes_by_x[float(x_value)],
            total,
        )
        probabilities[index] = probability
        probability_errors[index] = probability_error
        total_shots[index] = total

    return unique_x, probabilities, probability_errors, total_shots
