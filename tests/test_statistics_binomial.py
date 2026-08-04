"""Tests for generic binomial statistics helpers."""

import numpy as np

from dnamic_toolkit.statistics.binomial import (
    aggregate_binomial_chunk_statistics,
    jeffreys_probability_errors,
    jeffreys_probability_interval,
)


def test_jeffreys_probability_interval_uses_observed_probability():
    probability, low, high = jeffreys_probability_interval(
        np.array([0, 5, 10]),
        10,
    )

    np.testing.assert_allclose(probability, [0.0, 0.5, 1.0])
    assert low[0] == 0.0
    assert high[2] == 1.0
    assert high[0] > probability[0]
    assert low[2] < probability[2]
    assert low[1] < probability[1] < high[1]


def test_jeffreys_probability_errors_are_asymmetric_near_boundaries():
    probability, lower_error, upper_error = jeffreys_probability_errors(
        np.array([0, 5, 10]),
        10,
    )

    np.testing.assert_allclose(probability, [0.0, 0.5, 1.0])
    assert lower_error[0] == 0.0
    assert upper_error[0] > 0.0
    assert lower_error[2] > 0.0
    assert upper_error[2] == 0.0


def test_jeffreys_probability_interval_is_vectorised():
    probability, low, high = jeffreys_probability_interval(
        np.array([0, 5]),
        np.array([10, 10]),
    )

    assert probability.shape == (2,)
    assert low.shape == (2,)
    assert high.shape == (2,)


def test_jeffreys_probability_interval_reports_no_estimate_for_zero_shots():
    probability, low, high = jeffreys_probability_interval(0, 0)

    assert np.isnan(probability)
    assert np.isnan(low)
    assert np.isnan(high)


def test_aggregate_binomial_chunk_statistics():
    (
        x,
        probability,
        lower_error,
        upper_error,
        shots,
    ) = aggregate_binomial_chunk_statistics(
        [2.0, 1.0, 2.0],
        [1, 1, 2],
        [2, 2, 2],
    )

    np.testing.assert_allclose(x, [1.0, 2.0])
    np.testing.assert_allclose(probability, [0.5, 0.75])
    assert np.all(lower_error > 0.0)
    assert np.all(upper_error > 0.0)
    np.testing.assert_array_equal(shots, [2, 4])
