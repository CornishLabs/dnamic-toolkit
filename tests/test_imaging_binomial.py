import numpy as np

from dnamic_toolkit.imaging.binomial import (
    aggregate_binomial_chunk_statistics,
    estimate_probability_array,
    estimate_probability_from_counts,
    jeffreys_median_ci,
)


def test_estimate_probability_from_counts():
    probability, error = estimate_probability_from_counts(5, 10)

    assert probability == 0.5
    assert error > 0.0


def test_estimate_probability_array():
    probability, error = estimate_probability_array(np.array([0, 5, 10]), 10)

    np.testing.assert_allclose(probability, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(error[[0, 2]], [0.05, 0.05])


def test_jeffreys_median_ci_is_vectorised():
    median, low, high = jeffreys_median_ci(np.array([0, 5]), np.array([10, 10]))

    assert median.shape == (2,)
    assert np.all(low < median)
    assert np.all(median < high)


def test_aggregate_binomial_chunk_statistics():
    x, probability, error, shots = aggregate_binomial_chunk_statistics(
        [2.0, 1.0, 2.0],
        [1, 1, 2],
        [2, 2, 2],
    )

    np.testing.assert_allclose(x, [1.0, 2.0])
    np.testing.assert_allclose(probability, [0.5, 0.75])
    assert np.all(error > 0.0)
    np.testing.assert_array_equal(shots, [2, 4])
