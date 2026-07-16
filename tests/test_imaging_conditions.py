import numpy as np

from dnamic_toolkit.imaging.conditions import (
    Occupied,
    conditional_binomial,
    parse_condition_syntax,
)
from dnamic_toolkit.imaging.rois import counts_to_occupancy_stack


def test_condition_parser_for_compact_syntax():
    condition = parse_condition_syntax("1[0,1] & 2[!3]")

    assert condition.conditions[0].conditions[0] == Occupied(1, 0)


def test_conditional_binomial_from_occupancy_stack():
    counts0 = np.asarray(
        [
            [[12, 5], [4, 20]],
            [[11, 15], [1, 2]],
            [[2, 3], [30, 31]],
        ]
    )
    counts1 = np.asarray(
        [
            [[12, 13], [1, 20]],
            [[5, 15], [11, 1]],
            [[20, 21], [30, 5]],
        ]
    )
    occupancy = counts_to_occupancy_stack((counts0, counts1), threshold=10)

    result = conditional_binomial(
        occupancy,
        event=parse_condition_syntax("1[0,1]"),
    )

    np.testing.assert_array_equal(result.num_selected_by_group, [3, 3])
    np.testing.assert_array_equal(result.num_successes_by_group, [2, 0])
    np.testing.assert_allclose(result.probability_by_group, [2 / 3, 0.0])
    assert result.pooled_num_selected == 6
    assert result.pooled_num_successes == 2
