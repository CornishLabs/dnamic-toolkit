import numpy as np
import pytest

from dnamic_toolkit.imaging.rois import (
    counts_to_occupancy_stack,
    sum_counts_in_rois,
    threshold_counts_to_occupancy,
    validate_rois,
)


def test_sum_counts_in_rois():
    image = np.arange(20, dtype=np.uint16).reshape(4, 5)
    rois = (
        ((0, 2, 0, 2), (2, 4, 3, 5)),
        ((1, 3, 1, 4), (0, 1, 0, 5)),
    )

    counts = sum_counts_in_rois(image, rois)

    np.testing.assert_array_equal(
        counts,
        [
            [12, 64],
            [57, 10],
        ],
    )


def test_validate_rois_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="exceed"):
        validate_rois((((0, 10, 0, 10),),), image_shape=(4, 5))


def test_threshold_counts_to_occupancy():
    occupancy = threshold_counts_to_occupancy(np.asarray([[3, 4], [5, 6]]), 5)

    np.testing.assert_array_equal(occupancy, [[False, False], [True, True]])


def test_counts_to_occupancy_stack_allows_different_roi_counts():
    image0_counts = np.zeros((3, 2, 1), dtype=np.uint32)
    image1_counts = np.ones((3, 2, 2), dtype=np.uint32)

    occupancy = counts_to_occupancy_stack(
        (image0_counts, image1_counts),
        threshold=(1, 1),
    )

    assert occupancy[0].shape == (3, 2, 1)
    assert occupancy[1].shape == (3, 2, 2)
    assert not np.any(occupancy[0])
    assert np.all(occupancy[1])
