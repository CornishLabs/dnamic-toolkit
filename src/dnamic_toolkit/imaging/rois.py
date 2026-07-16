"""ROI reduction helpers for fluorescence images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

RoiBounds: TypeAlias = tuple[int, int, int, int]
RoiGroups: TypeAlias = tuple[tuple[RoiBounds, ...], ...]


def normalise_rois(rois: Sequence[Sequence[Sequence[int]]]) -> RoiGroups:
    """Return ROIs as immutable ``(group, roi, bounds)`` tuples.

    Bounds are ordered as ``(y0, y1, x0, x1)`` and use normal NumPy half-open
    slicing semantics.
    """

    normalised_groups = []
    for group in rois:
        normalised_rois = []
        for bounds in group:
            if len(bounds) != 4:
                raise ValueError("Each ROI must have four bounds: (y0, y1, x0, x1)")
            y0, y1, x0, x1 = (int(value) for value in bounds)
            normalised_rois.append((y0, y1, x0, x1))
        normalised_groups.append(tuple(normalised_rois))
    return tuple(normalised_groups)


def validate_rois(
    rois: Sequence[Sequence[Sequence[int]]],
    *,
    image_shape: tuple[int, int] | None = None,
    require_rectangular: bool = True,
) -> RoiGroups:
    """Validate and normalise one grouped ROI layout."""

    normalised = normalise_rois(rois)
    if not normalised:
        raise ValueError("At least one ROI group is required")

    expected_num_rois = len(normalised[0])
    if expected_num_rois == 0:
        raise ValueError("At least one ROI per group is required")

    for group_index, group in enumerate(normalised):
        if require_rectangular and len(group) != expected_num_rois:
            raise ValueError(
                "All ROI groups must have the same number of ROIs; "
                f"group 0 has {expected_num_rois}, group {group_index} has {len(group)}"
            )
        for bounds in group:
            y0, y1, x0, x1 = bounds
            if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
                raise ValueError(f"Invalid ROI bounds {bounds}")
            if image_shape is not None:
                image_height, image_width = image_shape
                if y1 > image_height or x1 > image_width:
                    raise ValueError(
                        f"ROI bounds {bounds} exceed image shape {image_shape}"
                    )
    return normalised


def sum_counts_in_rois(
    image: np.ndarray,
    rois: Sequence[Sequence[Sequence[int]]],
    *,
    dtype=np.uint32,
) -> np.ndarray:
    """Return integrated counts with shape ``(group, roi)`` for one image."""

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("image must be a 2D array")

    normalised = validate_rois(rois, image_shape=image_array.shape)
    counts = np.empty((len(normalised), len(normalised[0])), dtype=dtype)
    dtype_info = np.iinfo(counts.dtype) if np.issubdtype(counts.dtype, np.integer) else None

    for group_index, group in enumerate(normalised):
        for roi_index, (y0, y1, x0, x1) in enumerate(group):
            value = int(np.sum(image_array[y0:y1, x0:x1]))
            if dtype_info is not None:
                value = min(max(value, int(dtype_info.min)), int(dtype_info.max))
            counts[group_index, roi_index] = value

    return counts


def threshold_counts_to_occupancy(counts: np.ndarray, threshold) -> np.ndarray:
    """Infer boolean occupancy from integrated ROI counts."""

    return np.asarray(counts) >= np.asarray(threshold)


def counts_to_occupancy_stack(
    counts_by_image: Sequence[np.ndarray],
    *,
    threshold,
) -> tuple[np.ndarray, ...]:
    """Threshold one counts stack per image.

    Each counts array must have shape ``(shots, group, roi)``. All images must
    share the same number of shots and groups, but the ROI dimension may differ.
    """

    arrays = [np.asarray(counts) for counts in counts_by_image]
    if not arrays:
        raise ValueError("At least one image counts array is required")
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("Each counts array must have shape (shots, group, roi)")

    first_shots, first_groups, _ = arrays[0].shape
    for array in arrays[1:]:
        shots, groups, _ = array.shape
        if shots != first_shots or groups != first_groups:
            raise ValueError(
                "All image counts arrays must share the same number of shots and groups"
            )

    if isinstance(threshold, list | tuple) and len(threshold) == len(arrays):
        return tuple(
            threshold_counts_to_occupancy(array, threshold[index])
            for index, array in enumerate(arrays)
        )

    return tuple(threshold_counts_to_occupancy(array, threshold) for array in arrays)
