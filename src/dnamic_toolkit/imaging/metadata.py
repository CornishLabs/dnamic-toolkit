"""Metadata helpers describing how saved images map to ROI counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnamic_toolkit.imaging.rois import RoiGroups, normalise_rois

IMAGING_READOUT_BLOB_NAME = "lab.imaging_readout"
IMAGING_READOUT_BLOB_VERSION = 1


@dataclass(frozen=True)
class ImageReadoutSpec:
    """Description of one image and its corresponding ROI-count channel."""

    image_channel: str
    counts_channel: str
    shape: tuple[int, int]
    rois: RoiGroups
    image_dim_names: tuple[str, str] = ("y", "x")
    counts_dim_names: tuple[str, str] = ("group", "roi")


def rois_to_jsonable(rois) -> list[list[list[int]]]:
    """Return ROI bounds in a JSON/PYON-friendly nested-list representation."""

    return [[list(bounds) for bounds in group] for group in normalise_rois(rois)]


def rois_from_jsonable(data) -> RoiGroups:
    """Parse ROI bounds from a JSON/PYON-friendly nested-list representation."""

    return normalise_rois(data)


def _image_spec_to_blob(spec: ImageReadoutSpec) -> dict[str, Any]:
    return {
        "image_channel": spec.image_channel,
        "counts_channel": spec.counts_channel,
        "shape": list(spec.shape),
        "image_dim_names": list(spec.image_dim_names),
        "counts_dim_names": list(spec.counts_dim_names),
        "rois": rois_to_jsonable(spec.rois),
    }


def build_imaging_readout_blob(
    *,
    num_groups: int,
    images: dict[int, ImageReadoutSpec],
    threshold_parameter_fqn: str | None = None,
    threshold_parameter_path: str | None = None,
    kind: str = "roi_threshold_imaging",
    namespace: str = IMAGING_READOUT_BLOB_NAME,
    version: int = IMAGING_READOUT_BLOB_VERSION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata blob for online and offline image/ROI consumers."""

    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if not images:
        raise ValueError("At least one image spec is required")

    occupancy_rule: dict[str, Any] = {"kind": "threshold_counts"}
    if threshold_parameter_fqn is not None:
        occupancy_rule["threshold_parameter_fqn"] = threshold_parameter_fqn
    if threshold_parameter_path is not None:
        occupancy_rule["threshold_parameter_path"] = threshold_parameter_path

    blob: dict[str, Any] = {
        "namespace": namespace,
        "version": int(version),
        "kind": kind,
        "num_groups": int(num_groups),
        "images": {
            f"image{int(image_index)}": _image_spec_to_blob(spec)
            for image_index, spec in sorted(images.items())
        },
        "occupancy_rule": occupancy_rule,
    }
    if extra:
        blob.update(extra)
    return blob
