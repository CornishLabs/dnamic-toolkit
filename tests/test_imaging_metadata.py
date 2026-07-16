from dnamic_toolkit.imaging.metadata import (
    IMAGING_READOUT_BLOB_NAME,
    IMAGING_READOUT_BLOB_VERSION,
    ImageReadoutSpec,
    build_imaging_readout_blob,
    rois_from_jsonable,
    rois_to_jsonable,
)


def test_rois_json_round_trip():
    rois = (((0, 2, 1, 3),), ((2, 4, 3, 5),))

    assert rois_from_jsonable(rois_to_jsonable(rois)) == rois


def test_build_imaging_readout_blob():
    rois = (((0, 2, 1, 3),), ((2, 4, 3, 5),))
    blob = build_imaging_readout_blob(
        num_groups=2,
        images={
            0: ImageReadoutSpec(
                image_channel="shot/image0",
                counts_channel="shot/counts_image0",
                shape=(512, 512),
                rois=rois,
            )
        },
        threshold_parameter_fqn="experiment.threshold",
        threshold_parameter_path="threshold_counts",
    )

    assert blob["namespace"] == IMAGING_READOUT_BLOB_NAME
    assert blob["version"] == IMAGING_READOUT_BLOB_VERSION
    assert blob["images"]["image0"]["counts_channel"] == "shot/counts_image0"
    assert blob["occupancy_rule"]["threshold_parameter_path"] == "threshold_counts"
