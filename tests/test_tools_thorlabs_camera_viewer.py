import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from dnamic_toolkit.tools.thorlabs_camera_viewer import (
    CameraWorker,
    FramePacket,
    _simple_colormap,
    clipped_roi_bounds,
    roi_statistics,
)


def test_clipped_roi_bounds_rounds_and_stays_inside_image():
    assert clipped_roi_bounds((-4.2, 8.6), (5.4, 20.0), (12, 16)) == (
        0,
        5,
        9,
        12,
    )


def test_roi_statistics_uses_raw_camera_counts():
    image = np.arange(20, dtype=np.uint16).reshape(4, 5)

    result = roi_statistics(image, (1, 4, 1, 3))

    expected = image[1:3, 1:4]
    assert result.pixel_count == expected.size
    assert result.total == float(np.sum(expected))
    assert result.mean == float(np.mean(expected))
    assert result.minimum == float(np.min(expected))
    assert result.maximum == float(np.max(expected))


def test_camera_worker_keeps_only_the_latest_frame():
    worker = CameraWorker(camera_id=None)
    image = np.zeros((2, 2), dtype=np.uint16)

    for frame_count in range(100):
        worker._store_latest_frame(
            FramePacket(
                image=image,
                frame_count=frame_count,
                camera_timestamp_ns=None,
                received_at=float(frame_count),
                acquisition_fps=10.0,
                sdk_fps=10.0,
                skipped_frames=0,
            )
        )

    assert worker.take_latest_frame().frame_count == 99
    assert worker.take_latest_frame() is None


def test_colour_map_has_a_small_number_of_histogram_handles():
    assert len(_simple_colormap("magma").pos) == 6
    assert len(_simple_colormap("grey").pos) == 2


def test_colour_map_can_reserve_its_upper_limit_for_red():
    colour_map = _simple_colormap("magma", red_at_top=True)

    assert len(colour_map.pos) == 7
    np.testing.assert_array_equal(
        colour_map.getLookupTable(0.0, 1.0, 256)[-1, :3],
        [255, 0, 0],
    )
