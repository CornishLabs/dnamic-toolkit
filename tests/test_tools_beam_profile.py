import numpy as np
import pytest

from dnamic_toolkit.tools.beam_profile import (
    BeamProfileSettings,
    analyze_beam_profiles,
    build_parser,
    find_tiffs,
    gaussian_2d,
    main,
    numeric_stem,
)


def test_gaussian_2d_uses_one_over_e2_radius_convention():
    value_at_centre = gaussian_2d(
        np.asarray([0.0]),
        np.asarray([0.0]),
        [1.0, 10.0, 0.0, 0.0, 2.0, 4.0, 0.0],
    )
    value_at_w1 = gaussian_2d(
        np.asarray([2.0]),
        np.asarray([0.0]),
        [1.0, 10.0, 0.0, 0.0, 2.0, 4.0, 0.0],
    )

    np.testing.assert_allclose(value_at_centre, [11.0])
    np.testing.assert_allclose(value_at_w1, [1.0 + 10.0 * np.exp(-2.0)])


def test_numeric_tiff_discovery_sorts_by_filename_distance(tmp_path):
    for name in ("2.0.tif", "not-a-distance.tif", "-1.0.tiff", "0.tif"):
        (tmp_path / name).write_text("")

    paths = find_tiffs(tmp_path)

    assert [path.name for path in paths] == ["-1.0.tiff", "0.tif", "2.0.tif"]
    assert [numeric_stem(path) for path in paths] == [-1.0, 0.0, 2.0]


def test_cli_parser_accepts_centre_aliases(tmp_path):
    args = build_parser().parse_args(
        [str(tmp_path), "--center-x", "1.5", "--center-y", "2.5"]
    )

    assert args.centre_x == 1.5
    assert args.centre_y == 2.5


def test_cli_parser_defaults_match_settings_defaults(tmp_path):
    defaults = BeamProfileSettings(folder=tmp_path)
    args = build_parser().parse_args([str(tmp_path)])

    assert args.centre_x == defaults.centre_x
    assert args.centre_y == defaults.centre_y
    assert args.fit_half_size == defaults.fit_half_size
    assert args.fit_stride == defaults.fit_stride
    assert args.pixel_size == defaults.pixel_size
    assert args.waist_unit == defaults.waist_unit
    assert args.distance_unit == defaults.distance_unit
    assert args.min_angle_r2 == defaults.min_angle_r2
    assert args.robust == defaults.robust


def test_main_and_python_wrapper_report_invalid_inputs(tmp_path):
    missing = tmp_path / "missing"

    assert main([str(missing)]) == 2
    with pytest.raises(ValueError, match="not a directory"):
        analyze_beam_profiles(BeamProfileSettings(folder=missing))
