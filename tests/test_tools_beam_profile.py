import numpy as np
import pytest

from dnamic_toolkit.tools.beam_profile import (
    BeamProfileSettings,
    FitResult,
    ImageRecord,
    analyze_beam_profiles,
    build_parser,
    fit_gaussian_beam_propagation,
    finite_plot_arrays,
    find_tiffs,
    gaussian_2d,
    gaussian_beam_radius,
    gaussian_beam_radius_standard_error,
    main,
    numeric_stem,
    plot_waists,
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


def test_gaussian_beam_radius_error_propagates_parameter_covariance():
    covariance = np.diag([0.2**2, 0.3**2, 0.4**2])

    error = gaussian_beam_radius_standard_error(
        np.asarray([0.0, 2.0]),
        w0=2.0,
        z0=0.0,
        z_rayleigh=4.0,
        covariance=covariance,
    )

    assert error[0] == pytest.approx(0.2)
    assert np.all(np.isfinite(error))
    assert np.all(error > 0)


def test_propagation_fit_uses_image_fit_radius_errors():
    distance = np.linspace(-4.0, 4.0, 9)
    expected = gaussian_beam_radius(distance, w0=2.0, z0=0.25, z_rayleigh=3.5)
    measured = expected.copy()
    measured[-1] += 3.0

    equally_weighted = fit_gaussian_beam_propagation(
        distance,
        measured,
        np.ones_like(measured),
        robust_loss=False,
    )
    uncertainty_weighted = fit_gaussian_beam_propagation(
        distance,
        measured,
        np.asarray([0.05] * 8 + [10.0]),
        robust_loss=False,
    )

    dense_distance = np.linspace(-4.0, 4.0, 101)
    truth = gaussian_beam_radius(dense_distance, 2.0, 0.25, 3.5)
    equal_prediction = gaussian_beam_radius(
        dense_distance,
        equally_weighted.w0,
        equally_weighted.z0,
        equally_weighted.z_rayleigh,
    )
    weighted_prediction = gaussian_beam_radius(
        dense_distance,
        uncertainty_weighted.w0,
        uncertainty_weighted.z0,
        uncertainty_weighted.z_rayleigh,
    )

    assert equally_weighted.success
    assert uncertainty_weighted.success
    assert np.mean((weighted_prediction - truth) ** 2) < np.mean(
        (equal_prediction - truth) ** 2
    )


def test_propagation_parameter_errors_preserve_absolute_uncertainty_scale():
    distance = np.linspace(-4.0, 4.0, 9)
    measured = gaussian_beam_radius(distance, w0=2.0, z0=0.25, z_rayleigh=3.5)

    small_errors = fit_gaussian_beam_propagation(
        distance,
        measured,
        np.full_like(measured, 0.05),
        robust_loss=False,
    )
    large_errors = fit_gaussian_beam_propagation(
        distance,
        measured,
        np.full_like(measured, 0.15),
        robust_loss=False,
    )

    assert small_errors.success
    assert large_errors.success
    assert large_errors.w0_error == pytest.approx(3.0 * small_errors.w0_error)
    assert large_errors.z0_error == pytest.approx(3.0 * small_errors.z0_error)
    assert large_errors.z_rayleigh_error == pytest.approx(
        3.0 * small_errors.z_rayleigh_error
    )


def test_radius_values_and_errors_use_the_same_pixel_size_conversion(tmp_path):
    params = np.asarray([0.0, 1.0, 0.0, 0.0, 2.0, 3.0])
    errors = np.asarray([0.1, 0.1, 0.1, 0.1, 0.2, 0.3])
    fit = FitResult(
        success=True,
        message="synthetic fit",
        params=params,
        errors=errors,
        covariance=np.diag(errors**2),
        theta_fixed=0.0,
        r2=1.0,
        rmse=0.0,
        noise=0.1,
        crop_bounds=(0, 10, 0, 10),
        n_points=100,
    )
    record = ImageRecord(
        path=tmp_path / "0.tif",
        distance=0.0,
        aligned=fit,
        free=fit,
    )

    _, radius_1, error_1, radius_2, error_2 = finite_plot_arrays(
        [record], fit_selector=lambda item: item.aligned, pixel_size=6.5
    )

    np.testing.assert_allclose(radius_1, [13.0])
    np.testing.assert_allclose(error_1, [1.3])
    np.testing.assert_allclose(radius_2, [19.5])
    np.testing.assert_allclose(error_2, [1.95])


def test_propagation_fit_preserves_waist_unit_scaling():
    distance = np.linspace(-4.0, 4.0, 9)
    radii_pixels = gaussian_beam_radius(
        distance, w0=2.0, z0=0.25, z_rayleigh=3.5
    )
    errors_pixels = np.full_like(radii_pixels, 0.05)
    pixel_size = 6.5

    fit_pixels = fit_gaussian_beam_propagation(
        distance, radii_pixels, errors_pixels, robust_loss=False
    )
    fit_physical = fit_gaussian_beam_propagation(
        distance,
        radii_pixels * pixel_size,
        errors_pixels * pixel_size,
        robust_loss=False,
    )

    assert fit_physical.w0 == pytest.approx(fit_pixels.w0 * pixel_size)
    assert fit_physical.w0_error == pytest.approx(
        fit_pixels.w0_error * pixel_size
    )
    assert fit_physical.z0 == pytest.approx(fit_pixels.z0)
    assert fit_physical.z0_error == pytest.approx(fit_pixels.z0_error)
    assert fit_physical.z_rayleigh == pytest.approx(fit_pixels.z_rayleigh)
    assert fit_physical.z_rayleigh_error == pytest.approx(
        fit_pixels.z_rayleigh_error
    )


def test_waist_plot_renders_measurement_errors_and_fit_uncertainty(tmp_path):
    distance = np.linspace(-4.0, 4.0, 9)
    radii_1 = gaussian_beam_radius(distance, w0=2.0, z0=0.0, z_rayleigh=3.5)
    radii_2 = gaussian_beam_radius(distance, w0=3.0, z0=0.5, z_rayleigh=5.0)

    records = []
    for index, (z, radius_1, radius_2) in enumerate(
        zip(distance, radii_1, radii_2, strict=True)
    ):
        params = np.asarray([0.0, 1.0, 0.0, 0.0, radius_1, radius_2])
        errors = np.asarray([0.1, 0.1, 0.1, 0.1, 0.05, 0.08])
        fit = FitResult(
            success=True,
            message="synthetic fit",
            params=params,
            errors=errors,
            covariance=np.diag(errors**2),
            theta_fixed=0.0,
            r2=1.0,
            rmse=0.0,
            noise=0.1,
            crop_bounds=(0, 10, 0, 10),
            n_points=100,
        )
        records.append(
            ImageRecord(
                path=tmp_path / f"{index}.tif",
                distance=float(z),
                aligned=fit,
                free=fit,
            )
        )

    output = tmp_path / "waists.png"
    propagation_1, propagation_2 = plot_waists(
        records,
        fit_selector=lambda record: record.aligned,
        label_1="u radius",
        label_2="v radius",
        title="Synthetic beam radii",
        output_path=output,
        pixel_size=1.0,
        waist_unit="pixel",
        distance_unit="mm",
        robust_loss=False,
    )

    assert propagation_1.success
    assert propagation_2.success
    assert propagation_1.w0_error > 0
    assert propagation_2.w0_error > 0
    assert output.stat().st_size > 0


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
