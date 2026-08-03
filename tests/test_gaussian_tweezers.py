import pytest

from dnamic_toolkit.physics.gaussian_tweezers import (
    gaussian_beam_centre_intensity_w_m2,
    gaussian_tweezer_properties,
)


def test_centre_intensity_matches_cs_tweezer_example():
    intensity = gaussian_beam_centre_intensity_w_m2(
        power_w=7.07e-3,
        waist_x_m=1.05e-6,
        waist_y_m=1.16e-6,
    )

    assert intensity / 1e7 == pytest.approx(369.5, rel=1e-3)


def test_analytical_cs_tweezer_properties():
    properties = gaussian_tweezer_properties(
        power_w=7.07e-3,
        wavelength_m=1065e-9,
        waist_x_m=1.05e-6,
        waist_y_m=1.16e-6,
        axial_waist_m=1.19e-6,
        polarizability_au=1168,
        mass_amu=133,
    )

    assert properties.centre_intensity_kw_cm2 == pytest.approx(369.5, rel=1e-3)
    assert properties.trap_depth_hz / 1e6 == pytest.approx(20.23, rel=1e-3)
    assert properties.trap_depth_k * 1e3 == pytest.approx(0.9709, rel=1e-3)
    assert properties.rayleigh_range_m / 1e-6 == pytest.approx(4.177, rel=1e-3)
    assert properties.radial_x_frequency_hz / 1e3 == pytest.approx(74.7, rel=1e-3)
    assert properties.radial_y_frequency_hz / 1e3 == pytest.approx(67.6, rel=1e-3)
    assert properties.axial_frequency_hz / 1e3 == pytest.approx(13.3, rel=3e-3)


def test_invalid_centre_trap_inputs_are_rejected():
    with pytest.raises(ValueError, match="polarizability_au must be positive"):
        gaussian_tweezer_properties(
            power_w=1e-3,
            wavelength_m=1066e-9,
            waist_x_m=1e-6,
            waist_y_m=1e-6,
            axial_waist_m=1e-6,
            polarizability_au=-1,
            mass_amu=133,
        )

    with pytest.raises(ValueError, match="must not be negative"):
        gaussian_beam_centre_intensity_w_m2(-1, 1e-6, 1e-6)
