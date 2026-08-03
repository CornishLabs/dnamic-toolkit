"""Analytical properties of a scalar, red-detuned Gaussian optical tweezer.

The frequencies calculated here are the *local harmonic frequencies*: they come
from the curvature of the potential at the very bottom of the trap.  A quadratic
fit over a finite distance, or a parametric-heating measurement with warm atoms,
can give a slightly lower value because a Gaussian trap becomes shallower away
from its centre.

All function arguments use explicit SI units.  Keeping unit conversions at the
call site makes these short formulae much harder to use accidentally with, for
example, milliwatts in place of watts.
"""

from dataclasses import dataclass
from math import pi, sqrt

from scipy.constants import (
    Boltzmann,
    atomic_mass,
    c,
    epsilon_0,
    h,
    physical_constants,
)


BOHR_RADIUS_M = physical_constants["Bohr radius"][0]


def gaussian_beam_centre_intensity_w_m2(
    power_w: float,
    waist_x_m: float,
    waist_y_m: float,
) -> float:
    """Return the peak intensity of an elliptical Gaussian beam in W/m².

    ``waist_x_m`` and ``waist_y_m`` are the usual 1/e² intensity radii.  The
    factor of two in ``2 P / (pi w_x w_y)`` follows from integrating the
    Gaussian intensity profile over the transverse plane.
    """
    if power_w < 0:
        raise ValueError("power_w must not be negative")
    if waist_x_m <= 0 or waist_y_m <= 0:
        raise ValueError("Gaussian beam waists must be positive")

    return 2 * power_w / (pi * waist_x_m * waist_y_m)


def polarizability_au_to_si(polarizability_au: float) -> float:
    """Convert a polarizability from atomic units to SI units."""
    return polarizability_au * 4 * pi * epsilon_0 * BOHR_RADIUS_M**3


def scalar_dipole_potential_j(
    intensity_w_m2: float,
    polarizability_au: float,
) -> float:
    """Return the scalar optical dipole potential in joules.

    With this convention, positive polarizability gives a negative potential:
    the atom is attracted to the high-intensity centre of a red-detuned beam.
    """
    if intensity_w_m2 < 0:
        raise ValueError("intensity_w_m2 must not be negative")

    polarizability_si = polarizability_au_to_si(polarizability_au)
    return -(polarizability_si * intensity_w_m2) / (2 * epsilon_0 * c)


def rayleigh_range_m(waist_m: float, wavelength_m: float) -> float:
    """Return ``pi * waist**2 / wavelength`` in metres."""
    if waist_m <= 0:
        raise ValueError("waist_m must be positive")
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")

    return pi * waist_m**2 / wavelength_m


@dataclass(frozen=True)
class GaussianTweezerProperties:
    """Centre intensity, depth and local harmonic trap frequencies."""

    centre_intensity_w_m2: float
    trap_depth_j: float
    rayleigh_range_m: float
    radial_x_frequency_hz: float
    radial_y_frequency_hz: float
    axial_frequency_hz: float

    @property
    def centre_intensity_kw_cm2(self) -> float:
        """Peak intensity in the convenient laboratory unit kW/cm²."""
        return self.centre_intensity_w_m2 / 1e7

    @property
    def trap_depth_hz(self) -> float:
        """Trap depth divided by Planck's constant, in hertz."""
        return self.trap_depth_j / h

    @property
    def trap_depth_k(self) -> float:
        """Trap depth divided by Boltzmann's constant, in kelvin."""
        return self.trap_depth_j / Boltzmann


def gaussian_tweezer_properties(
    *,
    power_w: float,
    wavelength_m: float,
    waist_x_m: float,
    waist_y_m: float,
    axial_waist_m: float,
    polarizability_au: float,
    mass_amu: float,
) -> GaussianTweezerProperties:
    """Calculate analytical properties of one Gaussian optical tweezer.

    ``axial_waist_m`` is the effective waist used to calculate the Rayleigh
    range controlling axial confinement.  This is the same role played by
    ``waist_z_um`` in ``beams-to-potentials``.  It can differ from the two
    measured transverse waists when using an effective three-waist beam model.

    The centre of a Gaussian beam is a trap only for positive scalar
    polarizability.  Blue-detuned traps require a different intensity geometry
    and are therefore rejected here rather than returning imaginary frequencies.
    """
    if mass_amu <= 0:
        raise ValueError("mass_amu must be positive")
    if polarizability_au <= 0:
        raise ValueError(
            "polarizability_au must be positive for a centre-seeking Gaussian trap"
        )

    intensity = gaussian_beam_centre_intensity_w_m2(
        power_w,
        waist_x_m,
        waist_y_m,
    )
    potential_j = scalar_dipole_potential_j(intensity, polarizability_au)
    trap_depth_j = -potential_j
    mass_kg = mass_amu * atomic_mass
    axial_rayleigh_range_m = rayleigh_range_m(axial_waist_m, wavelength_m)

    # Expanding -U0 exp(-2 x²/w²) around x=0 gives
    # -U0 + 2 U0 x²/w².  Comparing that with m omega² x²/2 gives
    # omega² = 4 U0/(m w²).  Dividing omega by 2 pi returns ordinary Hz.
    radial_x_frequency_hz = sqrt(4 * trap_depth_j / (mass_kg * waist_x_m**2)) / (2 * pi)
    radial_y_frequency_hz = sqrt(4 * trap_depth_j / (mass_kg * waist_y_m**2)) / (2 * pi)

    # On axis, I(z) = I0/(1 + (z/z_R)²).  Its local curvature gives
    # omega_z² = 2 U0/(m z_R²).
    axial_frequency_hz = sqrt(
        2 * trap_depth_j / (mass_kg * axial_rayleigh_range_m**2)
    ) / (2 * pi)

    return GaussianTweezerProperties(
        centre_intensity_w_m2=intensity,
        trap_depth_j=trap_depth_j,
        rayleigh_range_m=axial_rayleigh_range_m,
        radial_x_frequency_hz=radial_x_frequency_hz,
        radial_y_frequency_hz=radial_y_frequency_hz,
        axial_frequency_hz=axial_frequency_hz,
    )
