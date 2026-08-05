"""Interactive example for molecule stick-spectrum calculations.

This helper has been made into a Python API rather than a command line API. The
inputs are physical choices that are usually nicest to edit in a script or
notebook, while some future tools may also deserve command line wrappers.
"""

# %%
import matplotlib.pyplot as plt

from dnamic_toolkit.molecules.stick_plot import (
    ACField,
    MoleculeEnvironment,
    StateLabel,
    calculate_stick_spectrum,
    format_transition_table,
    plot_stick_spectrum,
)


# %%
environment = MoleculeEnvironment(
    magnetic_field_gauss=181.699,
    ac_fields=[
        ACField(
            name="1065",
            wavelength_nm=1065,
            intensity_kw_per_cm2=1.89,
            beta=0,
        )
    ],
)

initial = StateLabel(N=2, MF=7, I=5)
target = StateLabel(N=3, MF=8, I=5)


# %%
spectrum = calculate_stick_spectrum(
    molecule_preset="Rb87Cs133",
    nmax=4,
    environment=environment,
    initial=initial,
    target=target,
    num_path_points=20,
    path_index=-1,
    min_dipole_fraction=0.001,
    # Set warn_mixed_labels=True if you want diatomic to report weak labels.
    warn_mixed_labels=False,
)


# %%
print(format_transition_table(spectrum))


# %%
fig, ax = plot_stick_spectrum(spectrum)
plt.show()


# %%
# The returned object keeps the useful arrays available for follow-up analysis.
transition_energy_mhz = spectrum.transition_energy_mhz
sigma_plus = spectrum.sigma_plus_dipole_fraction
pi = spectrum.pi_dipole_fraction
sigma_minus = spectrum.sigma_minus_dipole_fraction
