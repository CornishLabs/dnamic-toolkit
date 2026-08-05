"""Stick-spectrum helpers for singlet-sigma molecules.

This module is intentionally a Python API rather than a command line tool. The
useful workflow is normally to tweak the physical inputs in a script or notebook,
run the calculation, then inspect the table and plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

import diatomic
import diatomic.calculate as calculate
import diatomic.operators as operators
from diatomic.systems import SingletSigmaMolecule

#%%
GAUSS = 1e-4  # T
MHz = scipy.constants.h * 1e6
muN = scipy.constants.physical_constants["nuclear magneton"][0]

# Define helpful constants
pi = scipy.constants.pi
bohr = scipy.constants.physical_constants["Bohr radius"][0]
eps0 = scipy.constants.epsilon_0
to_cgs = 4 * pi * eps0 * bohr**3

kWpercm2 = 1e7


@dataclass(frozen=True)
class StateLabel:
    """Quantum-number label used to select an adiabatic state branch."""

    N: int
    MF: float | None = None
    I: float | None = None


@dataclass(frozen=True)
class ACField:
    """One perturbing AC field.

    The field is ramped from zero to ``intensity_kw_per_cm2`` along the
    adiabatic path used for state labelling.
    """

    wavelength_nm: int
    intensity_kw_per_cm2: float
    beta: float = 0
    name: str = ""


@dataclass(frozen=True)
class MoleculeEnvironment:
    """External fields used for the stick-spectrum calculation.

    The DC electric field is included for the same adiabatic-following machinery,
    but is assumed to point along the lab-frame z axis.
    """

    magnetic_field_gauss: float
    ac_fields: Sequence[ACField] = ()
    electric_field_v_per_cm: float = 0


@dataclass(frozen=True)
class StickSpectrum:
    """Calculated transition strengths for one selected intensity point."""

    molecule: Any
    molecule_preset: str
    nmax: int
    environment: MoleculeEnvironment
    path_fraction: np.ndarray
    path_index: int
    initial: StateLabel
    target: StateLabel | None
    label_names: tuple[str, ...]
    initial_state_index: int
    target_state_index: int | None
    state_energy_mhz: float
    target_transition_energy_mhz: float | None
    min_dipole_fraction: float
    labels: np.ndarray
    transition_energy_mhz: np.ndarray
    detuning_from_target_mhz: np.ndarray
    magnetic_moment_muN: np.ndarray
    sigma_plus_dipole_fraction: np.ndarray
    pi_dipole_fraction: np.ndarray
    sigma_minus_dipole_fraction: np.ndarray
    eigenenergies: np.ndarray
    eigenstates: np.ndarray


#%%
def label_to_indices(labels, N, MF=None, I=None, label_names=None):
    labels = np.asarray(labels)

    if label_names is None:
        label_names = ("N", "MF") if labels.shape[1] == 2 else ("N", "MF", "I")

    mask = np.ones(labels.shape[0], dtype=bool)
    for name, value in (("N", N), ("MF", MF), ("I", I)):
        if value is None:
            continue
        try:
            column = label_names.index(name)
        except ValueError as error:
            raise ValueError(f"Labels do not include {name!r}.") from error
        mask &= labels[:, column] == value

    return np.where(mask)[0]


def calculate_stick_spectrum(
    *,
    molecule_preset="Rb87Cs133",
    nmax=4,
    environment=None,
    initial=None,
    target=None,
    num_path_points=20,
    path_index=-1,
    min_dipole_fraction=0.001,
    configure_logging=True,
    warn_mixed_labels=False,
):
    """Calculate transition strengths for a stick-spectrum plot.

    States are labelled at the zero-perturbation end of the path, then followed
    adiabatically to the requested environment. This matters when AC fields mix
    the nominal quantum numbers, for example with ``beta != 0``.
    """

    if num_path_points < 2:
        raise ValueError("num_path_points must be at least 2.")
    if min_dipole_fraction < 0:
        raise ValueError("min_dipole_fraction must be non-negative.")

    if environment is None:
        environment = MoleculeEnvironment(
            magnetic_field_gauss=181.699,
            ac_fields=(ACField(wavelength_nm=1065, intensity_kw_per_cm2=1.89),),
        )
    environment = _normalise_environment(environment)

    if initial is None:
        initial = StateLabel(N=2, MF=7)

    if configure_logging:
        diatomic.configure_logging()

    # Generate Molecule
    mol = SingletSigmaMolecule.from_preset(molecule_preset)
    mol.Nmax = nmax

    # Generate Hamiltonians
    H0 = operators.hyperfine_ham(mol)
    Hz = operators.zeeman_ham(mol)
    B = environment.magnetic_field_gauss * GAUSS

    reference_hamiltonian = H0 + Hz * B
    perturbing_hamiltonian = _perturbing_hamiltonian(mol, environment, H0)
    path_fraction = np.linspace(0, 1, num_path_points)

    # Overall Hamiltonian. The first point has no perturbing fields, so labels
    # are assigned where N/MF/I are meaningful; later points follow that branch.
    Htot = (
        reference_hamiltonian[None, :, :]
        + path_fraction[:, None, None] * perturbing_hamiltonian
    )
    eigenenergies, eigenstates = calculate.solve_system(Htot)

    # Apply labels (in some way arbitrary) warn if duplicate
    label_names = _label_names_for_states(initial, target)
    eigenlabels = calculate.label_states(
        mol,
        eigenstates[0],
        list(label_names),
        warn_mixed=warn_mixed_labels,
    )

    magnetic_moments = calculate.magnetic_moment(mol, eigenstates)

    #%%
    try:
        idx = range(num_path_points)[path_index]
    except IndexError as error:
        raise IndexError("path_index is outside the generated path.") from error

    state = _single_state_index(eigenlabels, label_names, initial, "initial")
    target_state = None
    if target is not None:
        target_state = _single_state_index(eigenlabels, label_names, target, "target")

    transition_sigma_plus = calculate.transition_electric_moments(
        mol, eigenstates[:, :, :], h=1, from_states=state
    )
    transition_pi = calculate.transition_electric_moments(
        mol, eigenstates[:, :, :], h=0, from_states=state
    )
    transition_sigma_minus = calculate.transition_electric_moments(
        mol, eigenstates[:, :, :], h=-1, from_states=state
    )

    state_energy = eigenenergies[idx, state] / MHz
    transition_energy_mhz = eigenenergies[idx, :] / MHz - state_energy

    if target_state is None:
        target_transition_energy_mhz = None
        detuning_from_target_mhz = transition_energy_mhz
    else:
        target_transition_energy_mhz = float(transition_energy_mhz[target_state])
        detuning_from_target_mhz = (
            transition_energy_mhz - target_transition_energy_mhz
        )

    return StickSpectrum(
        molecule=mol,
        molecule_preset=molecule_preset,
        nmax=nmax,
        environment=environment,
        path_fraction=path_fraction,
        path_index=idx,
        initial=initial,
        target=target,
        label_names=label_names,
        initial_state_index=state,
        target_state_index=target_state,
        state_energy_mhz=float(state_energy),
        target_transition_energy_mhz=target_transition_energy_mhz,
        min_dipole_fraction=min_dipole_fraction,
        labels=eigenlabels,
        transition_energy_mhz=transition_energy_mhz,
        detuning_from_target_mhz=detuning_from_target_mhz,
        magnetic_moment_muN=magnetic_moments[idx, :] / muN,
        sigma_plus_dipole_fraction=transition_sigma_plus[idx, 0, :] / mol.d0,
        pi_dipole_fraction=transition_pi[idx, 0, :] / mol.d0,
        sigma_minus_dipole_fraction=transition_sigma_minus[idx, 0, :] / mol.d0,
        eigenenergies=eigenenergies,
        eigenstates=eigenstates,
    )


def visible_transition_indices(spectrum: StickSpectrum):
    dipole_above_threshold = (
        (spectrum.sigma_plus_dipole_fraction > spectrum.min_dipole_fraction)
        | (spectrum.pi_dipole_fraction > spectrum.min_dipole_fraction)
        | (spectrum.sigma_minus_dipole_fraction > spectrum.min_dipole_fraction)
    )
    not_below_initial_N = spectrum.labels[:, 0] >= spectrum.initial.N
    return np.where(dipole_above_threshold & not_below_initial_N)[0]


def format_transition_table(spectrum: StickSpectrum) -> str:
    lines = [
        f"Initial state: {_format_state_label(spectrum.initial)}",
        f"Environment: {_format_environment(spectrum)}",
        (
            "All transitions from initial state with dipole moment > "
            "%.3f d_mol" % spectrum.min_dipole_fraction
        ),
    ]

    if spectrum.target is not None:
        lines.insert(1, f"Target state: {_format_state_label(spectrum.target)}")
        lines.insert(
            2,
            (
                "Target transition: "
                f"{spectrum.target_transition_energy_mhz:.6f} MHz"
            ),
        )

    label_columns = "\t".join(spectrum.label_names)
    if spectrum.target is None:
        lines.append(f"State\tFreq (MHz)\t<mu>\t{label_columns}\td (d_mol)")
    else:
        lines.append(
            "State\tFreq (MHz)\tDetuning (MHz)\t"
            f"<mu>\t{label_columns}\td (d_mol)"
        )

    for i in visible_transition_indices(spectrum):
        label_values = "\t".join(str(value) for value in spectrum.labels[i])
        dipole_values = (
            "(%.3f, %.3f, %.3f)"
            % (
                spectrum.sigma_plus_dipole_fraction[i],
                spectrum.pi_dipole_fraction[i],
                spectrum.sigma_minus_dipole_fraction[i],
            )
        )
        if spectrum.target is None:
            lines.append(
                "%d\t%.6f\t%.3f\t%s\t%s"
                % (
                    i,
                    spectrum.transition_energy_mhz[i],
                    spectrum.magnetic_moment_muN[i],
                    label_values,
                    dipole_values,
                )
            )
        else:
            lines.append(
                "%d\t%.6f\t%.6f\t%.3f\t%s\t%s"
                % (
                    i,
                    spectrum.transition_energy_mhz[i],
                    spectrum.detuning_from_target_mhz[i],
                    spectrum.magnetic_moment_muN[i],
                    label_values,
                    dipole_values,
                )
            )

    return "\n".join(lines)


def print_transition_table(spectrum: StickSpectrum) -> None:
    print(format_transition_table(spectrum))


def plot_stick_spectrum(spectrum: StickSpectrum, ax=None, center_on_target=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    if center_on_target and spectrum.target is not None:
        x_values = spectrum.detuning_from_target_mhz
        x_label = "Detuning from target / h (MHz)"
    else:
        x_values = spectrum.transition_energy_mhz
        x_label = "Energy / h (MHz)"

    for i in visible_transition_indices(spectrum):
        if spectrum.sigma_minus_dipole_fraction[i] > spectrum.min_dipole_fraction:
            x_pos = x_values[i]
            y_pos = spectrum.sigma_minus_dipole_fraction[i]
            ax.scatter(x_pos, y_pos, color="red", alpha=0.5)
            ax.plot([x_pos, x_pos], [0, y_pos], color="red", alpha=0.5, linewidth=1)
        if spectrum.pi_dipole_fraction[i] > spectrum.min_dipole_fraction:
            x_pos = x_values[i]
            y_pos = spectrum.pi_dipole_fraction[i]
            ax.scatter(x_pos, y_pos, color="blue", alpha=0.5)
            ax.plot([x_pos, x_pos], [0, y_pos], color="blue", alpha=0.5, linewidth=1)
        if spectrum.sigma_plus_dipole_fraction[i] > spectrum.min_dipole_fraction:
            x_pos = x_values[i]
            y_pos = spectrum.sigma_plus_dipole_fraction[i]
            ax.scatter(x_pos, y_pos, color="green", alpha=0.5)
            ax.plot([x_pos, x_pos], [0, y_pos], color="green", alpha=0.5, linewidth=1)

    ax.set_ylim(bottom=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Transition Dipole Moment / $d_{mol}$")
    ax.set_title("Green +, Blue 0, Red -")
    return fig, ax


def _normalise_environment(environment):
    return MoleculeEnvironment(
        magnetic_field_gauss=environment.magnetic_field_gauss,
        ac_fields=tuple(environment.ac_fields),
        electric_field_v_per_cm=environment.electric_field_v_per_cm,
    )


def _perturbing_hamiltonian(mol, environment, reference_shape):
    perturbing_hamiltonian = np.zeros_like(reference_shape)

    if environment.electric_field_v_per_cm != 0:
        electric_field_v_per_m = environment.electric_field_v_per_cm * 100
        perturbing_hamiltonian += operators.dc_ham(mol) * electric_field_v_per_m

    for field in environment.ac_fields:
        field_hamiltonian = operators.ac_ham(
            mol, mol.a02[field.wavelength_nm], beta=field.beta
        )
        perturbing_hamiltonian += (
            field_hamiltonian * field.intensity_kw_per_cm2 * kWpercm2
        )

    return perturbing_hamiltonian


def _label_names_for_states(*states):
    label_names = ["N"]
    if any(state is not None and state.MF is not None for state in states):
        label_names.append("MF")
    if any(state is not None and state.I is not None for state in states):
        label_names.append("I")
    return tuple(label_names)


def _single_state_index(labels, label_names, state, role):
    state_indices = label_to_indices(
        labels,
        state.N,
        MF=state.MF,
        I=state.I,
        label_names=label_names,
    )
    if state_indices.size == 0:
        raise ValueError(f"Could not find {role} state {_format_state_label(state)}.")
    if state_indices.size > 1:
        raise ValueError(
            f"{role.capitalize()} state {_format_state_label(state)} matches "
            f"{state_indices.size} adiabatic branches; provide more labels."
        )
    return int(state_indices[0])


def _format_state_label(state):
    parts = [f"N={state.N}"]
    if state.MF is not None:
        parts.append(f"MF={state.MF}")
    if state.I is not None:
        parts.append(f"I={state.I}")
    return ", ".join(parts)


def _format_environment(spectrum):
    environment = spectrum.environment
    fraction = spectrum.path_fraction[spectrum.path_index]
    parts = [f"B={environment.magnetic_field_gauss:.3f} G"]

    if environment.electric_field_v_per_cm != 0:
        electric_field = environment.electric_field_v_per_cm * fraction
        parts.append(f"E={electric_field:.6g} V/cm")

    for index, field in enumerate(environment.ac_fields):
        name = field.name or f"AC{index}"
        intensity = field.intensity_kw_per_cm2 * fraction
        parts.append(
            f"{name}={field.wavelength_nm} nm, "
            f"{intensity:.6g} kW/cm^2, beta={field.beta:g}"
        )

    return "; ".join(parts)


__all__ = [
    "ACField",
    "MoleculeEnvironment",
    "StateLabel",
    "StickSpectrum",
    "calculate_stick_spectrum",
    "format_transition_table",
    "label_to_indices",
    "plot_stick_spectrum",
    "print_transition_table",
    "visible_transition_indices",
]
