"""Simple Rb/Cs dynamic polarizability helpers.

All wavelengths are in nm and polarizabilities are in atomic units.  ARC gives
``alpha0``, ``alpha1`` and ``alpha2`` components; portal CSVs give ``alpha0``
and ``alpha2`` plus uncertainties.  Use :func:`dynamic_polarizability_au` to
combine those components for a chosen ``m_j`` and light polarization.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from fractions import Fraction
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np
from arc import Caesium, DynamicPolarizability, Rubidium85, Rubidium87
from numpy.typing import ArrayLike, NDArray


_L_BY_LETTER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
_LETTER_BY_L = {value: key for key, value in _L_BY_LETTER.items()}
_STATE_RE = re.compile(r"^\s*(?P<n>\d+)\s*(?P<l>[spdfghSPDFGH])\s*(?P<j>[0-9./]+)\s*$")
_PORTAL_SPECIES_RE = re.compile(r"^\s*(?P<species>[A-Z][a-z]?)(?P<ion_stage>\d*)\s*$")


@dataclass(frozen=True)
class AlkaliState:
    """Fine-structure state used by the polarizability helpers."""

    species: str
    n: int
    l: int
    j: float
    isotope: int | None = None

    @classmethod
    def from_label(
        cls,
        species: str,
        label: str,
        *,
        isotope: int | None = None,
    ) -> "AlkaliState":
        """Build a state from labels such as ``"5s1/2"`` or ``"5p3"``."""
        match = _STATE_RE.match(label)
        l_letter = match.group("l").lower()
        return cls(
            species=normalise_species(species),
            n=int(match.group("n")),
            l=_L_BY_LETTER[l_letter],
            j=_parse_j(match.group("j")),
            isotope=isotope,
        )

    @property
    def label(self) -> str:
        return state_label(self.n, self.l, self.j)

    @property
    def display_label(self) -> str:
        isotope = "" if self.isotope is None else str(self.isotope)
        return f"{self.species}{isotope} {self.label}"


@dataclass(frozen=True)
class PortalPolarizabilityTable:
    """One portal CSV export for one atom and one fine-structure state."""

    state: AlkaliState
    wavelength_nm: NDArray[np.float64]
    alpha0_au: NDArray[np.float64]
    alpha0_uncertainty_au: NDArray[np.float64]
    alpha2_au: NDArray[np.float64]
    alpha2_uncertainty_au: NDArray[np.float64]
    source_path: Path | None = None
    ion_stage: int | None = None

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        return float(self.wavelength_nm[0]), float(self.wavelength_nm[-1])


@dataclass(frozen=True)
class PolarizabilityComponents:
    """Polarizability components on a wavelength grid."""

    state: AlkaliState
    wavelength_nm: NDArray[np.float64]
    alpha0_au: NDArray[np.float64]
    alpha1_au: NDArray[np.float64]
    alpha2_au: NDArray[np.float64]
    alpha0_uncertainty_au: NDArray[np.float64] | None = None
    alpha2_uncertainty_au: NDArray[np.float64] | None = None
    alpha0_valence_au: NDArray[np.float64] | None = None
    alpha0_core_au: NDArray[np.float64] | None = None
    alpha_ponderomotive_au: NDArray[np.float64] | None = None
    closest_state: tuple[Any, ...] = ()


def dynamic_polarizability_au(
    alpha0_au: ArrayLike,
    alpha1_au: ArrayLike | float = 0.0,
    alpha2_au: ArrayLike | float = 0.0,
    *,
    j: float,
    m_j: float,
    circularity: float = 0.0,
    theta_k_deg: float = 0.0,
    theta_p_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Combine scalar, vector and tensor polarizability components.

    ``circularity`` is 0 for linearly polarised light and +/-1 for pure
    circular light.  ``theta_k_deg`` is the angle between propagation and the
    quantization axis.  ``theta_p_deg`` is the angle between linear
    polarization and the quantization axis.
    """
    alpha0 = np.asarray(alpha0_au, dtype=float)
    alpha1 = np.asarray(alpha1_au, dtype=float)
    alpha2 = np.asarray(alpha2_au, dtype=float)

    vector = circularity * cos(radians(theta_k_deg)) * (m_j / j) * alpha1
    tensor = tensor_prefactor(j=j, m_j=m_j, theta_p_deg=theta_p_deg) * alpha2
    return alpha0 + vector + tensor


def dynamic_polarizability_from_components(
    components: PolarizabilityComponents,
    *,
    m_j: float | None = None,
    circularity: float = 0.0,
    theta_k_deg: float = 0.0,
    theta_p_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Combine a :class:`PolarizabilityComponents` object."""
    if m_j is None:
        m_j = components.state.j
    return dynamic_polarizability_au(
        components.alpha0_au,
        components.alpha1_au,
        components.alpha2_au,
        j=components.state.j,
        m_j=m_j,
        circularity=circularity,
        theta_k_deg=theta_k_deg,
        theta_p_deg=theta_p_deg,
    )


def dynamic_polarizability_uncertainty_au(
    alpha0_uncertainty_au: ArrayLike,
    alpha2_uncertainty_au: ArrayLike | float = 0.0,
    *,
    j: float,
    m_j: float,
    theta_p_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Propagate independent scalar/tensor portal uncertainties."""
    alpha0_uncertainty = np.asarray(alpha0_uncertainty_au, dtype=float)
    alpha2_uncertainty = np.asarray(alpha2_uncertainty_au, dtype=float)
    weight = tensor_prefactor(j=j, m_j=m_j, theta_p_deg=theta_p_deg)
    return np.sqrt(alpha0_uncertainty**2 + (weight * alpha2_uncertainty) ** 2)


def tensor_prefactor(
    *,
    j: float,
    m_j: float,
    theta_p_deg: float = 0.0,
) -> float:
    """Multiplier applied to ``alpha2`` for linearly polarised light."""
    if j <= 0.5:
        return 0.0
    geometry = 0.5 * (3 * cos(radians(theta_p_deg)) ** 2 - 1)
    sublevel = (3 * m_j**2 - j * (j + 1)) / (j * (2 * j - 1))
    return geometry * sublevel


def arc_polarizability(
    state: AlkaliState,
    wavelength_nm: ArrayLike,
    *,
    n_min: int | None = None,
    n_max: int = 30,
    include_core: bool = True,
    account_for_state_lifetime: bool = False,
    prefer_quantum_defects: bool = True,
    cpp_numerov: bool = True,
) -> PolarizabilityComponents:
    """Calculate ARC ``alpha0``, ``alpha1`` and ``alpha2`` components."""
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    atom = arc_atom(
        state.species,
        isotope=state.isotope,
        prefer_quantum_defects=prefer_quantum_defects,
        cpp_numerov=cpp_numerov,
    )
    if n_min is None:
        n_min = min(state.n, int(atom.groundStateN))

    calculator = DynamicPolarizability(atom=atom, n=state.n, l=state.l, j=state.j)
    calculator.defineBasis(n_min, n_max)

    alpha0_valence = np.empty_like(wavelength)
    alpha1 = np.empty_like(wavelength)
    alpha2 = np.empty_like(wavelength)
    alpha0_core = np.empty_like(wavelength)
    alpha_ponderomotive = np.empty_like(wavelength)
    closest_states = []

    for index, wavelength_value_nm in enumerate(wavelength):
        (
            alpha0_valence[index],
            alpha1[index],
            alpha2[index],
            alpha0_core[index],
            alpha_ponderomotive[index],
            closest_state,
        ) = calculator.getPolarizability(
            wavelength_value_nm * 1e-9,
            units="a.u.",
            accountForStateLifetime=account_for_state_lifetime,
        )
        closest_states.append(closest_state)

    alpha0 = alpha0_valence + alpha0_core if include_core else alpha0_valence
    return PolarizabilityComponents(
        state=state,
        wavelength_nm=wavelength,
        alpha0_au=alpha0,
        alpha1_au=alpha1,
        alpha2_au=alpha2,
        alpha0_valence_au=alpha0_valence,
        alpha0_core_au=alpha0_core,
        alpha_ponderomotive_au=alpha_ponderomotive,
        closest_state=tuple(closest_states),
    )


def arc_dynamic_polarizability(
    state: AlkaliState,
    wavelength_nm: ArrayLike,
    *,
    m_j: float | None = None,
    circularity: float = 0.0,
    theta_k_deg: float = 0.0,
    theta_p_deg: float = 0.0,
    **arc_kwargs: Any,
) -> NDArray[np.float64]:
    """Calculate ARC components and immediately combine them."""
    components = arc_polarizability(state, wavelength_nm, **arc_kwargs)
    return dynamic_polarizability_from_components(
        components,
        m_j=m_j,
        circularity=circularity,
        theta_k_deg=theta_k_deg,
        theta_p_deg=theta_p_deg,
    )


def arc_atom(
    species: str,
    *,
    isotope: int | None = None,
    prefer_quantum_defects: bool = True,
    cpp_numerov: bool = True,
) -> Any:
    """Return the ARC atom object for Rb or Cs."""
    atom_class = {
        ("Rb", None): Rubidium87,
        ("Rb", 87): Rubidium87,
        ("Rb", 85): Rubidium85,
        ("Cs", None): Caesium,
        ("Cs", 133): Caesium,
    }[(normalise_species(species), isotope)]
    return atom_class(
        preferQuantumDefects=prefer_quantum_defects,
        cpp_numerov=cpp_numerov,
    )


def load_portal_polarizability_csv(path: str | Path) -> PortalPolarizabilityTable:
    """Load one portal CSV export."""
    source_path = Path(path).expanduser()
    with source_path.open(newline="") as handle:
        metadata = next(csv.reader(handle))

    species, ion_stage = _parse_portal_species(metadata[0])
    state = AlkaliState.from_label(species, metadata[1].strip())
    data = np.genfromtxt(source_path, delimiter=",", skip_header=2)
    data = np.atleast_2d(data)
    data = data[np.argsort(data[:, 0])]

    return PortalPolarizabilityTable(
        state=state,
        wavelength_nm=data[:, 0],
        alpha0_au=data[:, 1],
        alpha0_uncertainty_au=data[:, 2],
        alpha2_au=data[:, 3],
        alpha2_uncertainty_au=data[:, 4],
        source_path=source_path,
        ion_stage=ion_stage,
    )


def load_portal_polarizability_folder(
    folder: str | Path,
) -> dict[tuple[str, str], PortalPolarizabilityTable]:
    """Load every portal CSV in ``folder`` as ``(species, state_label) -> table``."""
    folder_path = Path(folder).expanduser()
    tables = [load_portal_polarizability_csv(path) for path in folder_path.glob("*.csv")]
    return {(table.state.species, table.state.label): table for table in tables}


def portal_table(
    tables: dict[tuple[str, str], PortalPolarizabilityTable],
    species: str,
    state: str | AlkaliState,
) -> PortalPolarizabilityTable:
    """Pick one table from a folder loaded by ``load_portal_polarizability_folder``."""
    return tables[_table_key(species, state)]


def portal_states(
    tables: dict[tuple[str, str], PortalPolarizabilityTable],
    species: str | None = None,
) -> tuple[AlkaliState, ...]:
    """List the states in a loaded portal-table dictionary."""
    states = [table.state for table in tables.values()]
    if species is not None:
        species = normalise_species(species)
        states = [state for state in states if state.species == species]
    return tuple(sorted(states, key=lambda state: (state.species, state.n, state.l, state.j)))


def interpolate_portal_polarizability(
    table: PortalPolarizabilityTable,
    wavelength_nm: ArrayLike,
) -> PolarizabilityComponents:
    """Interpolate portal components onto ``wavelength_nm``."""
    wavelength = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    alpha1 = np.zeros_like(wavelength)
    return PolarizabilityComponents(
        state=table.state,
        wavelength_nm=wavelength,
        alpha0_au=np.interp(wavelength, table.wavelength_nm, table.alpha0_au),
        alpha1_au=alpha1,
        alpha2_au=np.interp(wavelength, table.wavelength_nm, table.alpha2_au),
        alpha0_uncertainty_au=np.interp(
            wavelength,
            table.wavelength_nm,
            table.alpha0_uncertainty_au,
        ),
        alpha2_uncertainty_au=np.interp(
            wavelength,
            table.wavelength_nm,
            table.alpha2_uncertainty_au,
        ),
    )


def portal_dynamic_polarizability(
    table: PortalPolarizabilityTable,
    wavelength_nm: ArrayLike,
    *,
    m_j: float | None = None,
    theta_p_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Interpolate portal components and immediately combine them."""
    components = interpolate_portal_polarizability(table, wavelength_nm)
    return dynamic_polarizability_from_components(
        components,
        m_j=m_j,
        theta_p_deg=theta_p_deg,
    )


def state_label(n: int, l: int, j: float) -> str:
    """Format quantum numbers as a compact fine-structure label."""
    j_fraction = Fraction(j).limit_denominator(12)
    j_text = (
        str(j_fraction.numerator)
        if j_fraction.denominator == 1
        else f"{j_fraction.numerator}/{j_fraction.denominator}"
    )
    return f"{n}{_LETTER_BY_L[l]}{j_text}"


def normalise_species(species: str) -> str:
    """Normalise species names to ``"Rb"`` or ``"Cs"``."""
    cleaned = species.strip().lower()
    if cleaned in {"rb", "rubidium"}:
        return "Rb"
    if cleaned in {"cs", "cesium", "caesium"}:
        return "Cs"
    return species


def _parse_j(text: str) -> float:
    if "/" in text:
        return float(Fraction(text))
    if "." in text:
        return float(text)
    return int(text) / 2


def _parse_portal_species(text: str) -> tuple[str, int | None]:
    match = _PORTAL_SPECIES_RE.match(text)
    ion_stage_text = match.group("ion_stage")
    ion_stage = int(ion_stage_text) if ion_stage_text else None
    return normalise_species(match.group("species")), ion_stage


def _table_key(
    species: str,
    state: str | AlkaliState,
) -> tuple[str, str]:
    species = normalise_species(species)
    if isinstance(state, AlkaliState):
        return species, state.label
    return species, AlkaliState.from_label(species, state).label


effective_polarizability_au = dynamic_polarizability_au

