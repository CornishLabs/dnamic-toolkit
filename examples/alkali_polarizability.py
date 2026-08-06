"""Compare Rb dynamic polarizabilities from UDEL portal data and ARC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dnamic_toolkit.display.colors import color
from dnamic_toolkit.physics.alkali_polarizability import (
    AlkaliState,
    arc_polarizability,
    dynamic_polarizability_from_components,
    dynamic_polarizability_uncertainty_au,
    interpolate_portal_polarizability,
    load_portal_polarizability_folder,
    portal_table,
)


RB_STATES = ("5s1/2", "5p1/2", "5p3/2")
DEFAULT_UDEL_DIR = Path.home() / "Downloads" / "Rb1Pol" / "Rb1Pol"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-nm", type=float, default=600.0)
    parser.add_argument("--max-nm", type=float, default=2000.0)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--arc-n-max", type=int, default=30)
    parser.add_argument("--theta-deg", type=float, default=10.0)
    parser.add_argument("--m-j", type=float)
    parser.add_argument("--udel-dir", type=Path, default=DEFAULT_UDEL_DIR)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    udel_tables = load_portal_polarizability_folder(args.udel_dir)

    plt.style.use("dnamic_toolkit.display.styles.tweezer_lab")
    fig, axes = plt.subplots(
        len(RB_STATES),
        1,
        figsize=(8, 8),
        sharex=True,
        constrained_layout=True,
    )

    for ax, state_label in zip(axes, RB_STATES, strict=True):
        state = AlkaliState.from_label("Rb", state_label)
        udel_table = portal_table(udel_tables, "Rb", state)
        low_nm = max(args.min_nm, udel_table.wavelength_range_nm[0])
        high_nm = min(args.max_nm, udel_table.wavelength_range_nm[1])
        wavelength_nm = np.linspace(low_nm, high_nm, args.points)
        m_j = state.j if args.m_j is None else args.m_j

        udel_components = interpolate_portal_polarizability(udel_table, wavelength_nm)
        udel_alpha = dynamic_polarizability_from_components(
            udel_components,
            m_j=m_j,
            theta_p_deg=args.theta_deg,
        )
        udel_uncertainty = dynamic_polarizability_uncertainty_au(
            udel_components.alpha0_uncertainty_au,
            udel_components.alpha2_uncertainty_au,
            j=state.j,
            m_j=m_j,
            theta_p_deg=args.theta_deg,
        )

        arc_components = arc_polarizability(
            state,
            wavelength_nm,
            n_max=args.arc_n_max,
        )
        arc_alpha = dynamic_polarizability_from_components(
            arc_components,
            m_j=m_j,
            circularity=0.0,
            theta_p_deg=args.theta_deg,
        )

        ax.axhline(0, color="0.82", linewidth=0.8)
        ax.plot(
            wavelength_nm,
            udel_alpha,
            color=color("tol_bright", "blue"),
            label="UDEL",
        )
        ax.fill_between(
            wavelength_nm,
            udel_alpha - udel_uncertainty,
            udel_alpha + udel_uncertainty,
            color=color("tol_bright", "blue"),
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            wavelength_nm,
            arc_alpha,
            color=color("tol_bright", "red"),
            linestyle="--",
            label="ARC",
        )
        ax.set_title(rf"Rb {state.label}, $m_j={m_j:g}$")
        ax.set_ylabel(r"$\alpha$ (a.u.)")
        ax.set_yscale("symlog", linthresh=500)
        ax.legend()

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        rf"Rb dynamic polarizability, linear light, $\theta={args.theta_deg:g}^\circ$"
    )

    if args.save is not None:
        fig.savefig(args.save, dpi=200)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

