# dnamic-toolkit
Contains common helper functions, physics calculations, for use in DNAMIC (c)ontrolled labs.

The default installation contains the generic NumPy/SciPy physics and statistics
helpers. Install only the capability extras needed by a particular workflow:

```bash
uv sync --extra plotting       # Matplotlib display helpers and styles
uv sync --extra alkali         # ARC alkali/Rydberg calculations
uv sync --extra molecules      # diatomic-py spectrum tools and plotting
uv sync --extra hdf5           # HDF5 result helpers
uv sync --extra beam-profile   # TIFF beam-profile command and plotting
```

## Gaussian tweezer properties

The lightweight analytical model can turn per-tweezer optical power and measured
waists into centre intensity, trap depth, and local harmonic trap frequencies:

```python
from dnamic_toolkit.physics.gaussian_tweezers import gaussian_tweezer_properties

# Apparatus-specific calibration records live in the experiment repository. Use
# the calibrated optical power from that lab record as the input here.
power_per_tweezer_w = 0.14767824386100453 * 0.43 / 9

trap = gaussian_tweezer_properties(
    power_w=power_per_tweezer_w,
    wavelength_m=1066e-9,
    waist_x_m=1.05e-6,
    waist_y_m=1.16e-6,
    axial_waist_m=1.19e-6,
    polarizability_au=1168,
    mass_amu=133,
)

print(trap.centre_intensity_kw_cm2)
print(trap.trap_depth_hz / 1e6, "MHz")
print(trap.radial_x_frequency_hz / 1e3, "kHz")
print(trap.radial_y_frequency_hz / 1e3, "kHz")
print(trap.axial_frequency_hz / 1e3, "kHz")
```

These are the frequencies from the curvature at the bottom of the trap. Warm
atoms and quadratic fits over a finite region sample the Gaussian anharmonicity
and can consequently give somewhat lower frequencies.

## Alkali polarizabilities

This module requires the `alkali` extra. The plotting example also requires
`plotting`.

Rb/Cs dynamic polarizabilities can be calculated with ARC, or loaded from portal
CSV exports that contain `alpha_0`, `alpha_2`, and their uncertainties:

```python
import numpy as np

from dnamic_toolkit.physics.alkali_polarizability import (
    AlkaliState,
    arc_polarizability,
    dynamic_polarizability_from_components,
    interpolate_portal_polarizability,
    load_portal_polarizability_folder,
    portal_table,
)

state = AlkaliState.from_label("Rb", "5p3/2")
wavelength_nm = np.linspace(600, 1200, 50)

portal_tables = load_portal_polarizability_folder("~/Downloads/Rb1Pol/Rb1Pol")
portal_components = interpolate_portal_polarizability(
    portal_table(portal_tables, "Rb", "5p3"),
    wavelength_nm,
)
portal_alpha = dynamic_polarizability_from_components(portal_components)

arc_components = arc_polarizability(state, wavelength_nm, n_max=30)
arc_alpha = dynamic_polarizability_from_components(arc_components)
```

See `examples/alkali_polarizability.py` for an Rb/Cs plotting example.

## Beam profiling

Install the `beam-profile` extra before using the command or Python module.

The beam-profiling tool fits numeric-named TIFF images with 2D Gaussians, writes
per-image fit CSVs, and fits the measured radii to Gaussian-beam propagation.
Use numeric TIFF stems for the propagation coordinate, for example
`-2.0.tif`, `0.0.tif`, and `2.0.tif`.

The radius plots show the per-image 1σ fit errors. Those errors are used as
absolute uncertainties in the Gaussian-beam propagation fit, and the shaded
region around each propagation curve is its propagated 1σ uncertainty.

`--pixel-size` must be the object-plane size represented by one array pixel in
the TIFF, in the unit named by `--waist-unit`. Include camera binning and imaging
magnification; for example, use `sensor_pixel_pitch * binning / magnification`.
The numeric TIFF stems must already be expressed in `--distance-unit`.

From the command line:

```bash
uv run dnamic-beam-profile PATH/TO/TIFF_FOLDER --centre-x 671 --centre-y 786
```

Useful options include `--output`, `--fit-half-size`, `--fit-stride`,
`--pixel-size`, `--waist-unit`, `--distance-unit`, and `--robust`.

From Python:

```python
from pathlib import Path

from dnamic_toolkit.tools.beam_profile import (
    BeamProfileSettings,
    analyze_beam_profiles,
)

result = analyze_beam_profiles(
    BeamProfileSettings(
        folder=Path("PATH/TO/TIFF_FOLDER"),
        centre_x=671,
        centre_y=786,
    )
)
print(result.results_csv)
```

See `examples/beam_profile.py` for a copy-editable script.

## Plotting styles and colours

The styles and colour definitions are packaged in the base installation. The
Matplotlib helpers require the `plotting` extra.

Use the bundled Matplotlib styles directly:

```python
import matplotlib.pyplot as plt

plt.style.use("dnamic_toolkit.display.styles.tweezer_lab")
```

Use named colours or palettes when code needs them explicitly:

```python
from dnamic_toolkit.display.colors import color, colors, palette_names
from dnamic_toolkit.display.helpers import errorbar_scatter

print(palette_names())
print(colors("tol_bright"))

fig, ax = plt.subplots()
errorbar_scatter(ax, x, y, yerr=yerr, color=color("durham", "purple"))
```

Format values with uncertainties using the same rounding rule in notebooks,
scripts, and plot labels:

```python
from dnamic_toolkit.display.formatting import format_uncertainty

format_uncertainty(0.031, 0.0099)
# "0.03(1)"

format_uncertainty(0.031, 0.0099, style="latex")
# "0.03 \\pm 0.01"
```

## Installing UV (the modern replacement for conda, pip, virtualenv, piptools,...)

Follow Astral’s install guide: https://docs.astral.sh/uv/getting-started/installation/ 

## Using `dnamic-toolkit` from another `uv` project without installing editably

From your *other* project directory:

```bash
uv add "dnamic-toolkit @ git+https://github.com/CornishLabs/dnamic-toolkit.git"
uv sync
```

For having an install, and being able to edit it, see below.

## Quickstart to developing + using simultaneously

### Clone + create the project environment
```bash
git clone https://github.com/CornishLabs/dnamic-toolkit.git
cd dnamic-toolkit
uv sync --all-extras # Install every capability for development and tests
```

This creates/updates the project’s `.venv` and installs the project in editable mode in this project venv for development.

### Run the tests

```bash
uv run --all-extras pytest
```

`uv run` executes commands inside the project environment (it syncs before if necessary).

### Try a quick import

```bash
uv run python -c "import dnamic_toolkit; print('import ok')"
```

(`src/` contains the package and `tests/` contains the test suite.)

### Run an example

```bash
uv run --all-extras python examples/<example_file>.py
```

(See the `examples/` folder for runnable scripts.)

---

## Use this editable install in another project setup with UV
```bash
uv add --editable /path/to/cloned/dnamic-toolkit
uv sync
```

---

## Notes for contributors

* Add runtime deps:

  ```bash
  uv add <package>
  ```

* Add dev deps (tests/lint tooling). The `dev` group is installed by default:

  ```bash
  uv add --dev pytest
  ```

* CI/repro builds: fail if `uv.lock` would change:

  ```bash
  uv sync --locked
  ```

  or

  ```bash
  uv run --locked pytest
  ```
