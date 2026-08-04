# dnamic-toolkit
Contains common helper functions, physics calculations, for use in DNAMIC (c)ontrolled labs.

## Gaussian tweezer properties

The lightweight analytical model can turn per-tweezer optical power and measured
waists into centre intensity, trap depth, and local harmonic trap frequencies:

```python
from dnamic_toolkit.calibrations.cs_1066_power import (
    cs_1066_test_point_power_w,
)
from dnamic_toolkit.physics.gaussian_tweezers import (
    gaussian_tweezer_properties,
)

test_point_power_w = cs_1066_test_point_power_w(5.65)
power_per_tweezer_w = test_point_power_w * 0.43 / 9

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

## Plotting styles and colours

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
uv sync # This updates the venv associated with this folder
```

This creates/updates the project’s `.venv` and installs the project in editable mode in this project venv for development.

### Run the tests

```bash
uv run pytest
```

`uv run` executes commands inside the project environment (it syncs before if necessary).

### Try a quick import

```bash
uv run python -c "import dnamic_toolkit; print('import ok')"
```

(`src/` contains the package and `tests/` contains the test suite.)

### Run an example

```bash
uv run python examples/<example_file>.py
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
