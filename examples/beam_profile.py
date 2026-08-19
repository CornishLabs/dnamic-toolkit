"""Run the beam-profile analysis from Python.

Equivalent command-line use:

    uv run dnamic-beam-profile PATH/TO/TIFF_FOLDER --centre-x 671 --centre-y 786

The TIFF filenames should have numeric stems, such as ``-2.0.tif``, ``0.tif``
and ``2.0.tif``. Those numbers are used as the propagation coordinate.
"""

from pathlib import Path

from dnamic_toolkit.tools.beam_profile import (
    BeamProfileSettings,
    analyze_beam_profiles,
)


data_folder = Path("PATH/TO/TIFF_FOLDER")

result = analyze_beam_profiles(
    BeamProfileSettings(
        folder=data_folder,
        centre_x=671,
        centre_y=786,
        fit_half_size=180,
        fit_stride=1,
        pixel_size=1.0,
        waist_unit="pixel",
        distance_unit="filename unit",
        robust=False,
    )
)

print(f"results CSV: {result.results_csv}")
print(f"principal-axis waist plot: {result.waists_principal_axes_plot}")
print(f"successful common-axis fits: {result.successful_common_fits}/{len(result.records)}")
