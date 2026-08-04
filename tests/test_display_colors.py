import matplotlib.pyplot as plt
import pytest

from dnamic_toolkit.display.colors import (
    color,
    colors,
    palette,
    palette_names,
)


def test_palettes_have_names_and_colours():
    assert palette_names() == (
        "durham",
        "tol_bright",
        "tol_muted",
        "tol_high_contrast",
    )
    assert palette("durham")["purple"] == "#68246D"
    assert tuple(palette("tol_bright"))[:2] == ("blue", "red")
    assert colors("tol_bright")[:2] == ("#4477AA", "#EE6677")
    assert color("tol_bright", "blue") == "#4477AA"


def test_colour_lookup_uses_exact_names():
    with pytest.raises(KeyError):
        palette("tol.bright")
    with pytest.raises(KeyError):
        color("durham", "Purple")


def test_package_style_name_loads_through_matplotlib():
    plt.style.use("default")
    plt.style.use("dnamic_toolkit.display.styles.tweezer_lab")

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle[0].lower() == "#4477aa"
    assert tuple(color.upper() for color in cycle) == colors("tol_bright")


def test_paper_style_uses_supported_palette():
    plt.style.use("default")
    plt.style.use("dnamic_toolkit.display.styles.tweezer_paper")

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert tuple(color.upper() for color in cycle) == colors("tol_muted")
