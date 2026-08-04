import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dnamic_toolkit.display.colors import color
from dnamic_toolkit.display.helpers import errorbar_scatter


def test_errorbar_scatter_draws_on_given_axes():
    fig, ax = plt.subplots()

    artists = errorbar_scatter(
        ax,
        [0, 1],
        [2, 3],
        yerr=[0.1, 0.2],
        color=color("tol_bright", "blue"),
    )

    assert artists.lines[0].get_marker() == "o"
    assert artists.lines[0] in ax.lines
    plt.close(fig)


def test_errorbar_scatter_can_use_hollow_markers():
    fig, ax = plt.subplots()

    artists = errorbar_scatter(
        ax,
        [0],
        [1],
        color=color("tol_bright", "blue"),
        hollow=True,
    )

    assert artists.lines[0].get_markerfacecolor() == "none"
    plt.close(fig)


def test_display_import_is_inert():
    plt.style.use("default")
    before = {
        "font.size": plt.rcParams["font.size"],
        "figure.figsize": list(plt.rcParams["figure.figsize"]),
        "axes.prop_cycle": plt.rcParams["axes.prop_cycle"],
    }

    import dnamic_toolkit.display as display

    assert display.__doc__
    assert plt.rcParams["font.size"] == before["font.size"]
    assert list(plt.rcParams["figure.figsize"]) == before["figure.figsize"]
    assert plt.rcParams["axes.prop_cycle"] == before["axes.prop_cycle"]
