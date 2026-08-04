"""Small axes-level helpers for Matplotlib plots."""

from __future__ import annotations

import colorsys

from matplotlib import colors as matplotlib_colors


__all__ = ["errorbar_scatter", "adjust_lightness"]


_DEFAULT_ERRORBAR_KWARGS = {
    "marker": "o",
    "linestyle": "none",
    "linewidth": 1,
    "capsize": 1,
}


def adjust_lightness(color_like, amount: float = 1.0) -> str:
    """Return ``color_like`` with HLS lightness scaled by ``amount``.

    ``amount=1`` leaves the color unchanged; values below one darken it and
    values above one lighten it.
    """

    if amount < 0:
        raise ValueError("amount must not be negative")

    rgba = matplotlib_colors.to_rgba(color_like)
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgba[:3])
    adjusted_rgb = colorsys.hls_to_rgb(
        hue,
        max(0.0, min(1.0, amount * lightness)),
        saturation,
    )
    return matplotlib_colors.to_hex((*adjusted_rgb, rgba[3]), keep_alpha=rgba[3] < 1.0)


def errorbar_scatter(
    ax,
    x,
    y,
    *,
    xerr=None,
    yerr=None,
    color=None,
    hollow: bool = False,
    outline_lightness: float = 0.75,
    face_lightness: float = 1.0,
    **kwargs,
):
    """Draw marker-only points with optional error bars.

    This is a light aesthetic wrapper around ``ax.errorbar``. Use plain
    ``ax.errorbar`` whenever these defaults are not helpful.
    """

    errorbar_kwargs = {**_DEFAULT_ERRORBAR_KWARGS, **kwargs}
    if color is not None:
        edge_color = adjust_lightness(color, outline_lightness)
        face_color = "none" if hollow else adjust_lightness(color, face_lightness)
        errorbar_kwargs.setdefault("color", face_color)
        errorbar_kwargs.setdefault("markerfacecolor", face_color)
        errorbar_kwargs.setdefault("markeredgecolor", edge_color)
        errorbar_kwargs.setdefault("ecolor", edge_color)
    elif hollow:
        errorbar_kwargs.setdefault("markerfacecolor", "none")

    return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kwargs)
