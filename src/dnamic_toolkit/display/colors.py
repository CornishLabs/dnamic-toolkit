"""Small, shared colour palettes for lab plots."""

from __future__ import annotations


PALETTES = {
    "durham": {
        "purple": "#68246D",
        "yellow": "#FFD53A",
        "cyan": "#00AEEF",
        "red": "#BE1E2D",
        "gold": "#AFA961",
        "heather": "#CBA8B1",
        "stone": "#DACDA2",
        "sky": "#A5C8D0",
        "cedar": "#B6AAA7",
        "concrete": "#B3BDB1",
    },
    "tol_bright": {
        "blue": "#4477AA",
        "red": "#EE6677",
        "green": "#228833",
        "yellow": "#CCBB44",
        "cyan": "#66CCEE",
        "purple": "#AA3377",
        "grey": "#BBBBBB",
    },
    "tol_muted": {
        "rose": "#CC6677",
        "indigo": "#332288",
        "sand": "#DDCC77",
        "green": "#117733",
        "cyan": "#88CCEE",
        "wine": "#882255",
        "teal": "#44AA99",
        "olive": "#999933",
        "purple": "#AA4499",
        "pale_grey": "#DDDDDD",
    },
    "tol_high_contrast": {
        "blue": "#004488",
        "yellow": "#DDAA33",
        "red": "#BB5566",
    },
}

__all__ = [
    "PALETTES",
    "color",
    "colors",
    "palette",
    "palette_names",
]


def palette_names() -> tuple[str, ...]:
    """Return the names of the available palettes."""

    return tuple(PALETTES)


def palette(name: str) -> dict[str, str]:
    """Return one palette as a copy of its ordered ``name -> colour`` mapping."""

    try:
        return dict(PALETTES[name])
    except KeyError as error:
        raise KeyError(_unknown_palette_message(name)) from error


def colors(palette_name: str) -> tuple[str, ...]:
    """Return the colours in one palette as an ordered tuple of hex strings."""

    return tuple(palette(palette_name).values())


def color(palette_name: str, color_name: str) -> str:
    """Return one colour by exact palette name and exact colour name."""

    palette_data = palette(palette_name)
    try:
        return palette_data[color_name]
    except KeyError as error:
        available = ", ".join(palette_data)
        raise KeyError(
            f"Unknown colour {color_name!r} in palette {palette_name!r}. "
            f"Available colours: {available}"
        ) from error


def _unknown_palette_message(name: str) -> str:
    available = ", ".join(PALETTES)
    return f"Unknown palette {name!r}. Available palettes: {available}"
