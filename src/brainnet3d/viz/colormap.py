"""
Color mapping utilities: numeric arrays and categorical labels → RGB tuples.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RGB = tuple[float, float, float]


def values_to_colors(
    values: np.ndarray,
    cmap:   str = "viridis",
    vmin:   float | None = None,
    vmax:   float | None = None,
) -> list[RGB]:
    """Map a 1-D array of numeric values to a list of (R, G, B) tuples in [0,1]."""
    values = np.asarray(values, dtype=float)
    vmin   = float(np.nanmin(values)) if vmin is None else vmin
    vmax   = float(np.nanmax(values)) if vmax is None else vmax

    if vmax == vmin:
        normed = np.full_like(values, 0.5)
    else:
        normed = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    cm = plt.get_cmap(cmap)
    return [cm(v)[:3] for v in normed]


def labels_to_colors(
    labels: list[str],
    cmap:   str = "Set3",
    palette: dict | None = None,
) -> list[RGB]:
    """Map a list of string labels to (R, G, B) tuples."""
    if palette is not None:
        return [_to_rgb(palette.get(lbl, "grey")) for lbl in labels]

    unique = list(dict.fromkeys(labels))
    cm     = plt.get_cmap(cmap)
    color_map = {lbl: cm(i % cm.N)[:3] for i, lbl in enumerate(unique)}
    return [color_map[lbl] for lbl in labels]


def edge_colors_from_weights(
    weights: np.ndarray,
    cmap:    str = "RdBu_r",
    vmin:    float | None = None,
    vmax:    float | None = None,
) -> list[RGB]:
    return values_to_colors(weights, cmap=cmap, vmin=vmin, vmax=vmax)


def edge_colors_from_node_colors(
    src_indices: list[int],
    node_colors: list[RGB],
) -> list[RGB]:
    return [node_colors[i] for i in src_indices]


def values_to_sizes(
    values: np.ndarray,
    size_range: tuple[float, float] = (2.0, 8.0),
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    if vmax == vmin:
        return np.full_like(values, (size_range[0] + size_range[1]) / 2.0)

    normed = (values - vmin) / (vmax - vmin)
    return size_range[0] + normed * (size_range[1] - size_range[0])


def values_to_widths(
    values: np.ndarray,
    width_range: tuple[float, float] = (0.5, 4.0),
) -> np.ndarray:
    return values_to_sizes(values, size_range=width_range)


def _to_rgb(color) -> RGB:
    return mcolors.to_rgba(color)[:3]
