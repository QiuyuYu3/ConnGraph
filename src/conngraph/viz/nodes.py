"""
Node rendering: positions + attributes → list of vedo Sphere objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brainnet3d.viz.colormap import (
    values_to_colors,
    labels_to_colors,
    values_to_sizes,
    _to_rgb,
)


def build_nodes(
    nodes_df:       pd.DataFrame,
    node_size:      str | float = 3.0,
    node_size_range: tuple[float, float] = (2.0, 8.0),
    node_color:     str | tuple = "network",
    node_cmap:      str | None = None,
    node_colorvminvmax: str | tuple | None = "minmax",
    node_alpha:     float = 1.0,
    node_res:       int   = 16,
    palette:        dict | None = None,
    positions:      np.ndarray | None = None,
) -> list:
    """
    Build vedo Sphere objects for all nodes.

    Parameters
    ----------
    nodes_df : DataFrame with columns label, x, y, z and optionally
               network, hemisphere, + any numeric columns.
    node_size : float → uniform radius. str → column in nodes_df rescaled to node_size_range.
    node_size_range : (min_r, max_r) in mm.
    node_color : RGB tuple or colour name → uniform. "network" / column name → categorical.
                 Numeric column → continuous colormap.
    node_cmap : matplotlib colormap name. None → "viridis" for numeric columns;
                "Set3" for up to 12 categories, else "tab20".
    node_colorvminvmax : colour scale for numeric node_color.
                         "minmax" (default) — maps min→max of the data.
                         "absmax"           — symmetric around 0: ±max(|values|).
                         (vmin, vmax) tuple — explicit limits.
    node_alpha : transparency 0–1.
    node_res : sphere tessellation resolution.
    palette : optional {label: color} dict for categorical colouring.
    positions : (N, 3) array overriding the x, y, z columns (e.g. a graph layout).

    Returns
    -------
    List of vedo Sphere objects.
    """
    from vedo import Sphere

    if positions is None:
        positions = nodes_df[["x", "y", "z"]].values
    positions = np.asarray(positions, dtype=float)
    n         = len(positions)

    radii  = _resolve_sizes(node_size, node_size_range, nodes_df, n)
    colors = _resolve_colors(node_color, node_cmap, node_colorvminvmax, nodes_df, n, palette)

    spheres = []
    for idx, (pos, r, c) in enumerate(zip(positions, radii, colors)):
        s = Sphere(pos=pos, r=float(r), c=c, res=node_res, alpha=node_alpha)
        s.lighting("ambient")
        s._node_idx = idx
        spheres.append(s)

    return spheres


def _resolve_sizes(
    node_size,
    node_size_range: tuple[float, float],
    nodes_df: pd.DataFrame,
    n: int,
) -> np.ndarray:

    if isinstance(node_size, (int, float)):
        return np.full(n, float(node_size))

    if isinstance(node_size, str):
        if node_size not in nodes_df.columns:
            raise ValueError(
                f"node_size='{node_size}' not found in nodes_df columns: "
                f"{list(nodes_df.columns)}"
            )
        return values_to_sizes(nodes_df[node_size].values, node_size_range)

    raise TypeError(f"node_size must be a float or a column name str, got {type(node_size)}")


def _resolve_colors(
    node_color,
    node_cmap: str | None,
    node_colorvminvmax,
    nodes_df:  pd.DataFrame,
    n:         int,
    palette:   dict | None,
) -> list:

    if isinstance(node_color, tuple):
        return [_to_rgb(node_color)] * n

    if isinstance(node_color, str):
        if node_color in nodes_df.columns:
            col = nodes_df[node_color]
            if pd.api.types.is_numeric_dtype(col):
                vmin, vmax = _resolve_vminvmax(col.values, node_colorvminvmax)
                return values_to_colors(col.values, cmap=node_cmap or "viridis", vmin=vmin, vmax=vmax)
            return labels_to_colors(col.astype(str).tolist(), cmap=node_cmap, palette=palette)

        try:
            fixed = _to_rgb(node_color)
            return [fixed] * n
        except ValueError:
            raise ValueError(
                f"node_color='{node_color}' is neither a column in nodes_df "
                f"nor a valid matplotlib colour."
            )

    raise TypeError(f"node_color must be a str or RGB tuple, got {type(node_color)}")


def _resolve_vminvmax(values: np.ndarray, mode) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if isinstance(mode, tuple):
        return float(mode[0]), float(mode[1])
    if mode == "absmax":
        m = float(np.nanmax(np.abs(values)))
        return -m, m
    # default "minmax"
    return float(np.nanmin(values)), float(np.nanmax(values))
