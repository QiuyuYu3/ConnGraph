"""
Edge rendering: connectivity matrix + node positions → vedo Line/Tube objects.
"""

from __future__ import annotations

import warnings

import numpy as np

from brainnet3d.viz.colormap import (
    edge_colors_from_weights,
    edge_colors_from_node_colors,
    _to_rgb,
)
from brainnet3d.viz.nodes import _resolve_vminvmax


def build_edges(
    matrix:        np.ndarray,
    positions:     np.ndarray,
    threshold:     float = 0.5,
    threshold_dir: str   = "absabove",
    edge_width:    str | float = "weight",
    edge_width_range: tuple[float, float] = (0.5, 4.0),
    edge_color:    str | tuple = "weight",
    edge_cmap:     str   = "RdBu_r",
    edge_colorvminvmax: str | tuple | None = "absmax",
    edge_alpha:    float = 0.7,
    node_colors:   list | None = None,
    use_tube:      bool  = False,
    highlight_edges: np.ndarray | None = None,
    highlight_level: float = 0.85,
    edge_sign_colors: tuple = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0)),
    bundling:      float = 0.0,
) -> list:
    """
    Build vedo Line (or Tube) objects for all edges above threshold.

    Parameters
    ----------
    matrix : (N, N) numpy array — connectivity matrix (upper triangle used).
    positions : (N, 3) numpy array — MNI coordinates.
    threshold     : numeric cutoff value.
    threshold_dir : how to apply the threshold.
                    "absabove" (default) — keep edges where |weight| > threshold.
                    "above"              — keep edges where weight > threshold (positive only).
                    "below"              — keep edges where weight < -threshold (negative only).
    edge_width : float → uniform. "weight" → rescaled to edge_width_range.
    edge_width_range : (min_w, max_w) in pixels.
    edge_color : "weight" → colormap. "node" → inherits node colour.
                 "sign" → edge_sign_colors. RGB tuple or colour name → uniform.
    edge_cmap : colormap used when edge_color="weight".
    edge_alpha : transparency 0–1.
    node_colors : list of RGB tuples (required when edge_color="node").
    use_tube : render edges as 3-D Tube instead of flat Line (slower).
    highlight_edges : (N, N) binary adjacency matrix marking significant edges
                      (e.g. NBSResult.adj[:,:,0]). Significant edges keep full
                      alpha; all others are dimmed by highlight_level.
                      Pass None to disable (default).
    highlight_level : dimming strength for non-highlighted edges (0 = no dim,
                      1 = fully invisible). Default 0.85.
    edge_sign_colors : (positive, negative) colours used when edge_color="sign".
    bundling : edge bundling strength. 0 (default) → straight lines; larger values pull
               edges that run close together into shared curved bundles.

    Returns
    -------
    List of vedo Line or Tube objects.
    """
    from vedo import Line, Tube

    rows, cols, weights = select_edges(matrix, threshold, threshold_dir)
    if not rows.size:
        warnings.warn("No edges pass the threshold; only nodes will be drawn.", stacklevel=4)
        return []

    colors, widths, alphas = style_edges(
        rows, cols, weights, weights,
        edge_width=edge_width, edge_width_range=edge_width_range, edge_color=edge_color, edge_cmap=edge_cmap,
        edge_colorvminvmax=edge_colorvminvmax, edge_alpha=edge_alpha, node_colors=node_colors,
        highlight_edges=highlight_edges, highlight_level=highlight_level, edge_sign_colors=edge_sign_colors,
    )

    paths = None
    if bundling:
        from brainnet3d.viz.bundling import bundle_paths

        positions = np.asarray(positions, dtype=float)
        paths = bundle_paths(positions[rows], positions[cols], strength=bundling)

    lines = []
    for idx, (i, j) in enumerate(zip(rows.tolist(), cols.tolist())):
        p1 = positions[i]
        p2 = positions[j]
        c  = colors[idx]
        w  = float(widths[idx])
        alpha = alphas[idx]

        # Tube reads an RGB tuple as one colour per point, so colour it afterwards
        if paths is not None:
            obj = Tube(paths[idx], r=w * 0.25).color(c).alpha(alpha) if use_tube else Line(paths[idx], c=c, alpha=alpha).lw(w)
        elif use_tube:
            obj = Tube([p1, p2], r=w * 0.25).color(c).alpha(alpha)
        else:
            obj = Line(p1, p2, c=c, alpha=alpha).lw(w)

        obj._endpoints  = (i, j)
        obj._weight     = float(weights[idx])
        obj._orig_color = c
        obj._orig_alpha = alpha
        lines.append(obj)

    return lines


def select_edges(
    matrix:        np.ndarray,
    threshold:     float,
    threshold_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle edges passing the threshold, as row indices, column indices and weights."""
    if threshold_dir == "absabove":
        _keep = lambda w: np.abs(w) > threshold
    elif threshold_dir == "above":
        _keep = lambda w: w > threshold
    elif threshold_dir == "below":
        _keep = lambda w: w < -threshold
    else:
        raise ValueError(
            f"threshold_dir='{threshold_dir}' not recognised. "
            "Choose: 'absabove', 'above', 'below'."
        )

    matrix = np.asarray(matrix)
    rows, cols = np.triu_indices(matrix.shape[0], 1)
    weights = matrix[rows, cols].astype(float)
    with np.errstate(invalid="ignore"):
        keep = ~np.isnan(weights) & _keep(weights)
    return rows[keep], cols[keep], weights[keep]


def style_edges(
    rows:          np.ndarray,
    cols:          np.ndarray,
    weights:       np.ndarray,
    scale_weights: np.ndarray,
    edge_width:    str | float = "weight",
    edge_width_range: tuple[float, float] = (0.5, 4.0),
    edge_color:    str | tuple = "weight",
    edge_cmap:     str   = "RdBu_r",
    edge_colorvminvmax: str | tuple | None = "absmax",
    edge_alpha:    float = 0.7,
    node_colors:   list | None = None,
    highlight_edges: np.ndarray | None = None,
    highlight_level: float = 0.85,
    edge_sign_colors: tuple = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0)),
) -> tuple[list, np.ndarray, list]:
    """Colour, width and alpha of each edge; colour limits and width scale come from scale_weights."""
    n_edges = len(weights)

    if isinstance(edge_width, (int, float)):
        widths = np.full(n_edges, float(edge_width))
    else:
        widths = _scaled_widths(np.abs(weights), np.abs(scale_weights), edge_width_range)

    if isinstance(edge_color, tuple):
        colors = [_to_rgb(edge_color)] * n_edges
    elif edge_color == "weight":
        vmin, vmax = _resolve_vminvmax(scale_weights, edge_colorvminvmax)
        colors = edge_colors_from_weights(weights, cmap=edge_cmap, vmin=vmin, vmax=vmax)
    elif edge_color == "sign":
        pos, neg = (_to_rgb(c) for c in edge_sign_colors)
        colors = [neg if w < 0 else pos for w in weights]
    elif edge_color == "node":
        if node_colors is None:
            raise ValueError("edge_color='node' requires node_colors to be provided.")
        colors = edge_colors_from_node_colors(rows.tolist(), node_colors)
    else:
        try:
            fixed = _to_rgb(edge_color)
            colors = [fixed] * n_edges
        except ValueError:
            raise ValueError(
                f"edge_color='{edge_color}' is not recognised. "
                "Use 'weight', 'node', 'sign', an RGB tuple, or a matplotlib colour name."
            )

    dim_alpha = edge_alpha * (1.0 - highlight_level)
    if highlight_edges is None:
        alphas = [edge_alpha] * n_edges
    else:
        alphas = [
            edge_alpha if bool(highlight_edges[i, j]) or bool(highlight_edges[j, i]) else dim_alpha
            for i, j in zip(rows.tolist(), cols.tolist())
        ]
    return colors, widths, alphas


def _scaled_widths(values: np.ndarray, scale_values: np.ndarray, width_range: tuple[float, float]) -> np.ndarray:
    # values outside the scale's range get the end widths
    lo, hi = float(np.nanmin(scale_values)), float(np.nanmax(scale_values))
    if hi == lo:
        return np.full_like(values, (width_range[0] + width_range[1]) / 2.0)
    normed = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return width_range[0] + normed * (width_range[1] - width_range[0])
