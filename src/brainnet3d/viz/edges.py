"""
Edge rendering: connectivity matrix + node positions → vedo Line/Tube objects.
"""

from __future__ import annotations

import warnings

import numpy as np

from brainnet3d.viz.colormap import (
    edge_colors_from_weights,
    edge_colors_from_node_colors,
    values_to_widths,
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

    Returns
    -------
    List of vedo Line or Tube objects.
    """
    from vedo import Line, Tube

    n = matrix.shape[0]

    if threshold_dir == "absabove":
        _keep = lambda w: abs(w) > threshold
    elif threshold_dir == "above":
        _keep = lambda w: w > threshold
    elif threshold_dir == "below":
        _keep = lambda w: w < -threshold
    else:
        raise ValueError(
            f"threshold_dir='{threshold_dir}' not recognised. "
            "Choose: 'absabove', 'above', 'below'."
        )

    rows, cols, weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            w = matrix[i, j]
            if not np.isnan(w) and _keep(w):
                rows.append(i)
                cols.append(j)
                weights.append(w)

    if not rows:
        warnings.warn("No edges pass the threshold; only nodes will be drawn.", stacklevel=4)
        return []

    weights  = np.array(weights, dtype=float)
    n_edges  = len(rows)

    if isinstance(edge_width, (int, float)):
        widths = np.full(n_edges, float(edge_width))
    else:
        widths = values_to_widths(np.abs(weights), edge_width_range)

    if isinstance(edge_color, tuple):
        colors = [_to_rgb(edge_color)] * n_edges
    elif edge_color == "weight":
        vmin, vmax = _resolve_vminvmax(weights, edge_colorvminvmax)
        colors = edge_colors_from_weights(weights, cmap=edge_cmap, vmin=vmin, vmax=vmax)
    elif edge_color == "sign":
        pos, neg = (_to_rgb(c) for c in edge_sign_colors)
        colors = [neg if w < 0 else pos for w in weights]
    elif edge_color == "node":
        if node_colors is None:
            raise ValueError("edge_color='node' requires node_colors to be provided.")
        colors = edge_colors_from_node_colors(rows, node_colors)
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

    lines = []
    for idx, (i, j) in enumerate(zip(rows, cols)):
        p1 = positions[i]
        p2 = positions[j]
        c  = colors[idx]
        w  = float(widths[idx])

        if highlight_edges is not None:
            significant = bool(highlight_edges[i, j]) or bool(highlight_edges[j, i])
            alpha = edge_alpha if significant else dim_alpha
        else:
            alpha = edge_alpha

        if use_tube:
            obj = Tube([p1, p2], r=w * 0.1, c=c, alpha=alpha)
        else:
            obj = Line(p1, p2, c=c, alpha=alpha).lw(w)

        obj._endpoints  = (i, j)
        obj._weight     = float(weights[idx])
        obj._orig_color = c
        obj._orig_alpha = alpha
        lines.append(obj)

    return lines
