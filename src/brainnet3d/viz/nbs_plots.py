from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from brainnet3d.viz.matrix_style import MATRIX_OPTIONS, draw_matrix, is_contiguous, matrix_order, tick_mode


def plot_nbs_matrices(
    mean_g1: np.ndarray,
    mean_g2: np.ndarray,
    adj: np.ndarray,
    labels: list[str] | None = None,
    vmin: float = -0.5,
    vmax: float = 0.5,
    cmap: str = "RdBu_r",
    group_names: tuple[str, str] = ("Group 1", "Group 2"),
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    sig_style: str = "auto",
    **matrix_options,
) -> Figure:
    """
    3-panel NBS result plot: group 1 mean | group 2 mean | group 1 mean with significant edges marked.

    Parameters
    ----------
    mean_g1, mean_g2 : N×N mean connectivity matrices for each group.
    adj              : N×N adjacency matrix from NBSResult.adj[:,:,c] for
                       one component, or the summed adj across all components.
    labels           : ROI/network label list (length N).
    vmin, vmax       : colour scale limits.
    cmap             : colormap for connectivity values.
    group_names      : display names for group 1 and group 2.
    title            : overall figure title.
    figsize          : figure size; auto-calculated from N if None.
    save_path        : save figure to this path at 150 dpi.
    sig_style        : "auto" (default) → "outline" for up to 60 nodes, else "fade".
                       "outline" → black outline around significant cells.
                       "fade" → non-significant cells drawn faint.
    **matrix_options : layout options of matrix_heatmap: network_labels, order, network_order,
                       tick_labels, network_boundaries, show_diagonal, network_palette.
                       "cluster" orders use the average of the two group matrices, so all
                       panels share one order.

    Returns
    -------
    matplotlib Figure.
    """
    unknown = set(matrix_options) - set(MATRIX_OPTIONS)
    if unknown:
        raise TypeError(f"plot_nbs_matrices() got unexpected keyword arguments: {sorted(unknown)}")
    network_labels = matrix_options.pop("network_labels", None)
    order          = matrix_options.pop("order", "network")
    network_order  = matrix_options.pop("network_order", None)

    mean_g1 = np.asarray(mean_g1, dtype=float)
    mean_g2 = np.asarray(mean_g2, dtype=float)
    n   = mean_g1.shape[0]
    idx = matrix_order((mean_g1 + mean_g2) / 2, network_labels, order, network_order)
    if figsize is None:
        mode = tick_mode(matrix_options.get("tick_labels", "auto"), labels, network_labels, n,
                         is_contiguous(network_labels, idx))
        w = max(15, n * 0.25) if mode == "roi" else 19
        figsize = (w, w / 3)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)
    panels = [(mean_g1, group_names[0], None), (mean_g2, group_names[1], None), (mean_g1, "Significant edges", adj)]
    for k, (ax, (mat, name, marks)) in enumerate(zip(axes, panels)):
        handles = draw_matrix(
            ax, mat, idx, labels=labels, network_labels=network_labels, vmin=vmin, vmax=vmax, cmap=cmap,
            marks=marks, sig_style=sig_style, colorbar=True if k == 2 else "space", title=name, **matrix_options,
        )
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=min(7, len(handles)),
                   fontsize=7, frameon=False)

    if title:
        fig.canvas.draw()
        top = fig.get_tightbbox(fig.canvas.get_renderer()).y1 / fig.get_figheight()
        fig.suptitle(title, y=top + 0.03, va="bottom")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
