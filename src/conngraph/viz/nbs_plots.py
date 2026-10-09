from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from conngraph.viz.matrix_style import MATRIX_OPTIONS, draw_matrix, is_contiguous, matrix_order, tick_mode


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
    diff_vmax: float | None = None,
    **matrix_options,
) -> Figure:
    """
    3-panel NBS result plot: group 1 mean | group 2 mean | group 1 − group 2 with significant edges marked.

    The difference has the same sign as the NBS t statistic (positive where group 1 is higher).

    Parameters
    ----------
    mean_g1, mean_g2 : N×N mean connectivity matrices for each group.
    adj              : N×N adjacency matrix from NBSResult.adj[:,:,c] for
                       one component, or the summed adj across all components.
    labels           : ROI/network label list (length N).
    vmin, vmax       : colour scale limits of the two group means.
    cmap             : colormap for connectivity values.
    group_names      : display names for group 1 and group 2.
    title            : overall figure title.
    figsize          : figure size; auto-calculated from N if None. Widened if labels would overlap a neighbouring panel.
    save_path        : save figure to this path at 300 dpi.
    sig_style        : "auto" (default) → "outline" for up to 60 nodes, else "fade".
                       "outline" → black outline around significant cells.
                       "fade" → non-significant cells drawn faint.
    diff_vmax        : colour limit of the difference panel, drawn from −diff_vmax to +diff_vmax.
                       Default: largest absolute off-diagonal difference.
    **matrix_options : layout options of matrix_heatmap: network_labels, order, network_order,
                       tick_labels, network_boundaries, show_diagonal, network_palette.
                       "cluster" and "network_chain" orders use the average of the two group matrices, so all
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

    diff = mean_g1 - mean_g2
    if diff_vmax is None:
        off_diag = ~np.eye(n, dtype=bool)
        diff_vmax = float(np.nanmax(np.abs(diff[off_diag]), initial=0.0)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)
    panels = [
        (mean_g1, group_names[0], None, (vmin, vmax), "space"),
        (mean_g2, group_names[1], None, (vmin, vmax), True),
        (diff, f"{group_names[0]} − {group_names[1]}", adj, (-diff_vmax, diff_vmax), True),
    ]
    groups = []
    for ax, (mat, name, marks, (lo, hi), colorbar) in zip(axes, panels):
        before = set(fig.axes)
        handles = draw_matrix(
            ax, mat, idx, labels=labels, network_labels=network_labels, vmin=lo, vmax=hi, cmap=cmap,
            marks=marks, sig_style=sig_style, colorbar=colorbar, title=name, **matrix_options,
        )
        groups.append([ax] + [a for a in fig.axes if a not in before])
    _separate_panels(fig, groups)
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=min(7, len(handles)),
                   fontsize=7, frameon=False)

    if title:
        fig.canvas.draw()
        top = fig.get_tightbbox(fig.canvas.get_renderer()).y1 / fig.get_figheight()
        fig.suptitle(title, y=top + 0.03, va="bottom")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _separate_panels(fig: Figure, groups: list[list], pad_pt: float = 8.0) -> None:
    # widen the gaps (and the figure) until each panel's labels clear its neighbour; panel sizes stay the same
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [Bbox.union([b for a in g if (b := a.get_tightbbox(renderer)) is not None]) for g in groups]
    extra = max(a.x1 - b.x0 for a, b in zip(boxes, boxes[1:])) / fig.dpi + pad_pt / 72
    if extra <= 0:
        return
    width, height = fig.get_size_inches()
    left, right, wspace = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.wspace
    k = len(groups)
    slot = width * (right - left) / (k + (k - 1) * wspace)
    new_width = width + (k - 1) * extra
    fig.set_size_inches(new_width, height)
    fig.subplots_adjust(left=left * width / new_width, right=1 - (1 - right) * width / new_width,
                        wspace=(wspace * slot + extra) / slot)
