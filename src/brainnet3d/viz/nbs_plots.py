from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
) -> Figure:
    """
    3-panel NBS result plot: group 1 mean | group 2 mean | significant edges.

    Parameters
    ----------
    mean_g1, mean_g2 : N×N mean connectivity matrices for each group.
    adj              : N×N adjacency matrix from NBSResult.adj[:,:,c] for
                       one component, or the summed adj across all components.
    labels           : ROI/network label list (length N). If None, no ticks.
    vmin, vmax       : colour scale limits.
    cmap             : colormap for connectivity values.
    group_names      : display names for group 1 and group 2.
    title            : overall figure title.
    figsize          : figure size; auto-calculated from N if None.
    save_path        : save figure to this path at 150 dpi.

    Returns
    -------
    matplotlib Figure.
    """
    n = mean_g1.shape[0]
    if figsize is None:
        w = max(15, n * 0.25)
        figsize = (w, w / 3)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, mat, name in zip(axes[:2], [mean_g1, mean_g2], group_names):
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        _set_ticks(ax, labels)

    # significant edges panel
    im2 = axes[2].imshow(mean_g1, cmap=cmap, vmin=vmin, vmax=vmax)
    overlay = np.ones((*adj.shape, 4))
    overlay[adj > 0] = [0, 0, 0, 0]   # transparent where significant
    overlay[adj == 0] = [1, 1, 1, 1]  # white where not significant
    axes[2].imshow(overlay)
    axes[2].set_title("Significant edges")
    _set_ticks(axes[2], labels)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax)

    if title:
        fig.suptitle(title, y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _set_ticks(ax: plt.Axes, labels: list[str] | None) -> None:
    if labels is None:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    n = len(labels)
    tick_fontsize = max(4, min(8, 120 // n))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
