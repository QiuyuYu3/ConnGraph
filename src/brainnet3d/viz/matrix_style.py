"""
Shared drawing for connectivity matrix plots: row order, network colour strips and names, boundaries, marked cells.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy

from brainnet3d.viz.colormap import labels_to_colors

MATRIX_OPTIONS = (
    "network_labels", "order", "network_order", "tick_labels",
    "network_boundaries", "show_diagonal", "network_palette",
)
_ORDERS          = ("network", "cluster", "network_cluster", None)
_ROI_TICK_LIMIT  = 40
_OUTLINE_LIMIT   = 60
_NAME_FONT       = 7


def matrix_order(
    matrix: np.ndarray,
    network_labels=None,
    order: str | None = "network",
    network_order: list[str] | None = None,
) -> np.ndarray:
    if order not in _ORDERS:
        raise ValueError(f"order='{order}' not recognised. Choose: 'network', 'cluster', 'network_cluster' or None.")
    n = matrix.shape[0]
    if order is None or (order == "network" and network_labels is None):
        return np.arange(n)
    if order == "cluster":
        return _cluster_order(matrix)
    if network_labels is None:
        raise ValueError(f"order='{order}' needs network_labels.")

    nets = np.asarray([str(x) for x in network_labels])
    idx: list[int] = []
    for name in _network_sequence(nets, network_order):
        members = np.flatnonzero(nets == name)
        if order == "network_cluster":
            members = members[_cluster_order(matrix[members])]
        idx.extend(members)
    return np.asarray(idx)


def _network_sequence(nets: np.ndarray, network_order) -> list[str]:
    present = sorted(set(nets))
    if network_order is None:
        return present
    unknown = [x for x in network_order if x not in present]
    if unknown:
        raise ValueError(f"network_order has names not in network_labels: {unknown}")
    return list(network_order) + [x for x in present if x not in network_order]


def _cluster_order(rows: np.ndarray) -> np.ndarray:
    # each row is one ROI's connectivity profile
    rows = np.nan_to_num(np.asarray(rows, dtype=float))
    if rows.shape[0] < 3:
        return np.arange(rows.shape[0])
    link = hierarchy.linkage(rows, method="average", metric="euclidean", optimal_ordering=True)
    return hierarchy.leaves_list(link)


def tick_mode(tick_labels, labels, network_labels, n: int, contiguous: bool) -> str | None:
    if tick_labels not in ("auto", "roi", "network", None):
        raise ValueError(f"tick_labels='{tick_labels}' not recognised. Choose: 'auto', 'roi', 'network' or None.")
    if tick_labels == "auto":
        if labels is not None and (network_labels is None or n <= _ROI_TICK_LIMIT):
            return "roi"
        return "network" if network_labels is not None and contiguous else None
    if tick_labels == "roi" and labels is None:
        raise ValueError("tick_labels='roi' needs labels.")
    if tick_labels == "network" and (network_labels is None or not contiguous):
        raise ValueError("tick_labels='network' needs network_labels and an order that keeps networks together.")
    return tick_labels


def is_contiguous(network_labels, idx: np.ndarray) -> bool:
    if network_labels is None:
        return False
    snets = [str(network_labels[i]) for i in idx]
    return len(_groups(snets)) == len(set(snets))


def _groups(snets: list[str]) -> list[tuple[str, int, int]]:
    cuts = [0] + [k for k in range(1, len(snets)) if snets[k] != snets[k - 1]] + [len(snets)]
    return [(snets[a], a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def draw_matrix(
    ax,
    matrix: np.ndarray,
    idx: np.ndarray,
    labels=None,
    network_labels=None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    tick_labels: str | None = "auto",
    network_boundaries: str | None = "lines",
    show_diagonal: bool | str = "auto",
    network_palette: dict | None = None,
    marks: np.ndarray | None = None,
    sig_style: str = "auto",
    colorbar: bool | str = True,
    title: str | None = None,
) -> list:
    """Draw matrix rows and columns in idx order with network decorations; returns network legend handles when networks are split."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    for name, seq in (("labels", labels), ("network_labels", network_labels)):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} entries; the matrix has {n} rows.")
    if network_boundaries not in ("lines", "boxes", None):
        raise ValueError(f"network_boundaries='{network_boundaries}' not recognised. Choose: 'lines', 'boxes' or None.")
    if sig_style not in ("auto", "outline", "fade"):
        raise ValueError(f"sig_style='{sig_style}' not recognised. Choose: 'auto', 'outline', 'fade'.")
    if show_diagonal not in (True, False, "auto"):
        raise ValueError(f"show_diagonal={show_diagonal!r} not recognised. Choose: True, False, 'auto'.")

    contiguous = is_contiguous(network_labels, idx)
    mode = tick_mode(tick_labels, labels, network_labels, n, contiguous)

    m = matrix[np.ix_(idx, idx)]
    diag = np.diag(m)
    if show_diagonal is False or (show_diagonal == "auto" and np.unique(diag[~np.isnan(diag)]).size <= 1):
        m = m.copy()
        np.fill_diagonal(m, np.nan)

    norm = Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    rgba = cm(norm(m))
    rgba[np.isnan(m)] = (1.0, 1.0, 1.0, 1.0)

    sig = None
    if marks is not None:
        sig = np.asarray(marks)[np.ix_(idx, idx)] != 0
        style = sig_style if sig_style != "auto" else ("outline" if n <= _OUTLINE_LIMIT else "fade")
        if style == "fade":
            rgba[~sig, :3] = rgba[~sig, :3] * 0.25 + 0.75

    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if sig is not None and style == "outline":
        ax.add_collection(LineCollection(_outline_segments(sig), colors="black", linewidths=1.6))

    divider = make_axes_locatable(ax)
    left = top = None
    if network_labels is not None:
        snets = [str(network_labels[i]) for i in idx]
        colors = dict(zip(*_network_colors(network_labels, network_palette)))
        strip = np.array([colors[x] for x in snets])
        left = divider.append_axes("left", size="3%", pad=0.03)
        top = divider.append_axes("top", size="3%", pad=0.03)
        left.imshow(strip[:, None, :], aspect="auto", interpolation="nearest")
        top.imshow(strip[None, :, :], aspect="auto", interpolation="nearest")
        for strip_ax in (left, top):
            strip_ax.set_xticks([])
            strip_ax.set_yticks([])
            for spine in strip_ax.spines.values():
                spine.set_visible(False)
    if colorbar:
        cax = divider.append_axes("right", size="3%", pad=0.08)
        # "space" keeps the panel the same size as one that has a colour bar
        if colorbar == "space":
            cax.set_axis_off()
        else:
            ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cm), cax=cax)

    if mode == "roi":
        _roi_ticks(ax, left, top, [str(labels[i]) for i in idx])
    elif mode == "network":
        _network_names(ax, left, top, _groups(snets))

    if contiguous:
        groups = _groups(snets)
        if network_boundaries == "lines" and len(groups) < n:
            for _, a, _ in groups[1:]:
                ax.axhline(a - 0.5, color="black", linewidth=0.6, alpha=0.7)
                ax.axvline(a - 0.5, color="black", linewidth=0.6, alpha=0.7)
        elif network_boundaries == "boxes":
            for name, a, b in groups:
                ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, edgecolor=colors[name], linewidth=1.6))

    if title:
        _title(ax, top, title)

    if network_labels is None or contiguous:
        return []
    return [Patch(facecolor=c, label=name) for name, c in colors.items()]


def _network_colors(network_labels, palette: dict | None) -> tuple[list[str], list]:
    names = [str(x) for x in network_labels]
    first = dict(zip(names, labels_to_colors(names, palette=palette)))
    return list(first), list(first.values())


def _outline_segments(sig: np.ndarray) -> list:
    n = sig.shape[0]
    segs = []
    for i, j in zip(*np.nonzero(sig)):
        if i == 0 or not sig[i - 1, j]:
            segs.append([(j - 0.5, i - 0.5), (j + 0.5, i - 0.5)])
        if i == n - 1 or not sig[i + 1, j]:
            segs.append([(j - 0.5, i + 0.5), (j + 0.5, i + 0.5)])
        if j == 0 or not sig[i, j - 1]:
            segs.append([(j - 0.5, i - 0.5), (j - 0.5, i + 0.5)])
        if j == n - 1 or not sig[i, j + 1]:
            segs.append([(j + 0.5, i - 0.5), (j + 0.5, i + 0.5)])
    return segs


def _roi_ticks(ax, left, top, labels: list[str]) -> None:
    n = len(labels)
    fig = ax.figure
    fig.canvas.draw()
    pts_per_row = ax.get_window_extent().height / n * 72 / fig.dpi
    fontsize = max(3.0, min(8.0, 0.8 * pts_per_row))
    y_ax, x_ax = (left, top) if left is not None else (ax, ax)
    y_ax.set_yticks(range(n), labels, fontsize=fontsize)
    x_ax.set_xticks(range(n), labels, rotation=90, fontsize=fontsize)
    if top is not None:
        top.xaxis.set_ticks_position("top")
        for strip_ax in (left, top):
            strip_ax.tick_params(length=0)


def _network_names(ax, left, top, groups: list) -> None:
    fig = ax.figure
    fig.canvas.draw()
    n = groups[-1][2]
    pts_per_row = ax.get_window_extent().height / n * 72 / fig.dpi
    mids = [(a + b - 1) / 2 for _, a, b in groups]
    pos = _spread(mids, _NAME_FONT * 1.25 / pts_per_row, 0, n - 1)

    line = dict(arrowstyle="-", lw=0.4, color="0.5", shrinkA=0, shrinkB=2)
    y_coords = blended_transform_factory(left.transAxes, left.transData)
    x_coords = blended_transform_factory(top.transData, top.transAxes)
    for (name, _, _), mid, p in zip(groups, mids, pos):
        left.annotate(name, xy=(0, mid), xycoords=y_coords, xytext=(-12, p), textcoords=("offset points", "data"),
                      ha="right", va="center", fontsize=_NAME_FONT, arrowprops=line)
        top.annotate(name, xy=(mid, 1), xycoords=x_coords, xytext=(p, 12), textcoords=("data", "offset points"),
                     ha="center", va="bottom", rotation=90, fontsize=_NAME_FONT, arrowprops=line)


def _spread(mids: list[float], gap: float, lo: float, hi: float) -> np.ndarray:
    pos = np.array(mids, dtype=float)
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    pos[-1] = min(pos[-1], hi)
    for k in range(len(pos) - 2, -1, -1):
        pos[k] = min(pos[k], pos[k + 1] - gap)
    return np.maximum(pos, lo)


def _title(ax, top, title: str) -> None:
    if top is None:
        ax.set_title(title)
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    above = [top.get_window_extent(renderer).y1] + [t.get_window_extent(renderer).y1 for t in top.texts]
    above += [t.get_window_extent(renderer).y1 for t in top.get_xticklabels() if t.get_text()]
    y = top.transAxes.inverted().transform((0, max(above)))[1]
    top.set_title(title, y=y, pad=6)
