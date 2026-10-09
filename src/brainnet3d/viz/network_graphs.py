from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import PathPatch, Polygon, Wedge
from matplotlib.path import Path
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull

from brainnet3d.viz.colormap import labels_to_colors, values_to_colors, values_to_widths
from brainnet3d.viz.layouts import grouped_layout
from brainnet3d.viz.nodes import _resolve_vminvmax

_HUB_RADIUS   = 0.55
_EDGE_SAMPLES = 40


def spring_plot_3d(
    G: nx.Graph,
    node_colors: list | None = None,
    network_labels: list[str] | None = None,
    net2color: dict[str, str] | None = None,
    node_radius: float = 1.5,
    edge_color: str = "grey",
    edge_lw: float = 1,
    seed: int = 42,
    scale: float = 100.0,
    title: str = "Spring Layout 3D",
    camera: dict | None = None,
    interactive: bool = False,
    save_path: str | None = None,
    html: str | None = None,
    layout: str = "spring",
) -> np.ndarray | None:
    """
    Interactive 3-D spring-layout graph rendered with vedo.

    Coloring priority: node_colors > network_labels (net2color, or the default network colours) > tab20 by index.

    Parameters
    ----------
    G               : nx.Graph with optional 'weight' edge attributes.
    node_colors     : per-node color list (length N); any vedo-accepted format.
    network_labels  : subnetwork name per node, length N.
    net2color       : {network_name: colour_string}. None → the 3-D plot's default network colours.
    node_radius     : sphere radius in layout units.
    edge_color      : colour for all edges.
    edge_lw         : edge line width.
    seed            : spring layout random seed.
    scale           : multiplier applied to spring layout coordinates.
    title           : vedo window title.
    camera          : vedo camera dict (pos, focalPoint, viewup).
    interactive     : if True, open a vedo window and return None; otherwise render off screen.
    save_path       : save a screenshot to this path (PNG/JPG) at twice the window resolution.
    html            : save a standalone interactive HTML page (needs brainnet3d[html]).
    layout          : "spring" (default) or "network" (each network in its own ball,
                      placed closer to networks it shares more |weight| with; needs network_labels).

    Returns
    -------
    (H, W, 3) image array when rendered off screen, else None.
    """
    try:
        from vedo import Sphere, Line, Plotter as VPlotter
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")

    nodes = list(G.nodes())
    n = len(nodes)
    if layout == "network":
        if network_labels is None:
            raise ValueError("layout='network' needs network_labels.")
        pos = grouped_layout(G, network_labels, dim=3, seed=seed)
        # bring the layout into the [-1, 1] range that the spring layout uses
        peak = max(float(np.abs(p).max()) for p in pos.values()) or 1.0
        pos = {nd: p / peak for nd, p in pos.items()}
    elif layout == "spring":
        pos = nx.spring_layout(G, dim=3, seed=seed)
    else:
        raise ValueError(f"layout='{layout}' not recognised. Choose: 'spring', 'network'.")
    pos_scaled = {nd: np.array(pos[nd]) * scale for nd in nodes}

    if node_colors is not None:
        colors = list(node_colors)
    elif network_labels is not None:
        net2color = _network_colors(network_labels, net2color)
        colors = [net2color[network_labels[i]] for i in range(n)]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N)[:3] for i in range(n)]

    spheres = [
        Sphere(pos_scaled[nd], r=node_radius, c=colors[i], alpha=0.85).lighting("ambient")
        for i, nd in enumerate(nodes)
    ]
    lines = [
        Line(pos_scaled[u], pos_scaled[v]).lw(edge_lw).c(edge_color)
        for u, v in G.edges()
    ]
    actors = spheres + lines

    from brainnet3d.viz.views import _finish_render

    vp = VPlotter(title=title, bg="white", axes=0, offscreen=not interactive)
    return _finish_render(vp, actors, interactive, screenshot=save_path, html=html, camera=camera)


def matrix_heatmap(
    matrix: np.ndarray,
    labels: list[str] | None = None,
    network_labels: list[str] | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    save_path: str | None = None,
    order: str | None = "network",
    network_order: list[str] | None = None,
    tick_labels: str | None = "auto",
    network_boundaries: str | None = "lines",
    show_diagonal: bool | str = "auto",
    network_palette: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Connectivity matrix heatmap, optionally grouped by network.

    With network_labels, rows and columns are grouped by network, colour strips along the
    top and left edges show each node's network, and thin lines mark network boundaries.

    Parameters
    ----------
    matrix         : square N×N array of connectivity values.
    labels         : ROI label per node (length N).
    network_labels : network name per node (length N). Strip colours match the 3-D plot's
                     node_color="network" when both have the same set of networks.
    vmin, vmax     : colour scale limits.
    cmap           : colormap name.
    figsize        : figure size. Auto-calculated if None.
    title          : axes title.
    save_path      : save figure to this path at 300 dpi.
    order          : "network" (default) → grouped by network; input order without network_labels.
                     "cluster" → hierarchical clustering of each node's row (average linkage,
                     Euclidean distance, optimal leaf order), ignoring networks.
                     "network_cluster" → grouped by network, clustered within each network.
                     "network_chain" → grouped by network; networks follow their strongest mean
                     connections (signed), and nodes in each network run from the side of the
                     previous network to the side of the next.
                     None → input order.
    network_order  : network names in display order; unlisted networks follow in name order.
                     Default: name order, with numbers inside names compared by value.
    tick_labels    : "auto" (default) → ROI labels for up to 40 nodes, otherwise network names
                     at the middle of each group. "roi", "network" or None (no labels).
    network_boundaries : "lines" (default) → lines between networks. "boxes" → a coloured box
                     around each network's diagonal block. None → neither.
    show_diagonal  : "auto" (default) → blank when every diagonal value is the same (e.g. 1),
                     shown otherwise (e.g. within-network means). True or False to force.
    network_palette : {network: colour} override for the strip colours.

    Returns
    -------
    (fig, ax)
    """
    from brainnet3d.viz.matrix_style import draw_matrix, is_contiguous, matrix_order, tick_mode

    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    idx = matrix_order(matrix, network_labels, order, network_order)
    if figsize is None:
        roi_ticks = tick_mode(tick_labels, labels, network_labels, n, is_contiguous(network_labels, idx)) == "roi"
        s = max(8, min(24, n * 0.08)) if roi_ticks else 8
        figsize = (s + 1.5, s)

    fig, ax = plt.subplots(figsize=figsize)
    handles = draw_matrix(
        ax, matrix, idx, labels=labels, network_labels=network_labels, vmin=vmin, vmax=vmax, cmap=cmap,
        tick_labels=tick_labels, network_boundaries=network_boundaries, show_diagonal=show_diagonal,
        network_palette=network_palette, title=title,
    )
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(5, len(handles)),
                  fontsize=7, frameon=False)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def spring_plot(
    G: nx.Graph,
    roi_labels: list[str],
    network_labels: list[str],
    net2color: dict[str, str] | None = None,
    figsize: tuple[float, float] = (22, 22),
    node_size_scale: float = 800.0,
    edge_alpha: float = 0.35,
    label_fontsize: float = 5,
    spring_k: float = 0.15,
    seed: int = 42,
    save_path: str | None = None,
    edge_color: str | tuple = "grey",
    edge_cmap: str = "RdBu_r",
    edge_sign_colors: tuple = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0)),
    network_hulls: bool | None = None,
    layout: str = "spring",
    edge_colorvminvmax: str | tuple | None = "absmax",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Spring-layout 2-D plot of a brain network.

    Parameters
    ----------
    G : nx.Graph with 'weight' edge attributes.
    roi_labels : node label strings, length N.
    network_labels : subnetwork name per node, length N.
    net2color : {network_name: colour}. None → the 3-D plot's default network colours.
    save_path : if given, save figure to this path at 300 dpi.
    edge_color : colour name or RGB tuple; "weight" → edge_cmap centred on 0;
                 "sign" → edge_sign_colors (positive, negative).
    network_hulls : shade a rounded convex hull behind each network's nodes.
                    None (default) → only with layout="network".
    layout : "spring" (default) → spring layout of the whole graph (uses spring_k).
             "network" → each network in its own disc, placed closer to networks it
             shares more |weight| with; spring layout of the network's own edges inside.
    edge_colorvminvmax : colour limits for edge_color="weight", as in plot(). "absmax" (default)
                         → ±max(|weight|), 0 at the colormap centre. "minmax" → data min to max.
                         (vmin, vmax) tuple → explicit limits.
    """
    if layout not in ("spring", "network"):
        raise ValueError(f"layout='{layout}' not recognised. Choose: 'spring', 'network'.")
    if network_hulls is None:
        network_hulls = layout == "network"
    net2color     = _network_colors(network_labels, net2color)
    weights       = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    edge_weights  = np.abs(weights)
    strength_vals = np.array(
        [sum(abs(a["weight"]) for a in G.adj[nd].values()) for nd in G], dtype=float
    )
    peak          = strength_vals.max(initial=0.0) or 1.0  # no edges: every node falls to the minimum size
    node_sizes    = np.clip(strength_vals / peak * node_size_scale, 50, node_size_scale)
    node_colors   = [net2color[net] for net in network_labels]

    if layout == "network":
        pos = grouped_layout(G, network_labels, dim=2, seed=seed)
    else:
        pos = nx.spring_layout(G, seed=seed, k=spring_k)

    fig, ax = plt.subplots(figsize=figsize)
    if layout == "network":
        ax.set_aspect("equal")
    if network_hulls:
        xy  = np.array([pos[i] for i in range(len(network_labels))])
        pad = 0.025 * (np.ptp(xy, axis=0).max() or 1.0)
        for net in dict.fromkeys(network_labels):
            members = [i for i, s in enumerate(network_labels) if s == net]
            ax.add_patch(_hull_patch(xy[members], pad, net2color[net]))
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=np.power(edge_weights, 1.5),
                           edge_color=_edge_colors(weights, edge_color, edge_cmap, edge_sign_colors, edge_colorvminvmax),
                           alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: roi_labels[i] for i in range(len(roi_labels))},
                            font_size=label_fontsize, font_color="black")

    from brainnet3d.viz.matrix_style import natural_key

    unique_nets    = sorted(set(network_labels), key=natural_key)
    legend_handles = [mpatches.Patch(color=net2color[n], label=n) for n in unique_nets]
    ax.legend(handles=legend_handles, title="Subnetwork",
              loc="upper left", fontsize=10, framealpha=0.8)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def circos_plot(
    G: nx.Graph,
    roi_labels: list[str],
    network_labels: list[str],
    net2color: dict[str, str] | None = None,
    figsize: tuple[float, float] = (22, 22),
    edge_alpha: float = 0.6,
    edge_color: str | tuple = "weight",
    node_radius: float | None = None,
    label_fontsize: float = 5,
    gap_between_groups: float = 0.04,
    save_path: str | None = None,
    edge_style: str = "both",
    edge_width: float | str = "weight",
    edge_width_range: tuple[float, float] = (0.3, 2.5),
    edge_cmap: str = "RdBu_r",
    edge_sign_colors: tuple = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0)),
    network_ring: bool = True,
    order: str | None = "network",
    network_order: list[str] | None = None,
    order_matrix: np.ndarray | None = None,
    edge_colorvminvmax: str | tuple | None = "absmax",
    edge_colorbar: bool = True,
    bundle_tension: float = 0.85,
) -> tuple[plt.Figure, plt.Axes] | tuple[tuple[plt.Figure, plt.Axes], tuple[plt.Figure, plt.Axes]]:
    """
    Circos-style plot: nodes arranged in a circle grouped by subnetwork.

    By default two figures are returned as ((fig, ax), (fig, ax)): curved chords coloured by
    edge_color, then edges bundled through their networks and coloured by network.
    Nodes in the same network are placed adjacently in input order; a small angular
    gap separates consecutive network groups. Stronger edges are drawn on top.
    Labels use a darker shade of the network colour.

    Parameters
    ----------
    G : nx.Graph with 'weight' edge attributes.
    roi_labels : node label strings, length N.
    network_labels : subnetwork name per node, length N.
    net2color : {network_name: colour}. None → the 3-D plot's default network colours.
    edge_alpha : edge transparency.
    edge_color : "weight" (default) → edge_cmap centred on 0. "sign" → edge_sign_colors
                 (positive, negative). "network" → fades from one end's network colour to the
                 other's. Colour name or RGB tuple → uniform.
    node_radius : node circle radius (circle radius is 1). None → fitted to the node spacing, at most 0.03.
    gap_between_groups : angular gap (radians) between network groups.
    save_path : if given, save figure to this path at 300 dpi. With edge_style="both" the two
                figures go to <name>_curved and <name>_bundled.
    edge_style : "both" (default) → one "curved" figure and one "bundled" figure with
                 edge_color="network". "curved" → chords bend towards the centre, more for
                 distant nodes. "straight" → straight chords. "bundled" → each edge runs through
                 its networks' hubs, so edges between the same two networks merge into a bundle;
                 falls back to "curved" with a warning when the order splits networks.
    edge_width : "weight" (default) → |weight| scaled to edge_width_range. Number → uniform.
    network_ring : draw a coloured arc and the network name outside each group instead of a legend.
                   When the order splits networks, arcs follow each run and names go to a legend.
    order : node order around the circle, as in matrix_heatmap: "network" (default),
            "network_cluster", "network_chain", "cluster" (by connectivity profile, ignoring
            networks; gaps then sit at the largest splits of the clustering tree and widen with
            the split height) or None (input order).
    network_order : network names in display order, as in matrix_heatmap.
    order_matrix : N×N matrix used to order the nodes. None → the edge weights of G, which
                   may be sparse after thresholding; pass the full matrix for a better order.
    edge_colorvminvmax : colour limits for edge_color="weight", as in plot(). "absmax" (default)
                         → ±max(|weight|), 0 at the colormap centre. "minmax" → data min to max.
                         (vmin, vmax) tuple → explicit limits.
    edge_colorbar : with edge_color="weight", draw a small edge colour bar in the lower right corner.
    bundle_tension : 0 to 1, how tightly bundled edges follow their networks; 0 → straight chords.
    """
    call = dict(locals())
    if edge_style not in ("both", "curved", "straight", "bundled"):
        raise ValueError(f"edge_style='{edge_style}' not recognised. Choose: 'both', 'curved', 'straight', 'bundled'.")
    if not 0.0 <= bundle_tension <= 1.0:
        raise ValueError(f"bundle_tension must be between 0 and 1, got {bundle_tension}.")
    if edge_style == "both":
        figures = []
        for style, colour in (("curved", edge_color), ("bundled", "network")):
            path = None
            if save_path is not None:
                root, ext = os.path.splitext(save_path)
                path = f"{root}_{style}{ext}"
            figures.append(circos_plot(**{**call, "edge_style": style, "edge_color": colour, "save_path": path}))
        return tuple(figures)

    from brainnet3d.viz.matrix_style import _groups, is_contiguous, matrix_order, merge_heights, natural_key

    net2color   = _network_colors(network_labels, net2color)
    n           = len(roi_labels)
    unique_nets = sorted(set(network_labels), key=natural_key)

    if order == "network" and network_order is None:
        order = sorted(range(n), key=lambda i: (natural_key(network_labels[i]), i))
        values = None
    else:
        values = np.zeros((n, n)) if order_matrix is None else np.asarray(order_matrix, dtype=float)
        if order_matrix is None:
            for u, v, w in G.edges(data="weight", default=1.0):
                values[u, v] = values[v, u] = w
        clustered = order == "cluster"
        order = [int(i) for i in matrix_order(values, network_labels, order, network_order)]
    sorted_labels = [roi_labels[i]      for i in order]
    sorted_nets   = [network_labels[i]  for i in order]
    old2new       = {old: new for new, old in enumerate(order)}
    contiguous    = values is None or is_contiguous(network_labels, np.asarray(order))
    if edge_style == "bundled" and not contiguous:
        warnings.warn("The node order splits networks, so edges are drawn curved instead of bundled.", stacklevel=2)
        edge_style = "curved"

    total_gap      = gap_between_groups * len(unique_nets)
    angle_per_node = (2 * np.pi - total_gap) / n

    if contiguous:
        gap_after = [gap_between_groups if sorted_nets[i] != sorted_nets[i + 1] else 0.0 for i in range(n - 1)]
    else:
        # the largest splits of the tree get the gaps, wider for higher splits; one share is left for the wrap
        heights = merge_heights(values, np.asarray(order)) if clustered else np.zeros(n - 1)
        cuts = [i for i in np.argsort(heights)[::-1][: len(unique_nets) - 1] if heights[i] > 0]
        gap_after = [0.0] * (n - 1)
        for i in cuts:
            gap_after[i] = heights[i] / heights[cuts].sum() * total_gap * (len(unique_nets) - 1) / len(unique_nets)

    angles, current_angle = [], np.pi / 2
    for i in range(n):
        if i > 0:
            current_angle += gap_after[i - 1]
        angles.append(current_angle)
        current_angle += angle_per_node

    R  = 1.0
    xs = [R * np.cos(a) for a in angles]
    ys = [R * np.sin(a) for a in angles]
    if node_radius is None:
        node_radius = min(0.03, 0.45 * angle_per_node * R)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    edges   = sorted((e for e in G.edges(data="weight", default=1.0) if e[0] != e[1]), key=lambda e: abs(e[2]))
    weights = np.array([w for _, _, w in edges], dtype=float)
    by_network = isinstance(edge_color, str) and edge_color == "network"
    colors  = [None] * len(edges) if by_network else _edge_colors(weights, edge_color, edge_cmap, edge_sign_colors,
                                                                    edge_colorvminvmax)
    if isinstance(edge_width, (int, float)):
        widths = np.full(len(edges), float(edge_width))
    else:
        widths = values_to_widths(np.abs(weights), edge_width_range) if len(edges) else []
    xy = np.array([[xs[old2new[i]], ys[old2new[i]]] for i in range(n)])
    hubs = _network_hubs(xy, network_labels) if edge_style == "bundled" else {}
    polylines = []
    for (u, v, _), c, lw in zip(edges, colors, widths):
        p0, p2 = xy[u], xy[v]
        if edge_style == "bundled":
            polylines.append(_bundled_path(p0, p2, hubs[network_labels[u]], hubs[network_labels[v]],
                                           network_labels[u] == network_labels[v], bundle_tension))
            continue
        if edge_style == "curved":
            # pull the control point towards the centre, further for longer chords
            ctrl = (p0 + p2) / 2 * (1 - np.linalg.norm(p2 - p0) / (2 * R))
            path = Path([p0, ctrl, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        else:
            path = Path([p0, p2], [Path.MOVETO, Path.LINETO])
        if by_network:
            polylines.append(path.interpolated(_EDGE_SAMPLES).vertices if edge_style == "straight"
                             else _sample_quadratic(path.vertices))
        else:
            ax.add_patch(PathPatch(path, fill=False, edgecolor=c, linewidth=lw, alpha=edge_alpha, zorder=1))
    if polylines and by_network:
        ends = [(net2color[network_labels[u]], net2color[network_labels[v]]) for u, v, _ in edges]
        ax.add_collection(_gradient_lines(polylines, ends, widths, edge_alpha))
    elif polylines:
        ax.add_collection(LineCollection(polylines, colors=[(*to_rgb(c), edge_alpha) for c in colors],
                                         linewidths=widths, capstyle="butt", joinstyle="round", zorder=1))

    for i in range(n):
        ax.add_patch(plt.Circle((xs[i], ys[i]), node_radius,
                                color=net2color[sorted_nets[i]], zorder=2))

    label_R = R + node_radius + 0.04
    if network_ring:
        ring_in  = R + node_radius + 0.012
        ring_out = ring_in + 0.025
        for _, a, b in _groups([str(s) for s in sorted_nets]):
            ax.add_patch(Wedge(
                (0, 0), ring_out,
                np.degrees(angles[a] - angle_per_node / 2), np.degrees(angles[b - 1] + angle_per_node / 2),
                width=ring_out - ring_in, color=net2color[sorted_nets[a]], zorder=2,
            ))
        label_R = ring_out + 0.02

    roi_texts = []
    for i, (lbl, a) in enumerate(zip(sorted_labels, angles)):
        deg      = (np.degrees(a) + 180) % 360 - 180
        ha       = "left" if -90 < deg <= 90 else "right"
        rotation = deg if ha == "left" else deg + 180
        roi_texts.append(ax.text(label_R * np.cos(a), label_R * np.sin(a), lbl,
                                 ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
                                 fontsize=label_fontsize, color=_text_color(net2color[sorted_nets[i]])))

    if network_ring and contiguous:
        _ring_names(ax, unique_nets, sorted_nets, angles, roi_texts, label_R, net2color)
    else:
        legend_handles = [mpatches.Patch(color=net2color[n], label=n) for n in unique_nets]
        ax.legend(handles=legend_handles, title="Subnetwork",
                  loc="upper right", fontsize=10, framealpha=0.8,
                  bbox_to_anchor=(1.15, 1.05))

    if edge_colorbar and isinstance(edge_color, str) and edge_color == "weight" and len(edges):
        vmin, vmax = _resolve_vminvmax(weights, edge_colorvminvmax)
        # the lower right corner lies outside the ring and its labels
        cax = ax.inset_axes([0.74, 0.03, 0.24, 0.012])
        bar = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), edge_cmap), cax=cax, orientation="horizontal",
                           alpha=edge_alpha)
        bar.ax.tick_params(labelsize=8)
        cax.set_title("edge weight", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def _network_hubs(xy: np.ndarray, network_labels: list) -> dict:
    nets = np.asarray(network_labels, dtype=object)
    hubs = {}
    for net in dict.fromkeys(network_labels):
        mean = xy[nets == net].mean(axis=0)
        norm = np.linalg.norm(mean)
        hubs[net] = mean / norm * _HUB_RADIUS if norm > 1e-9 else np.zeros(2)
    return hubs


def _bundled_path(p0, p1, hub0, hub1, same_network: bool, tension: float) -> np.ndarray:
    ctrl = np.array([p0, hub0, p1] if same_network else [p0, hub0, np.zeros(2), hub1, p1])
    # straighten the control polygon towards the chord; tension 0 gives the chord itself
    chord = p0 + np.linspace(0, 1, len(ctrl))[:, None] * (p1 - p0)
    ctrl = tension * ctrl + (1 - tension) * chord
    k = min(3, len(ctrl) - 1)
    knots = np.concatenate([np.zeros(k), np.linspace(0, 1, len(ctrl) - k + 1), np.ones(k)])
    return BSpline(knots, ctrl, k)(np.linspace(0, 1, _EDGE_SAMPLES))


def _sample_quadratic(ctrl: np.ndarray) -> np.ndarray:
    t = np.linspace(0, 1, _EDGE_SAMPLES)[:, None]
    return (1 - t) ** 2 * ctrl[0] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * ctrl[2]


def _gradient_lines(polylines: list, ends: list, widths, alpha: float) -> LineCollection:
    segments, colors, lws = [], [], []
    for pts, (c0, c1), lw in zip(polylines, ends, widths):
        t = np.linspace(0, 1, len(pts) - 1)[:, None]
        segments.extend(np.stack([pts[:-1], pts[1:]], axis=1))
        # network colours are saturated, so they are drawn lighter than weight colours
        colors.extend((*c, alpha * 0.6) for c in (1 - t) * np.array(to_rgb(c0)) + t * np.array(to_rgb(c1)))
        lws.extend([lw] * (len(pts) - 1))
    return LineCollection(segments, colors=colors, linewidths=lws, capstyle="butt", zorder=1)


def _ring_names(ax, nets: list, sorted_nets: list, angles: list, roi_texts: list, label_r: float, net2color: dict) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data  = ax.transData.inverted()
    px_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]

    outer = label_r
    for t in roi_texts:
        box = t.get_window_extent(renderer)
        corners = to_data.transform([(box.x0, box.y0), (box.x0, box.y1), (box.x1, box.y0), (box.x1, box.y1)])
        outer = max(outer, float(np.linalg.norm(corners, axis=1).max()))

    fontsize = 12
    sizes = []
    for net in nets:
        probe = ax.text(0, 0, net, fontsize=fontsize, fontweight="bold")
        box = probe.get_window_extent(renderer)
        sizes.append((box.width / px_per_unit, box.height / px_per_unit))
        probe.remove()

    radius = outer + 0.03 + max(h for _, h in sizes) / 2
    mids   = [np.mean([a for a, s in zip(angles, sorted_nets) if s == net]) for net in nets]
    order  = np.argsort(mids)
    spread = _spread_angles([mids[k] for k in order], [sizes[k][0] / radius for k in order], 0.02)
    for k, theta in zip(order, spread):
        rotation = (np.degrees(theta) - 90 + 180) % 360 - 180
        if abs(rotation) > 90:
            rotation += 180
        ax.text(radius * np.cos(theta), radius * np.sin(theta), nets[k], ha="center", va="center",
                rotation=rotation, rotation_mode="anchor", fontsize=fontsize, fontweight="bold",
                color=_text_color(net2color[nets[k]]))


def _spread_angles(centres: list[float], widths: list[float], pad: float, iterations: int = 500) -> np.ndarray:
    pos = np.array(centres, dtype=float)
    for _ in range(iterations):
        moved = False
        for k in range(len(pos) - 1):
            need = (widths[k] + widths[k + 1]) / 2 + pad
            short = need - (pos[k + 1] - pos[k])
            if short > 1e-9:
                pos[k] -= short / 2
                pos[k + 1] += short / 2
                moved = True
        if len(pos) > 1:
            # the last and first names meet at the top of the circle
            short = (widths[-1] + widths[0]) / 2 + pad - (pos[0] + 2 * np.pi - pos[-1])
            if short > 1e-9:
                pos[-1] -= short / 2
                pos[0] += short / 2
                moved = True
        if not moved:
            break
    return pos


def _text_color(color) -> tuple:
    # a darker shade keeps light network colours readable as text on white
    return tuple(0.6 * x for x in to_rgb(color))


def _network_colors(network_labels: list, net2color: dict | None) -> dict:
    if net2color is not None:
        return net2color
    return dict(zip(network_labels, labels_to_colors([str(x) for x in network_labels])))


def _edge_colors(weights: np.ndarray, edge_color, cmap: str, sign_colors: tuple, vminvmax="absmax") -> list:
    if isinstance(edge_color, str) and edge_color == "weight":
        if weights.size == 0:
            return []
        vmin, vmax = _resolve_vminvmax(weights, vminvmax)
        return values_to_colors(weights, cmap=cmap, vmin=vmin, vmax=vmax)
    if isinstance(edge_color, str) and edge_color == "sign":
        pos, neg = (to_rgb(c) for c in sign_colors)
        return [neg if w < 0 else pos for w in weights]
    return [to_rgb(edge_color)] * len(weights)


def _hull_patch(points: np.ndarray, pad: float, color) -> Polygon:
    # a ring of points around every node gives the hull rounded corners and a margin
    t     = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring  = np.c_[np.cos(t), np.sin(t)] * pad
    cloud = (points[:, None, :] + ring[None, :, :]).reshape(-1, 2)
    hull  = ConvexHull(cloud)
    return Polygon(cloud[hull.vertices], closed=True, facecolor=color, edgecolor=color,
                   alpha=0.18, linewidth=1.0, zorder=0)
