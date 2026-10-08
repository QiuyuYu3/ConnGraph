from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


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
) -> np.ndarray | None:
    """
    Interactive 3-D spring-layout graph rendered with vedo.

    Coloring priority: node_colors > (network_labels + net2color) > tab20 by index.

    Parameters
    ----------
    G               : nx.Graph with optional 'weight' edge attributes.
    node_colors     : per-node color list (length N); any vedo-accepted format.
    network_labels  : subnetwork name per node, length N.
    net2color       : {network_name: colour_string}.
    node_radius     : sphere radius in layout units.
    edge_color      : colour for all edges.
    edge_lw         : edge line width.
    seed            : spring layout random seed.
    scale           : multiplier applied to spring layout coordinates.
    title           : vedo window title.
    camera          : vedo camera dict (pos, focalPoint, viewup).
    interactive     : if True, open a vedo window and return None; otherwise render off screen.
    save_path       : save a screenshot to this path (PNG/JPG).
    html            : save a standalone interactive HTML page (needs brainnet3d[html]).

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
    pos = nx.spring_layout(G, dim=3, seed=seed)
    pos_scaled = {nd: np.array(pos[nd]) * scale for nd in nodes}

    if node_colors is not None:
        colors = list(node_colors)
    elif network_labels is not None and net2color is not None:
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
                     node_color="network" when the nodes are in the same order.
    vmin, vmax     : colour scale limits.
    cmap           : colormap name.
    figsize        : figure size. Auto-calculated if None.
    title          : axes title.
    save_path      : save figure to this path at 150 dpi.
    order          : "network" (default) → grouped by network; input order without network_labels.
                     "cluster" → hierarchical clustering of each node's row (average linkage,
                     Euclidean distance, optimal leaf order), ignoring networks.
                     "network_cluster" → grouped by network, clustered within each network.
                     None → input order.
    network_order  : network names in display order; unlisted networks follow alphabetically.
                     Default: alphabetical.
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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def spring_plot(
    G: nx.Graph,
    roi_labels: list[str],
    network_labels: list[str],
    net2color: dict[str, str],
    figsize: tuple[float, float] = (22, 22),
    node_size_scale: float = 800.0,
    edge_alpha: float = 0.35,
    label_fontsize: float = 5,
    spring_k: float = 0.15,
    seed: int = 42,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Spring-layout 2-D plot of a brain network.

    Parameters
    ----------
    G : nx.Graph with 'weight' edge attributes.
    roi_labels : node label strings, length N.
    network_labels : subnetwork name per node, length N.
    net2color : {network_name: colour_string}.
    save_path : if given, save figure to this path at 150 dpi.
    """
    edge_weights  = np.abs([d["weight"] for _, _, d in G.edges(data=True)])
    strength_vals = np.array(
        [sum(abs(a["weight"]) for a in G.adj[nd].values()) for nd in G], dtype=float
    )
    peak          = strength_vals.max(initial=0.0) or 1.0  # no edges: every node falls to the minimum size
    node_sizes    = np.clip(strength_vals / peak * node_size_scale, 50, node_size_scale)
    node_colors   = [net2color[net] for net in network_labels]

    pos = nx.spring_layout(G, seed=seed, k=spring_k)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=np.power(edge_weights, 1.5),
                           edge_color="grey", alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: roi_labels[i] for i in range(len(roi_labels))},
                            font_size=label_fontsize, font_color="black")

    unique_nets    = sorted(set(network_labels))
    legend_handles = [mpatches.Patch(color=net2color[n], label=n) for n in unique_nets]
    ax.legend(handles=legend_handles, title="Subnetwork",
              loc="upper left", fontsize=10, framealpha=0.8)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def circos_plot(
    G: nx.Graph,
    roi_labels: list[str],
    network_labels: list[str],
    net2color: dict[str, str],
    figsize: tuple[float, float] = (22, 22),
    edge_alpha: float = 0.15,
    edge_color: str = "grey",
    node_radius: float = 0.03,
    label_fontsize: float = 5,
    gap_between_groups: float = 0.04,
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Circos-style plot: nodes arranged in a circle grouped by subnetwork.

    Nodes in the same network are placed adjacently; a small angular gap
    separates consecutive network groups.

    Parameters
    ----------
    G : nx.Graph.
    roi_labels : node label strings, length N.
    network_labels : subnetwork name per node, length N.
    net2color : {network_name: colour_string}.
    gap_between_groups : angular gap (radians) between network groups.
    save_path : if given, save figure to this path at 150 dpi.
    """
    n           = len(roi_labels)
    unique_nets = sorted(set(network_labels))

    order        = sorted(range(n), key=lambda i: (network_labels[i], roi_labels[i]))
    sorted_labels = [roi_labels[i]      for i in order]
    sorted_nets   = [network_labels[i]  for i in order]
    old2new       = {old: new for new, old in enumerate(order)}

    total_gap      = gap_between_groups * len(unique_nets)
    angle_per_node = (2 * np.pi - total_gap) / n

    angles, current_angle, prev_net = [], np.pi / 2, None
    for net in sorted_nets:
        if prev_net is not None and net != prev_net:
            current_angle += gap_between_groups
        angles.append(current_angle)
        current_angle += angle_per_node
        prev_net = net

    R  = 1.0
    xs = [R * np.cos(a) for a in angles]
    ys = [R * np.sin(a) for a in angles]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    for u, v in G.edges():
        u2, v2 = old2new[u], old2new[v]
        ax.plot([xs[u2], xs[v2]], [ys[u2], ys[v2]],
                color=edge_color, alpha=edge_alpha, linewidth=0.5, zorder=1)

    for i in range(n):
        ax.add_patch(plt.Circle((xs[i], ys[i]), node_radius,
                                color=net2color[sorted_nets[i]], zorder=2))

    label_R = R + node_radius + 0.04
    for i, (lbl, a) in enumerate(zip(sorted_labels, angles)):
        deg      = (np.degrees(a) + 180) % 360 - 180
        ha       = "left" if -90 < deg <= 90 else "right"
        rotation = deg if ha == "left" else deg + 180
        ax.text(label_R * np.cos(a), label_R * np.sin(a), lbl,
                ha=ha, va="center", rotation=rotation, rotation_mode="anchor",
                fontsize=label_fontsize, color=net2color[sorted_nets[i]])

    legend_handles = [mpatches.Patch(color=net2color[n], label=n) for n in unique_nets]
    ax.legend(handles=legend_handles, title="Subnetwork",
              loc="upper right", fontsize=10, framealpha=0.8,
              bbox_to_anchor=(1.15, 1.05))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
