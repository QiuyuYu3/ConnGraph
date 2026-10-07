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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Connectivity matrix heatmap, optionally grouped by network.

    When network_labels is provided, rows and columns are sorted so nodes in
    the same network are adjacent, and thin lines mark network boundaries.

    Parameters
    ----------
    matrix         : square N×N array of connectivity values.
    labels         : ROI label per node (length N). If None, axes have no ticks.
    network_labels : network name per node (length N). Used for sorting and
                     boundary lines. If None, original order is kept.
    vmin, vmax     : colour scale limits.
    cmap           : colormap name.
    figsize        : figure size. Auto-calculated if None.
    title          : axes title.
    save_path      : save figure to this path at 150 dpi.

    Returns
    -------
    (fig, ax)
    """
    n = matrix.shape[0]
    if figsize is None:
        s = max(8, min(24, n * 0.08))
        figsize = (s + 1.5, s)

    if network_labels is not None:
        order = sorted(range(n), key=lambda i: (network_labels[i], i))
    else:
        order = list(range(n))

    mat_sorted = matrix[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order] if labels is not None else None
    sorted_nets   = [network_labels[i] for i in order] if network_labels is not None else None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat_sorted, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if sorted_labels is not None:
        tick_fs = max(3, min(8, 120 // n))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(sorted_labels, rotation=90, fontsize=tick_fs)
        ax.set_yticklabels(sorted_labels, fontsize=tick_fs)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if sorted_nets is not None:
        prev = sorted_nets[0]
        for i, net in enumerate(sorted_nets[1:], start=1):
            if net != prev:
                ax.axhline(i - 0.5, color="black", linewidth=0.6, alpha=0.7)
                ax.axvline(i - 0.5, color="black", linewidth=0.6, alpha=0.7)
            prev = net

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

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
    edge_weights  = [d["weight"] for _, _, d in G.edges(data=True)]
    strength_vals = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    node_sizes    = np.clip(strength_vals / strength_vals.max() * node_size_scale, 50, node_size_scale)
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
        deg      = np.degrees(a)
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
