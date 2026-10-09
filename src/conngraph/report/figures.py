"""
Plotly figures for the HTML reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from conngraph.viz.colormap import natural_key

INK = "#2c3e50"
ACCENT = "#2980b9"
_FONT = dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12, color=INK)
_AXIS = dict(showline=True, linecolor="#b8c2cc", gridcolor="#eef0f3", zeroline=False, ticks="outside", tickcolor="#b8c2cc")
_LEVEL_NAMES = {"node": "Node level", "network": "Network level", "network_hemi": "Network level (hemispheres)"}
_FADED = 0.25
# plot height of ordered_heatmap and the spacing its network names need
_HEATMAP_PLOT_PX = 500
_TICK_GAP_PX = 13


def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    layout.setdefault("margin", dict(l=60, r=20, t=30, b=50))
    fig.update_layout(template="none", font=_FONT, height=height, plot_bgcolor="white", paper_bgcolor="white",
                      hoverlabel=dict(font_size=12), **layout)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"scale": 3}})


def network_order(names: list[str], groups: list[str] | None) -> list[int]:
    """Positions sorted by group (naturally, "None" last), keeping the input order within a group."""
    if groups is None:
        return list(range(len(names)))
    return sorted(range(len(names)), key=lambda i: (groups[i] == "None", natural_key(groups[i]), i))


def level_boxplot(df: pd.DataFrame, palette: dict, title: str, hemi_split: bool) -> go.Figure:
    """One box per network node over subjects; hemisphere-split nodes are paired by network."""
    fig = go.Figure()
    if hemi_split:
        split = {c: c.partition("_")[::2] for c in df.columns}
        nets = sorted({net for _, net in split.values()}, key=natural_key)
        hemis = sorted({hemi for hemi, _ in split.values()})
        slot = 0.8 / len(hemis)
        for j, hemi in enumerate(hemis):
            offset = (j - (len(hemis) - 1) / 2) * slot
            opacity = 0.9 if hemi == "L" else 0.4
            for i, net in enumerate(nets):
                col = f"{hemi}_{net}"
                if col in df:
                    fig.add_traces(_box(df[col], i + offset, palette.get(net), slot * 0.8, f"{hemi} {net}",
                                        seed=j * len(nets) + i, opacity=opacity, size=6))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{hemi} hemisphere",
                                     marker=dict(color="#9aa5b1", size=6, opacity=opacity, line=dict(width=0.5, color=INK))))
        _category_axis(fig, nets)
        style(fig, 380, yaxis_title=title, legend=dict(orientation="h", x=1, xanchor="right", y=1.12))
    else:
        cols = sorted(df.columns, key=natural_key)
        for i, col in enumerate(cols):
            fig.add_traces(_box(df[col], i, palette.get(col), 0.6, col, seed=i))
        _category_axis(fig, cols)
        style(fig, 360, yaxis_title=title, showlegend=False)
    return fig


def _box(values: pd.Series, x: float, color: str | None, width: float, hover: str, seed: int, opacity: float = 0.9,
         size: int = 7) -> list:
    """A hollow box with a dashed mean line, and the values as jittered points (hover for the ID)."""
    color = color or "#9aa5b1"
    v = values.to_numpy(float)
    jitter = np.random.default_rng(seed).uniform(-width * 0.32, width * 0.32, len(v))
    return [
        go.Box(x=[x] * len(v), y=v, width=width, boxpoints=False, boxmean=True, line=dict(color=color, width=1.2),
               fillcolor="rgba(0,0,0,0)", hoverinfo="skip", showlegend=False),
        go.Scatter(x=x + jitter, y=v, mode="markers", showlegend=False, text=[str(i) for i in values.index],
                   marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.5, color=INK)),
                   hovertemplate="<b>%{text}</b><br>" + hover + ": %{y:.4g}<extra></extra>"),
    ]


def _category_axis(fig: go.Figure, names: list[str]) -> None:
    fig.update_xaxes(tickvals=list(range(len(names))), ticktext=names, range=[-0.6, len(names) - 0.4], automargin=True)


def node_boxplot(values: pd.Series, networks: pd.Series, palette: dict, title: str) -> go.Figure:
    """One box per network over its regions' values."""
    fig = go.Figure()
    nets = sorted(networks.unique(), key=lambda s: (s == "None", natural_key(s)))
    for i, net in enumerate(nets):
        fig.add_traces(_box(values[networks == net], i, palette.get(net), 0.6, net, seed=i, size=5))
    _category_axis(fig, nets)
    return style(fig, 340, yaxis_title=title, showlegend=False)


def ordered_heatmap(M: np.ndarray, names: list[str], groups: list[str] | None, zmax: float, value: str,
                    marks: list[tuple[str, str]] | None = None) -> go.Figure:
    """Matrix ordered by group with lines at group boundaries; marks are (row, column) labels kept at full colour."""
    order = network_order(names, groups)
    labels = [names[i] for i in order]
    z = np.round(M[np.ix_(order, order)], 4)
    scale = dict(colorscale="RdBu_r", zmid=0, zmin=-zmax, zmax=zmax)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, **scale, opacity=_FADED if marks else None,
        colorbar=dict(title=value, thickness=12), hovertemplate="%{y}<br>%{x}<br>" + value + " = %{z}<extra></extra>"))
    if marks:
        # unmarked cells stay faded underneath; marked ones are drawn again at full strength
        pos = {lab: k for k, lab in enumerate(labels)}
        kept = np.full(z.shape, np.nan)
        for r, c in marks:
            kept[pos[r], pos[c]] = z[pos[r], pos[c]]
        fig.add_trace(go.Heatmap(z=kept, x=labels, y=labels, **scale, showscale=False, hoverinfo="skip"))
    ticks, text = labels, labels
    shapes = []
    if groups is not None:
        ordered = [groups[i] for i in order]
        bounds = [i for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1]]
        line = dict(color="#555", width=0.6)
        n = len(ordered)
        shapes = ([dict(type="line", x0=b - 0.5, x1=b - 0.5, y0=-0.5, y1=n - 0.5, line=line) for b in bounds]
                  + [dict(type="line", y0=b - 0.5, y1=b - 0.5, x0=-0.5, x1=n - 0.5, line=line) for b in bounds])
        if n > 40:
            starts, ends = [0] + bounds, bounds + [n]
            ticks, text = _spaced_group_ticks(labels, ordered, starts, ends, _HEATMAP_PLOT_PX)
    style(fig, 640, shapes=shapes, width=740, margin=dict(l=120, r=20, t=20, b=120))
    fig.update_xaxes(tickvals=ticks, ticktext=text, tickangle=-45, showgrid=False, showline=False)
    fig.update_yaxes(tickvals=ticks, ticktext=text, autorange="reversed", showgrid=False, showline=False)
    return fig


def _spaced_group_ticks(labels: list[str], ordered: list[str], starts: list[int], ends: list[int],
                        plot_px: float) -> tuple[list[str], list[str]]:
    """One tick per group at its middle; when two names would overlap, only the larger group keeps its name."""
    gap = _TICK_GAP_PX * len(labels) / plot_px
    kept: list[tuple[float, int, int]] = []
    for s, e in zip(starts, ends):
        centre = (s + e - 1) / 2
        if kept and centre - kept[-1][0] < gap:
            if e - s > kept[-1][1]:
                kept[-1] = (centre, e - s, s)
            continue
        kept.append((centre, e - s, s))
    return [labels[int(c)] for c, _, _ in kept], [ordered[s] for _, _, s in kept]


def count_heatmap(counts: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=counts.values, x=list(counts.columns), y=list(counts.index), colorscale="Reds",
                               text=counts.values, texttemplate="%{text}", colorbar=dict(title="edges", thickness=12),
                               hovertemplate="%{y} – %{x}: %{z} edges<extra></extra>"))
    size = 160 + 32 * len(counts)
    style(fig, size, width=size + 80, margin=dict(l=110, r=20, t=10, b=110))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def curves_figure(curves: pd.DataFrame, metrics: list[str], titles: dict[str, str], xlabel: str) -> go.Figure:
    """Mean over nodes, then mean ± SD over subjects, at each value of the swept parameter."""
    levels = [lv for lv in _LEVEL_NAMES if lv in set(curves["level"])]
    fig = make_subplots(rows=len(levels), cols=len(metrics), subplot_titles=[titles[m] for m in metrics],
                        vertical_spacing=0.18 if len(levels) > 1 else 0.1, horizontal_spacing=0.07)
    for r, lv in enumerate(levels, 1):
        sub = curves[curves["level"] == lv]
        for c, m in enumerate(metrics, 1):
            per = sub[sub["metric"] == m].groupby(["ID", "threshold"])["value"].mean().unstack()
            x = per.columns.to_numpy(float)
            mu, sd = per.mean().to_numpy(), per.std().fillna(0).to_numpy()
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself", mode="lines",
                                     fillcolor="rgba(41,128,185,0.15)", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(x=x, y=mu, mode="lines+markers", line=dict(color=ACCENT, width=2), marker=dict(size=5),
                                     customdata=sd, showlegend=False,
                                     hovertemplate="%{x}: %{y:.4g} ± %{customdata:.3g}<extra></extra>"), row=r, col=c)
        fig.update_yaxes(title_text=_LEVEL_NAMES[lv], row=r, col=1)
    for c in range(1, len(metrics) + 1):
        fig.update_xaxes(title_text=xlabel, row=len(levels), col=c)
    style(fig, 280 * len(levels) + 40, margin=dict(l=70, r=20, t=40, b=50))
    fig.update_annotations(font_size=12)
    return fig


def null_histogram(null: np.ndarray, sizes: np.ndarray, labels: list[str], significant: list[bool]) -> go.Figure:
    fig = go.Figure(go.Histogram(x=null, nbinsx=50, marker_color="#b8c2cc", hovertemplate="%{x}: %{y}<extra></extra>"))
    for size, label, sig in zip(sizes, labels, significant):
        color = "#c0392b" if sig else "#7f8c8d"
        fig.add_vline(x=size, line_color=color, line_width=2, annotation_text=label, annotation_font_color=color,
                      annotation_position="top")
    return style(fig, 300, xaxis_title="Largest component size under permutation (edges)", yaxis_title="Permutations",
                 bargap=0.05, margin=dict(l=60, r=20, t=40, b=50))


def surface_meshes(surfaces: tuple[str, str] | None, fraction: float = 0.06) -> list[tuple[np.ndarray, np.ndarray]]:
    """Vertices and faces of the surfaces, simplified so the page stays small."""
    if surfaces is None:
        return []
    from conngraph.viz.surface import load_surface

    meshes = []
    for mesh in load_surface(*surfaces):
        if mesh.npoints > 5000:
            mesh = mesh.decimate(fraction=fraction)
        meshes.append((np.asarray(mesh.vertices), np.asarray(mesh.cells)))
    return meshes


def _scene(traces: list, meshes: list, height: int = 580) -> go.Figure:
    surface = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#d9d4ca", opacity=0.12,
                         hoverinfo="skip", showscale=False, lighting=dict(ambient=0.7, diffuse=0.5, specular=0.05))
               for v, f in meshes]
    fig = go.Figure(surface + traces)
    blank = dict(visible=False, showbackground=False)
    fig.update_layout(template="none", font=_FONT, height=height, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis=blank, yaxis=blank, zaxis=blank, aspectmode="data",
                                 camera=dict(eye=dict(x=-1.6, y=0.1, z=0.35), up=dict(x=0, y=0, z=1))),
                      legend=dict(itemsizing="constant"))
    return fig


def brain_values(xyz: np.ndarray, labels: list[str], networks: list[str], values: np.ndarray, title: str,
                 meshes: list) -> go.Figure:
    """Regions coloured and sized by a value, on a rotatable surface."""
    span = np.nanmax(values) - np.nanmin(values)
    size = 4 + 10 * (values - np.nanmin(values)) / (span if span > 0 else 1)
    dots = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", text=labels, customdata=networks, showlegend=False,
        marker=dict(size=size, color=values, colorscale="Viridis", colorbar=dict(title=title, thickness=12, len=0.6),
                    line=dict(width=0), opacity=1),
        hovertemplate="%{text}<br>%{customdata}<br>%{marker.color:.4g}<extra></extra>")
    return _scene([dots], meshes)


def brain_edges(xyz: np.ndarray, labels: list[str], networks: list[str], edges: list[tuple[int, int]],
                weights: np.ndarray, degree: np.ndarray, palette: dict, meshes: list,
                sign_names: tuple[str, str]) -> go.Figure:
    """Edges coloured by the sign of their weight and regions coloured by network, on a rotatable surface."""
    traces = []
    for sign, color, name in ((1, "#c0392b", sign_names[0]), (-1, "#2471a3", sign_names[1])):
        pairs = [(i, j) for (i, j), w in zip(edges, weights) if np.sign(w) == sign]
        if not pairs:
            continue
        pts = np.full((3 * len(pairs), 3), np.nan)
        for n, (i, j) in enumerate(pairs):
            pts[3 * n], pts[3 * n + 1] = xyz[i], xyz[j]
        traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", name=name,
                                   line=dict(color=color, width=2), opacity=0.55, hoverinfo="skip"))
    networks = np.asarray(networks)
    for net in sorted(set(networks), key=lambda s: (s == "None", natural_key(s))):
        idx = np.flatnonzero(networks == net)
        traces.append(go.Scatter3d(
            x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode="markers", name=net,
            marker=dict(size=2.5 + 1.4 * np.sqrt(degree[idx]), color=palette.get(net, "#9aa5b1"), line=dict(width=0)),
            text=[f"{labels[i]}<br>{net}<br>{degree[i]} edges" for i in idx], hovertemplate="%{text}<extra></extra>"))
    return _scene(traces, meshes, 600)


_RING = (1.03, 1.065, 1.15)  # inner and outer radius of the network arcs, radius of the network names
_EDGE_BINS = 10
_NAME_PX = 0.06  # approximate width of a name character in circle radii, for spacing the names


def circos_figure(G, labels: list[str], nets: list[str], palette: dict, colorbar_title: str,
                  bundled: bool = False) -> go.Figure:
    """Nodes on a circle grouped by network: chords coloured by weight, or bundled through networks in their colours."""
    from conngraph.viz.colormap import values_to_widths
    from conngraph.viz.network_graphs import _bundled_path, _edge_colors, _network_hubs, _sample_quadratic

    xy, angles, step = _ring_positions(nets)
    edges = sorted((e for e in G.edges(data="weight", default=1.0) if e[0] != e[1]), key=lambda e: abs(e[2]))
    w = np.array([e[2] for e in edges], dtype=float)
    widths = values_to_widths(np.abs(w), (0.3, 2.5)) if len(edges) else np.array([])
    fig = go.Figure()
    if bundled:
        hubs = _network_hubs(xy, nets)
        halves: dict[str, list] = {}
        for u, v, _ in edges:
            path = _bundled_path(xy[u], xy[v], hubs[nets[u]], hubs[nets[v]], nets[u] == nets[v], 0.85)
            m = len(path) // 2
            # each half takes the colour of the network at its end
            halves.setdefault(nets[u], []).append(path[: m + 1])
            halves.setdefault(nets[v], []).append(path[m:])
        for net, paths in halves.items():
            fig.add_trace(_lines(paths, palette.get(net, "#9aa5b1"), 0.9, 0.35))
    elif len(edges):
        paths = []
        for u, v, _ in edges:
            p0, p2 = xy[u], xy[v]
            # pull the control point towards the centre, further for longer chords
            paths.append(_sample_quadratic(np.array([p0, (p0 + p2) / 2 * (1 - np.linalg.norm(p2 - p0) / 2), p2])))
        colors = _edge_colors(w, "weight", "RdBu_r", ((1, .25, .25), (.25, .25, 1)))
        fig.add_traces(_binned_lines(paths, w, colors, widths, 0.6))
        lim = float(np.abs(w).max()) or 1.0
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip", marker=dict(
            colorscale="RdBu_r", cmin=-lim, cmax=lim, color=[0], showscale=True, colorbar=dict(
                title=dict(text=colorbar_title, side="top"), orientation="h", len=0.45, thickness=10, x=0.98,
                xanchor="right", y=0.02, nticks=5, tickfont=dict(size=9)))))
    traces, names = _ring_traces(G, labels, nets, palette, xy, angles, step)
    fig.add_traces(traces)
    fig.update_layout(annotations=names)
    return _square(fig, 1.4, legend=False)


def spring_figure(G, labels: list[str], nets: list[str], palette: dict, seed: int = 42) -> go.Figure:
    """Spring layout of the graph; node size is strength, colour the network; click a network in the legend to hide it."""
    import networkx as nx

    pos = nx.spring_layout(G, seed=seed, k=0.15)
    xy = np.array([pos[i] for i in range(len(labels))])
    xy = xy - xy.mean(axis=0)
    xy = xy / (np.abs(xy).max() or 1.0)
    edges = [e for e in G.edges(data="weight", default=1.0) if e[0] != e[1]]
    w = np.abs([e[2] for e in edges]).astype(float)
    fig = go.Figure()
    if len(edges):
        widths = 0.25 + 0.75 * np.power(w / (w.max() or 1.0), 1.5)
        fig.add_traces(_binned_lines([xy[[u, v]] for u, v, _ in edges], w, ["#9aa5b1"] * len(w), widths, 0.4, bins=4))
    strength = _strength(G, len(labels))
    peak = strength.max() or 1.0
    for net in _ordered_networks(nets):
        sel = [i for i, x in enumerate(nets) if x == net]
        fig.add_trace(go.Scatter(
            x=xy[sel, 0], y=xy[sel, 1], mode="markers", name=net,
            marker=dict(color=palette.get(net, "#9aa5b1"), size=7 + 11 * strength[sel] / peak, opacity=0.9,
                        line=dict(width=0.5, color="white")),
            **_node_hover(labels, sel, net, G, strength)))
    return _square(fig, 1.1, legend=True)


def _ring_positions(nets: list[str], gap: float = 0.04) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(nets)
    order = sorted(range(n), key=lambda i: (natural_key(nets[i]), i))
    step = (2 * np.pi - gap * len(set(nets))) / n
    angles, a = np.zeros(n), np.pi / 2
    for k, i in enumerate(order):
        if k and nets[i] != nets[order[k - 1]]:
            a += gap
        angles[i] = a
        a += step
    return np.c_[np.cos(angles), np.sin(angles)], angles, step


def _ring_traces(G, labels, nets, palette, xy, angles, step) -> tuple[list, list[dict]]:
    """Nodes and a filled arc per network, and the network names along the circle as annotations."""
    from conngraph.viz.network_graphs import _spread_angles, _text_color

    inner, outer, name_r = _RING
    strength = _strength(G, len(labels))
    traces, mids, names = [], [], []
    for net in _ordered_networks(nets):
        sel = [i for i, x in enumerate(nets) if x == net]
        color = palette.get(net, "#9aa5b1")
        a = np.sort(angles[sel])
        arc = np.linspace(a[0] - step / 2, a[-1] + step / 2, max(8, int(200 * (a[-1] - a[0] + step) / np.pi)))
        unit = np.c_[np.cos(arc), np.sin(arc)]
        ring = np.round(np.r_[outer * unit, inner * unit[::-1]], 4)
        traces.append(go.Scatter(x=ring[:, 0], y=ring[:, 1], mode="lines", fill="toself", fillcolor=color,
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter(x=np.round(xy[sel, 0], 4), y=np.round(xy[sel, 1], 4), mode="markers", name=net,
                                 marker=dict(color=color, size=4), showlegend=False,
                                 **_node_hover(labels, sel, net, G, strength)))
        mids.append(float(np.mean(a)))
        names.append(net)
    order = np.argsort(mids)
    spread = _spread_angles([mids[k] for k in order], [len(names[k]) * _NAME_PX / name_r for k in order], 0.04)
    notes = []
    for k, theta in zip(order, spread):
        rotation = (np.degrees(theta) - 90 + 180) % 360 - 180
        if abs(rotation) > 90:
            rotation += 180
        r, g, b = (round(255 * c) for c in _text_color(palette.get(names[k], "#9aa5b1")))
        notes.append(dict(x=name_r * np.cos(theta), y=name_r * np.sin(theta), text=names[k], showarrow=False,
                          textangle=-rotation, font=dict(size=10, color=f"rgb({r},{g},{b})")))
    return traces, notes


def _binned_lines(paths: list, values: np.ndarray, colors: list, widths: np.ndarray, opacity: float,
                  bins: int = _EDGE_BINS) -> list:
    """Edges grouped into a few traces of similar value, so hundreds of edges stay quick to draw."""
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)
    traces = []
    for b in range(bins):
        sel = np.flatnonzero(which == b)
        if len(sel):
            mid = sel[np.argsort(values[sel])[len(sel) // 2]]
            traces.append(_lines([paths[k] for k in sel], colors[mid], float(widths[mid]) * 1.2, opacity))
    return traces


def _lines(paths: list, color, width: float, opacity: float) -> go.Scatter:
    xs, ys = [], []
    for p in paths:
        xs += np.round(p[:, 0], 4).tolist() + [None]
        ys += np.round(p[:, 1], 4).tolist() + [None]
    if not isinstance(color, str):
        color = "rgb({},{},{})".format(*(round(255 * c) for c in color[:3]))
    return go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False, opacity=opacity,
                      line=dict(color=color, width=width))


def _strength(G, n: int) -> np.ndarray:
    return np.array([sum(abs(d.get("weight", 1.0)) for j, d in G.adj[i].items() if j != i) for i in range(n)])


def _node_hover(labels, sel, net, G, strength) -> dict:
    degree = [G.degree(i) for i in sel]
    return dict(customdata=np.c_[[labels[i] for i in sel], degree, np.round(strength[sel], 4)].astype(object),
                hovertemplate="<b>%{customdata[0]}</b><br>" + str(net)
                              + "<br>%{customdata[1]} edges, strength %{customdata[2]}<extra></extra>")


def _ordered_networks(nets: list[str]) -> list[str]:
    return sorted(set(nets), key=lambda s: (s == "None", natural_key(s)))


def _square(fig: go.Figure, lim: float, legend: bool) -> go.Figure:
    axis = dict(visible=False, range=[-lim, lim])
    fig.update_layout(template="none", font=_FONT, height=520, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", xaxis=axis, yaxis=dict(axis, scaleanchor="x"),
                      showlegend=legend, legend=dict(font=dict(size=10), itemsizing="constant", orientation="h",
                                                     x=0.5, xanchor="center", y=0, yanchor="top"),
                      hoverlabel=dict(font_size=12))
    return fig
