"""
Plotly figures for the HTML reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from brainnet3d.viz.colormap import natural_key

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
        for hemi in hemis:
            for net in nets:
                col = f"{hemi}_{net}"
                if col in df:
                    fig.add_trace(_box(df[col], net, palette.get(net), name=f"{hemi} hemisphere", legendgroup=hemi,
                                       showlegend=net == nets[0], offsetgroup=hemi, opacity=0.9 if hemi == "L" else 0.45,
                                       hover=f"{hemi} "))
        style(fig, 380, boxmode="group", yaxis_title=title, legend=dict(orientation="h", x=1, xanchor="right", y=1.12))
    else:
        for col in sorted(df.columns, key=natural_key):
            fig.add_trace(_box(df[col], col, palette.get(col), name=col, showlegend=False))
        style(fig, 360, yaxis_title=title)
    return fig


def _box(values: pd.Series, x: str, color: str | None, hover: str = "", **kw) -> go.Box:
    color = color or "#9aa5b1"
    return go.Box(y=values, x=[x] * len(values), boxpoints="all", jitter=0.35, pointpos=0,
                  marker=dict(color=color, size=4, opacity=0.8), line=dict(color=INK, width=1), fillcolor=color,
                  text=[str(i) for i in values.index], hovertemplate="%{text}<br>" + hover + "%{x}: %{y:.4g}<extra></extra>", **kw)


def node_boxplot(values: pd.Series, networks: pd.Series, palette: dict, title: str) -> go.Figure:
    """One box per network over its regions' values."""
    fig = go.Figure()
    for net in sorted(networks.unique(), key=lambda s: (s == "None", natural_key(s))):
        sel = networks == net
        fig.add_trace(_box(values[sel], net, palette.get(net), name=net, showlegend=False))
    return style(fig, 340, yaxis_title=title)


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
    from brainnet3d.viz.surface import load_surface

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
