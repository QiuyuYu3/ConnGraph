"""
Multi-view figures: render several camera views of one scene and lay them out with a legend strip.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MaxNLocator

from brainnet3d.exceptions import DataValidationError
from brainnet3d.viz.nodes import _resolve_vminvmax

VIEW_DIRECTIONS: dict[str, tuple[tuple, tuple]] = {
    "L": ((-1, 0, 0), (0, 0, 1)),
    "R": (( 1, 0, 0), (0, 0, 1)),
    "S": ((0, 0,  1), (0, 1, 0)),
    "I": ((0, 0, -1), (0, 1, 0)),
    "A": ((0,  1, 0), (0, 0, 1)),
    "P": ((0, -1, 0), (0, 0, 1)),
}
_VIEW_NAMES   = dict(L="left", R="right", S="superior", I="inferior", A="anterior", P="posterior")
_PANEL_KEYS   = {"view", "hemisphere", "title"}
LEGEND_NAMES  = ("node_color", "node_size", "edge_color", "edge_width")

# reference size: a 600-pixel panel is 4 inches wide and line widths are used as given
_REF_PANEL_PX   = 600
_PANEL_INCHES   = 4.0
_MARGIN, _GAP   = 0.15, 0.2
_TITLE_H        = 0.3
_FONT, _TITLE_FONT = 9, 10


def parse_views(views) -> list[list[dict]]:
    if views is None:
        views = [{"view": "L"}, {"view": "S"}, {"view": "R"}]
    rows = [views] if all(isinstance(v, dict) for v in views) else views

    parsed = []
    for row in rows:
        if isinstance(row, dict) or not all(isinstance(p, dict) for p in row):
            raise TypeError("views must be a list of panel dicts, or a list of such lists (one per row).")
        parsed.append([_parse_panel(p) for p in row])

    if not any(parsed):
        raise ValueError("views has no panels.")
    if not all(parsed):
        raise ValueError("views has an empty row; remove it or give it panels.")
    return parsed


def _parse_panel(panel: dict) -> dict:
    unknown = set(panel) - _PANEL_KEYS
    if unknown:
        raise ValueError(f"Unknown panel keys {sorted(unknown)}; use 'view', 'hemisphere' and 'title'.")

    view = str(panel.get("view", "")).upper()
    if view not in VIEW_DIRECTIONS:
        raise ValueError(f"Panel view {panel.get('view')!r} not recognised. Choose from: L, R, S, I, A, P.")

    hemi = panel.get("hemisphere", "both")
    if hemi != "both":
        hemi = str(hemi).upper()
        if hemi not in ("L", "R"):
            raise ValueError(f"Panel hemisphere {panel['hemisphere']!r} not recognised. Choose from: 'both', 'L', 'R'.")
    return dict(view=view, hemisphere=hemi, title=panel.get("title"))


def panel_title(view: str, hemisphere: str) -> str:
    if hemisphere == "both":
        return _VIEW_NAMES[view].capitalize()
    side = ("lateral" if view == hemisphere else "medial") if view in "LR" else _VIEW_NAMES[view]
    return f"{'Left' if hemisphere == 'L' else 'Right'} hemisphere, {side}"


def panel_actors(scene, hemisphere: str) -> list:
    if hemisphere == "both":
        return scene.actors

    nodes_df = scene.nodes_df
    if "hemisphere" not in nodes_df.columns:
        raise DataValidationError(
            f"A panel with hemisphere='{hemisphere}' needs a 'hemisphere' column in the nodes table."
        )
    keep = (nodes_df["hemisphere"].astype(str).str.upper() == hemisphere).to_numpy()
    if not keep.any():
        raise DataValidationError(
            f"No nodes found for hemisphere='{hemisphere}'. Check the 'hemisphere' column in your nodes file."
        )

    return (
        [m for m in scene.surfaces if m._hemisphere == hemisphere]
        + [s for s in scene.nodes if keep[s._node_idx]]
        + [e for e in scene.edges if keep[e._endpoints[0]] and keep[e._endpoints[1]]]
        + scene.extras
    )


def _bounds(actors: list) -> tuple[np.ndarray, np.ndarray]:
    b = np.array([a.bounds() for a in actors], dtype=float)
    return b[:, 0::2].min(axis=0), b[:, 1::2].max(axis=0)


def render_panel(actors: list, view: str, parallel_scale: float, size: int, background) -> np.ndarray:
    from vedo import Plotter

    direction, viewup = VIEW_DIRECTIONS[view]
    lo, hi = _bounds(actors)
    centre = (lo + hi) / 2

    vp = Plotter(offscreen=True, size=(size, size), bg=background, axes=0)
    # depth peeling sorts translucent surfaces, which otherwise show dark streaks
    vp.window.SetAlphaBitPlanes(1)
    vp.window.SetMultiSamples(0)
    vp.renderer.SetUseDepthPeeling(True)
    vp.renderer.SetMaximumNumberOfPeels(20)
    vp.renderer.SetOcclusionRatio(0.0)

    camera = dict(pos=centre + 10 * parallel_scale * np.asarray(direction), focal_point=centre, viewup=viewup)
    vp.show(actors, interactive=False, camera=camera)
    vp.camera.SetParallelProjection(True)
    vp.camera.SetParallelScale(parallel_scale)
    vp.renderer.ResetCameraClippingRange()
    vp.render()
    image = np.asarray(vp.screenshot(asarray=True))
    vp.close()
    return _crop(image, background, pad=max(2, size // 100))


def _crop(image: np.ndarray, background, pad: int) -> np.ndarray:
    bg = np.array(mcolors.to_rgb(background)) * 255
    mask = np.abs(image[..., :3].astype(float) - bg).max(axis=2) > 8
    if not mask.any():
        return image
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, image.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def views_figure(scene, a: dict, panels: list[list[dict]], legend, panel_size: int, titles: bool,
                 width: float | None = None, legend_titles: dict | None = None) -> Figure:
    lo, hi = _bounds(scene.actors)
    parallel_scale = 0.5 * float((hi - lo).max()) * 1.03
    px_per_mm = panel_size / (2 * parallel_scale)
    dpi = panel_size / _PANEL_INCHES

    # keep line widths proportional to the panel so larger renders look the same
    width_factor = panel_size / _REF_PANEL_PX
    if width is not None:
        # panels are shown at a set width, so thicken lines by the same ratio to print as thick as by default
        widths_px = [[_panel_width_px(scene, p, px_per_mm, panel_size) for p in row] for row in panels]
        width_factor *= _fit_dpi(widths_px, width) / dpi
    if width_factor != 1 and not a["use_tube"]:
        for e in scene.edges:
            e.lw(e.properties.GetLineWidth() * width_factor)

    images = [
        [render_panel(panel_actors(scene, p["hemisphere"]), p["view"], parallel_scale, panel_size, a["background"]) for p in row]
        for row in panels
    ]
    labels = [
        [(p["title"] if p["title"] is not None else panel_title(p["view"], p["hemisphere"])) if titles else None for p in row]
        for row in panels
    ]
    items = legend_items(scene, a, legend, width_factor, legend_titles)

    fg = "white" if np.mean(mcolors.to_rgb(a["background"])) < 0.5 else "black"
    style = {"text.color": fg, "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg, "axes.edgecolor": fg}
    with plt.rc_context(style):
        return _compose(images, labels, items, dpi, px_per_mm, a["background"], has_titles=titles, width=width)


def _panel_width_px(scene, panel: dict, px_per_mm: float, size: int) -> float:
    # width of the cropped render, known before rendering from the extent across the view
    lo, hi = _bounds(panel_actors(scene, panel["hemisphere"]))
    across = 1 if panel["view"] in "LR" else 0
    return min(size, (hi - lo)[across] * px_per_mm + 2 * max(2, size // 100) + 1)


def _fit_dpi(widths_px: list[list[float]], width: float) -> float:
    room = [width - 2 * _MARGIN - _GAP * (len(row) - 1) for row in widths_px]
    if min(room) <= 0:
        raise ValueError(f"width={width} leaves no room for the panels; use a larger width.")
    return max(sum(row) / r for row, r in zip(widths_px, room))


def legend_items(scene, a: dict, legend, width_factor: float = 1.0, legend_titles: dict | None = None) -> list[dict]:
    titles = dict(legend_titles or {})
    unknown = set(titles) - set(LEGEND_NAMES)
    if unknown:
        raise ValueError(f"Unknown legend_titles entries {sorted(unknown)}. Choose from: {', '.join(LEGEND_NAMES)}.")
    if legend is False:
        return []
    if legend is True:
        names = list(LEGEND_NAMES[:3])
    else:
        names = [legend] if isinstance(legend, str) else list(legend)
    unknown = set(names) - set(LEGEND_NAMES)
    if unknown:
        raise ValueError(f"Unknown legend entries {sorted(unknown)}. Choose from: {', '.join(LEGEND_NAMES)}.")

    items = []
    for name in names:
        item = _legend_item(name, scene, a, width_factor)
        if item is not None:
            if name in titles:
                item["title"] = str(titles[name])
            items.append(item)
        elif legend is not True:
            warnings.warn(f"No legend for '{name}': that style is not mapped to data.", stacklevel=4)
    return items


def _legend_item(name: str, scene, a: dict, width_factor: float) -> dict | None:
    nodes_df = scene.nodes_df
    weights = np.array([e._weight for e in scene.edges], dtype=float)

    if name == "node_color":
        col = a["node_color"]
        if not (isinstance(col, str) and col in nodes_df.columns):
            return None
        values = nodes_df[col]
        if pd.api.types.is_numeric_dtype(values):
            vmin, vmax = _resolve_vminvmax(values.to_numpy(float), a["node_colorvminvmax"])
            return dict(name=name, kind="colorbar", title=col, cmap=a["node_cmap"] or "viridis", vmin=vmin, vmax=vmax)
        entries = {str(k): c for k, c in zip(values.astype(str), scene.node_colors)}
        return dict(name=name, kind="categories", title=col, entries=entries)

    if name == "node_size":
        col = a["node_size"]
        if not isinstance(col, str):
            return None
        values = nodes_df[col].to_numpy(float)
        shown, scaled = _representative(values, a["node_size_range"])
        return dict(name=name, kind="sizes", title=col, values=shown, radii_mm=scaled)

    if name == "edge_color":
        if a["edge_color"] == "sign" and weights.size:
            pos, neg = (mcolors.to_rgb(c) for c in a["edge_sign_colors"])
            entries = {"positive": pos} if (weights >= 0).any() else {}
            if (weights < 0).any():
                entries["negative"] = neg
            return dict(name=name, kind="categories", title="edge sign", entries=entries)
        if a["edge_color"] != "weight" or weights.size == 0:
            return None
        vmin, vmax = _resolve_vminvmax(weights, a["edge_colorvminvmax"])
        return dict(name=name, kind="colorbar", title="edge weight", cmap=a["edge_cmap"], vmin=vmin, vmax=vmax)

    if name == "edge_width":
        if isinstance(a["edge_width"], (int, float)) or weights.size == 0 or a["use_tube"]:
            return None
        shown, scaled = _representative(np.abs(weights), a["edge_width_range"])
        return dict(name=name, kind="widths", title="|edge weight|", values=shown, widths_px=scaled * width_factor)

    return None


def _representative(values: np.ndarray, out_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi == lo:
        return np.array([lo]), np.array([(out_range[0] + out_range[1]) / 2])
    shown = _round_values(lo, hi)
    return shown, out_range[0] + (shown - lo) / (hi - lo) * (out_range[1] - out_range[0])


def _round_values(lo: float, hi: float) -> np.ndarray:
    # three round numbers inside the data range, as tick labels would pick; fall back to min, middle, max
    for bins in range(3, 9):
        ticks = MaxNLocator(nbins=bins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        if len(ticks) >= 3:
            # with an even count, drop the smallest so the three values are evenly spaced
            ticks = ticks[1:] if len(ticks) % 2 == 0 else ticks
            return ticks[[0, len(ticks) // 2, -1]]
    return np.linspace(lo, hi, 3)


def _number(value: float) -> str:
    if float(value).is_integer() and abs(value) < 1e6:
        return f"{int(value):,}"
    return f"{value:.3g}" if abs(value) < 1000 else f"{value:,.0f}"


def _legend_size(item: dict, dpi: float, px_per_mm: float, avail: float = math.inf) -> tuple[float, float]:
    line = _FONT / 72 * 1.5
    if item["kind"] == "categories":
        n = len(item["entries"])
        longest = max(len(k) for k in item["entries"])
        col_w = 0.45 + 0.075 * longest
        full = math.ceil(n / min(n, 5))
        item["ncol"] = full if math.isinf(avail) else min(full, max(1, int(avail // col_w)))
        nrows = min(n, 5) if item["ncol"] == full else math.ceil(n / item["ncol"])
        return item["ncol"] * col_w, 0.35 + nrows * line
    if item["kind"] == "colorbar":
        return min(2.4, avail), 0.85
    if item["kind"] == "sizes":
        diam = 2 * item["radii_mm"] * px_per_mm / dpi
        return max(1.6, diam.sum() + 0.4 * (len(diam) + 1)), 0.35 + diam.max() + 0.3
    return 1.6, 0.35 + len(item["values"]) * line


def _compose(images, labels, items, dpi: float, px_per_mm: float, background, has_titles: bool,
             width: float | None = None) -> Figure:
    title_h = _TITLE_H if has_titles else 0.0
    avail = math.inf if width is None else width - 2 * _MARGIN
    if width is not None:
        # display the same pixels at the dpi that lets the widest row fill the width, so resolution is kept
        dpi = _fit_dpi([[im.shape[1] for im in row] for row in images], width)
    sizes = [[(im.shape[1] / dpi, im.shape[0] / dpi) for im in row] for row in images]
    row_w = [sum(w for w, _ in row) + _GAP * (len(row) - 1) for row in sizes]
    row_h = [max(h for _, h in row) + title_h for row in sizes]
    title_size = _TITLE_FONT + 1
    if width is not None:
        fits = [_title_size(t, w + _GAP) for row, ws in zip(labels, sizes) for t, (w, _) in zip(row, ws) if t]
        title_size = min(fits, default=title_size)

    legend_sizes = [_legend_size(it, dpi, px_per_mm, avail) for it in items]
    legend_rows = _pack_rows([w for w, _ in legend_sizes], avail)
    legend_w = [sum(legend_sizes[i][0] for i in r) + _GAP * (len(r) - 1) for r in legend_rows]
    legend_h = [max(legend_sizes[i][1] for i in r) for r in legend_rows]

    fig_w = width if width is not None else max(max(row_w), max(legend_w, default=0.0)) + 2 * _MARGIN
    fig_h = 2 * _MARGIN + sum(row_h) + _GAP * (len(images) - 1) + sum(_GAP + h for h in legend_h)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=background)

    def add_axes(x, top, w, h, label):
        return fig.add_axes((x / fig_w, (fig_h - top - h) / fig_h, w / fig_w, h / fig_h), label=label)

    top = _MARGIN
    for r, (row, row_sizes, rw, rh) in enumerate(zip(images, sizes, row_w, row_h)):
        x = (fig_w - rw) / 2
        for c, (im, (w, h)) in enumerate(zip(row, row_sizes)):
            ax = add_axes(x, top + title_h + (rh - title_h - h) / 2, w, h, f"panel:{r},{c}")
            ax.imshow(im, interpolation="none")
            ax.set_axis_off()
            if labels[r][c]:
                fig.text((x + w / 2) / fig_w, 1 - (top + title_h / 2) / fig_h, labels[r][c],
                         ha="center", va="center", fontsize=title_size)
            x += w + _GAP
        top += rh + _GAP

    for r, rw, rh in zip(legend_rows, legend_w, legend_h):
        x = (fig_w - rw) / 2
        for i in r:
            _draw_legend(add_axes, items[i], x, top, legend_sizes[i][0], rh, dpi, px_per_mm)
            x += legend_sizes[i][0] + _GAP
        top += rh + _GAP
    return fig


def _pack_rows(widths: list[float], avail: float) -> list[list[int]]:
    rows: list[list[int]] = []
    used = 0.0
    for i, w in enumerate(widths):
        if rows and used + _GAP + w <= avail:
            rows[-1].append(i)
            used += _GAP + w
        else:
            rows.append([i])
            used = w
    return rows


def _title_size(text: str, room: float) -> float:
    # shrink a title that would run past its panel, down to 6 pt; 0.55 em is a typical average glyph width
    size = _TITLE_FONT + 1
    estimate = 0.55 * size / 72 * len(text)
    return size if estimate <= room else max(6.0, size * room / estimate)


def _draw_legend(add_axes, item: dict, x: float, top: float, w: float, h: float, dpi: float, px_per_mm: float) -> None:
    label = f"legend:{item['name']}"
    kind = item["kind"]

    if kind == "colorbar":
        cax = add_axes(x + 0.1 * w, top + 0.25, 0.8 * w, 0.14, label)
        norm = mcolors.Normalize(item["vmin"], item["vmax"])
        bar = cax.figure.colorbar(ScalarMappable(norm=norm, cmap=item["cmap"]), cax=cax, orientation="horizontal")
        if item["vmax"] > item["vmin"]:
            # three round ticks, as in the size legend; the default locator crowds a narrow bar
            bar.set_ticks(_round_values(item["vmin"], item["vmax"]))
        bar.ax.tick_params(labelsize=_FONT)
        cax.set_title(item["title"], fontsize=_TITLE_FONT)
        return

    ax = add_axes(x, top, w, h, label)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()

    if kind == "categories":
        handles = [Patch(facecolor=c, edgecolor="0.4", linewidth=0.5, label=k) for k, c in item["entries"].items()]
        ax.legend(handles=handles, loc="upper center", ncol=item["ncol"], frameon=False,
                  title=item["title"], fontsize=_FONT, title_fontsize=_TITLE_FONT, borderaxespad=0, columnspacing=1.0)
        return

    ax.text(w / 2, h, item["title"], ha="center", va="top", fontsize=_TITLE_FONT)
    if kind == "sizes":
        radii = item["radii_mm"] * px_per_mm / dpi
        gap = (w - 2 * radii.sum()) / (len(radii) + 1)
        cy = h - 0.3 - radii.max()
        cx = gap
        for value, r in zip(item["values"], radii):
            ax.add_patch(Circle((cx + r, cy), r, color="0.6", linewidth=0))
            ax.text(cx + r, cy - radii.max() - 0.05, _number(value), ha="center", va="top", fontsize=_FONT)
            cx += 2 * r + gap
        return

    line = _FONT / 72 * 1.5
    for k, (value, width_px) in enumerate(zip(item["values"], item["widths_px"])):
        y = h - 0.35 - (k + 0.5) * line
        ax.add_line(Line2D([0.25 * w, 0.5 * w], [y, y], linewidth=width_px * 72 / dpi, color="0.4"))
        ax.text(0.55 * w, y, _number(value), va="center", fontsize=_FONT)
