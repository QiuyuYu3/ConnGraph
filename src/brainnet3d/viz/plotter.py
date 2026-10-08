"""
BrainNetPlotter: the main user-facing visualisation class.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd

from brainnet3d.core.dataset import ConnectivityDataset
from brainnet3d.exceptions import DataValidationError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from brainnet3d.graph_theory.runner import GraphMetricsResult
from brainnet3d.viz.surface import load_surface
from brainnet3d.viz.nodes   import build_nodes, _resolve_colors
from brainnet3d.viz.edges   import build_edges
from brainnet3d.viz.views   import make_axis_arrows


class _Scene(NamedTuple):
    nodes_df:    pd.DataFrame
    surfaces:    list
    nodes:       list
    edges:       list
    extras:      list
    node_colors: list

    @property
    def actors(self) -> list:
        return self.surfaces + self.nodes + self.edges + self.extras


_NOT_IN_PLOT_VIEWS = {
    "show_hemisphere":    "set 'hemisphere' in each panel instead",
    "screenshot":         "save the returned figure with fig.savefig",
    "html":               "use plot(html=...) for an interactive page",
    "interactive":        "use plot(interactive=True) for a window",
    "highlight_on_click": "use plot(interactive=True, highlight_on_click=True)",
    "title":              "set 'title' in each panel or call fig.suptitle",
}


class BrainNetPlotter:
    """
    3-D interactive brain network visualiser built on vedo.

    Quick start
    -----------
    >>> from brainnet3d.loaders import load
    >>> from brainnet3d.viz.plotter import BrainNetPlotter
    >>>
    >>> dataset = load("matrix.csv", "nodes.csv")
    >>> p = BrainNetPlotter(dataset)
    >>> p.plot()

    Parameters
    ----------
    dataset : ConnectivityDataset
    subject_id : str
        Which subject's matrix to show.  Pass "mean" to average all subjects.
    """

    def __init__(
        self,
        dataset:    ConnectivityDataset,
        subject_id: str = "single",
    ):
        self.dataset     = dataset
        self.subject_id  = subject_id
        self._extra_cols: dict = {}

    def attach_metrics(
        self,
        result: GraphMetricsResult,
        metrics: list[str] | None = None,
    ) -> None:
        """
        Attach node-level graph metrics so they can be used as node_color / node_size.

        After calling this, pass the metric name directly to plot():
            p.attach_metrics(result)
            p.plot(node_color="strength", node_size="btwn_cent")

        Parameters
        ----------
        result  : GraphMetricsResult from compute_graph_metrics (needs node_df).
        metrics : subset of metrics to attach, e.g. ["strength", "btwn_cent"].
                  Defaults to all metrics found in result.node_df.
        """
        if result.node_df is None:
            raise ValueError(
                "result.node_df is None — run compute_graph_metrics with "
                "level='node' or level='both'."
            )

        sid = self.subject_id
        if sid not in result.node_df.index:
            raise KeyError(
                f"subject_id='{sid}' not found in result.node_df. "
                f"Available: {list(result.node_df.index)}"
            )

        row       = result.node_df.loc[sid]
        available = row.index.get_level_values(0).unique()
        chosen    = metrics if metrics is not None else available

        self._extra_cols = {
            metric: row[metric].to_dict() for metric in chosen if metric in available
        }

    def plot(
        self,
        # Surface
        surface_L:      str | None              = None,
        surface_R:      str | None              = None,
        surface_color:  str | tuple          = (0.93, 0.90, 0.84),
        surface_alpha:  float                      = 0.15,
        show_surface:   bool                       = True,

        # Nodes
        node_size:           str | float     = 3.0,
        node_size_range:     tuple[float, float]   = (2.0, 8.0),
        node_color:          str | tuple     = "network",
        node_cmap:           str | None            = None,
        node_colorvminvmax:  str | tuple | None = "minmax",
        node_alpha:          float                 = 1.0,
        node_res:        int                       = 16,
        node_palette:    dict | None            = None,

        # Edges
        edge_threshold:      float                 = 0.5,
        edge_threshold_dir:  str                   = "absabove",
        edge_width:          str | float     = "weight",
        edge_width_range: tuple[float, float]      = (0.5, 4.0),
        edge_color:      str | tuple         = "weight",
        edge_cmap:       str                       = "RdBu_r",
        edge_colorvminvmax: str | tuple | None  = "absmax",
        edge_alpha:      float                     = 0.7,
        use_tube:        bool                      = False,

        # Display
        background:      str                       = "white",
        title:           str                       = "Brain Network",
        show_hemisphere: str                       = "both",
        screenshot:      str | None             = None,
        html:            str | None             = None,
        interactive:     bool                      = False,
        highlight_on_click: bool                   = False,
        layout:          str | None             = None,
        layout_seed:     int                       = 42,
        arrowaxis:       str | list[str] | None = None,
        highlight_edges: np.ndarray | None      = None,
        highlight_level: float                     = 0.85,
        edge_sign_colors: tuple                    = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0)),
    ) -> np.ndarray | None:
        """
        Render the brain network off screen and return the image; optionally open a window or write files.

        Parameters
        ----------
        surface_L, surface_R : paths to .surf.gii hemisphere files.
        surface_color : colour of the brain surface mesh.
        surface_alpha : surface transparency (0=invisible, 1=opaque).
        show_surface  : master switch for surface rendering.

        node_size : float → uniform radius (mm). str → column in nodes.csv.
        node_size_range : (min_r, max_r) in mm.
        node_color : RGB tuple / colour name → uniform.
                     "network" / "hemisphere" / column name → categorical.
                     numeric column name → continuous colormap.
        node_cmap  : colormap for node_color. None → "viridis" for numeric columns;
                     "Set3" for up to 12 categories, else "tab20".
        node_colorvminvmax : colour limits for numeric node_color.
                     "minmax" → data min to max. "absmax" → symmetric ±max(|values|).
                     (vmin, vmax) tuple → explicit limits.
        node_alpha : node transparency.
        node_res   : sphere tessellation (higher = smoother).
        node_palette : {category_label: colour} override for categorical colouring.

        edge_threshold : cutoff applied according to edge_threshold_dir.
        edge_threshold_dir : "absabove" → keep |weight| > threshold.
                     "above" → keep weight > threshold. "below" → keep weight < -threshold.
        edge_width : float → uniform. "weight" → scaled to edge_width_range.
        edge_width_range : (min_w, max_w) in pixels.
        edge_color : "weight" → colormap. "node" → inherits node colour.
                     "sign" → positive and negative edges in edge_sign_colors.
                     RGB tuple / colour name → uniform.
        edge_sign_colors : (positive, negative) colours for edge_color="sign".
        edge_cmap  : colormap for edge_color="weight".
        edge_colorvminvmax : colour limits for edge_color="weight".
                     "absmax" (default) → symmetric ±max(|weight|), so 0 sits at the colormap centre.
                     "minmax" → data min to max. (vmin, vmax) tuple → explicit limits.
        edge_alpha : edge transparency.
        use_tube   : use 3-D Tube instead of flat Line (slower).

        background : "white" or "black".
        title      : window title.
        show_hemisphere : "both" | "L" | "R" — filter nodes/surface by hemi.
        screenshot : path to save a PNG snapshot. None = skip.
        html       : path to save a standalone interactive HTML page (needs brainnet3d[html]).
        interactive : if True, open a vedo window (needed for highlight_on_click) and return None.
        highlight_on_click : click a node to highlight its edges and grey out
            all others.  Click the same node again or click empty space to reset.
        layout : None → use MNI coordinates from nodes_df.
                 "spring" → NetworkX Fruchterman-Reingold 3-D layout.
                 "kamada_kawai" → NetworkX Kamada-Kawai 3-D layout (edge length 1/|w|).
                 "spectral" → NetworkX spectral 3-D layout.
                 "forceatlas2" → NetworkX ForceAtlas2 3-D layout weighted by |w|.
                 Brain surface is automatically hidden when a layout is used.
        layout_seed : random seed for "spring", "kamada_kawai" and "forceatlas2" layout reproducibility.
        arrowaxis : add orientation arrows: "all", one of "LR", "AP", "SI", or a list of them.
        highlight_edges : (N, N) array marking edges to keep fully visible (e.g. NBSResult.adj),
                          ordered like the dataset matrix labels. Other edges are dimmed.
        highlight_level : dimming of non-highlighted edges (0 = none, 1 = invisible).
        """
        # every argument, so plot_views can build the same scene from one dict
        args = dict(locals())
        del args["self"]
        from vedo import Plotter

        scene      = self._build_scene(args, show_hemisphere)
        actors     = scene.actors
        edge_lines = scene.edges

        # Render
        plt = Plotter(title=title, bg=background, axes=0, offscreen=not interactive)

        if highlight_on_click and edge_lines:
            node_to_edges: dict = {}
            for ea in edge_lines:
                i, j = ea._endpoints
                node_to_edges.setdefault(i, []).append(ea)
                node_to_edges.setdefault(j, []).append(ea)

            _selected = [None]

            def _on_click(evt):
                actor    = evt.actor
                node_idx = getattr(actor, "_node_idx", None)

                if node_idx is None:
                    for ea in edge_lines:
                        ea.color(ea._orig_color).alpha(ea._orig_alpha)
                    _selected[0] = None
                    plt.render()
                    return

                if node_idx == _selected[0]:
                    for ea in edge_lines:
                        ea.color(ea._orig_color).alpha(ea._orig_alpha)
                    _selected[0] = None
                else:
                    _selected[0] = node_idx
                    connected_ids = {id(ea) for ea in node_to_edges.get(node_idx, [])}
                    for ea in edge_lines:
                        if id(ea) in connected_ids:
                            ea.color(ea._orig_color).alpha(min(ea._orig_alpha * 1.4, 1.0))
                        else:
                            ea.color((0.55, 0.55, 0.55)).alpha(0.06)

                plt.render()

            plt.add_callback("LeftButtonPress", _on_click)

        from brainnet3d.viz.views import _finish_render
        return _finish_render(plt, actors, interactive, screenshot=screenshot, html=html)

    def plot_views(
        self,
        views:      list | None       = None,
        legend:     bool | list[str]  = True,
        panel_size: int               = 600,
        titles:     bool              = True,
        **kwargs,
    ) -> Figure:
        """
        Render several camera views off screen and return them as one matplotlib Figure with a legend.

        Example
        -------
        >>> fig = p.plot_views(
        ...     views=[[{"view": "L", "hemisphere": "L"}, {"view": "R", "hemisphere": "L"}],
        ...            [{"view": "R", "hemisphere": "R"}, {"view": "L", "hemisphere": "R"}]],
        ...     node_color="network", node_size="strength", surface_alpha=0.5,
        ... )
        >>> fig.savefig("network.png")

        Parameters
        ----------
        views : panels as dicts; a list of dicts is one row, a list of such lists is several rows.
                Default: [{"view": "L"}, {"view": "S"}, {"view": "R"}].
                Panel keys:
                "view" (required) → side the camera looks from: "L", "R", "S", "I", "A" or "P".
                "hemisphere" → "both" (default), "L" or "R"; drops the other hemisphere's
                               nodes, surface and any edge touching it.
                "title" → panel title; otherwise generated, e.g. "Left hemisphere, medial".
        legend : True → a legend for every style mapped to data: node colour (category swatches
                 or colour bar), node size and edge colour. False → no legend.
                 List of "node_color", "node_size", "edge_color", "edge_width" → just those.
        panel_size : render size of each panel in pixels before cropping. A 600-pixel panel is
                     4 inches wide; the figure DPI scales with panel_size, so fig.savefig keeps
                     the rendered resolution and the layout looks the same at any size.
        titles : draw a title above each panel.
        **kwargs : any style argument of plot(), e.g. node_color, node_size, edge_threshold,
                   surface_L, surface_alpha, background. Not accepted: show_hemisphere
                   (use the panel "hemisphere" key), screenshot (use fig.savefig), html,
                   interactive, highlight_on_click and title.

        Notes
        -----
        Colour and size ranges are computed over all nodes and edges, so they match across
        panels and may differ from plot(show_hemisphere=...), which uses one hemisphere only.
        Panels use a parallel projection at one common scale, so the node size legend matches
        the drawn spheres. With the default surface_alpha a lateral and a medial panel look
        alike; raise surface_alpha (around 0.5) so nodes behind the surface fade.
        """
        from brainnet3d.viz.panels import parse_views, views_figure

        args   = self._plot_args(**kwargs)
        panels = parse_views(views)
        scene  = self._build_scene(args, "both")
        return views_figure(scene, args, panels, legend, panel_size, titles)

    def _plot_args(self, **kwargs) -> dict:
        for name, hint in _NOT_IN_PLOT_VIEWS.items():
            if name in kwargs:
                raise TypeError(f"plot_views() does not take '{name}': {hint}.")
        bound = inspect.signature(self.plot).bind(**kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _build_scene(self, a: dict, hemisphere: str) -> _Scene:
        nodes_df = self._filter_hemisphere(self.dataset.nodes_df, hemisphere)
        for col_name, label_to_val in self._extra_cols.items():
            nodes_df[col_name] = nodes_df["label"].map(label_to_val)
        matrix = self._get_matrix(nodes_df)
        highlight_edges = a["highlight_edges"]
        if highlight_edges is not None:
            highlight_edges = self._align_to_nodes(highlight_edges, nodes_df)

        show_surface = a["show_surface"]
        if a["layout"] is not None:
            positions = self._compute_layout(matrix, a["layout"], a["edge_threshold"], a["layout_seed"])
            show_surface = False
        else:
            positions = nodes_df[["x", "y", "z"]].values.astype(float)

        surfaces: list = []
        if show_surface:
            paths = {
                h: path for h, path in (("L", a["surface_L"]), ("R", a["surface_R"]))
                if path and (hemisphere == "both" or hemisphere.upper() == h)
            }
            surfaces = load_surface(
                surface_L=paths.get("L"),
                surface_R=paths.get("R"),
                color=a["surface_color"],
                alpha=a["surface_alpha"],
            )
            for h, mesh in zip(paths, surfaces):
                mesh._hemisphere = h

        node_spheres = build_nodes(
            nodes_df           = nodes_df,
            node_size          = a["node_size"],
            node_size_range    = a["node_size_range"],
            node_color         = a["node_color"],
            node_cmap          = a["node_cmap"],
            node_colorvminvmax = a["node_colorvminvmax"],
            node_alpha         = a["node_alpha"],
            node_res           = a["node_res"],
            palette            = a["node_palette"],
            positions          = positions,
        )
        node_colors = _resolve_colors(
            a["node_color"], a["node_cmap"], a["node_colorvminvmax"], nodes_df, len(nodes_df), a["node_palette"]
        )

        edge_lines = build_edges(
            matrix           = matrix,
            positions        = positions,
            threshold        = a["edge_threshold"],
            threshold_dir    = a["edge_threshold_dir"],
            edge_width       = a["edge_width"],
            edge_width_range = a["edge_width_range"],
            edge_color       = a["edge_color"],
            edge_cmap        = a["edge_cmap"],
            edge_colorvminvmax = a["edge_colorvminvmax"],
            edge_alpha       = a["edge_alpha"],
            node_colors      = node_colors,
            use_tube         = a["use_tube"],
            highlight_edges  = highlight_edges,
            highlight_level  = a["highlight_level"],
            edge_sign_colors = a["edge_sign_colors"],
        )

        extras = make_axis_arrows(axes=a["arrowaxis"]) if a["arrowaxis"] is not None else []
        return _Scene(nodes_df, surfaces, node_spheres, edge_lines, extras, node_colors)

    def _get_matrix(self, nodes_df: pd.DataFrame) -> np.ndarray:
        if self.subject_id == "mean":
            df = self.dataset.mean_matrix()
        elif self.subject_id in self.dataset.matrices:
            df = self.dataset.matrices[self.subject_id]
        else:
            raise KeyError(
                f"subject_id='{self.subject_id}' not found. "
                f"Available: {self.dataset.subject_ids}"
            )

        labels = nodes_df["label"].tolist()
        return df.loc[labels, labels].values.astype(float)

    def _align_to_nodes(self, arr: np.ndarray, nodes_df: pd.DataFrame) -> np.ndarray:
        # mean_matrix() keeps the first subject's label order
        key    = next(iter(self.dataset.matrices)) if self.subject_id == "mean" else self.subject_id
        source = list(self.dataset.matrices[key].columns)
        arr    = np.asarray(arr)
        if arr.shape != (len(source), len(source)):
            raise ValueError(
                f"highlight_edges has shape {arr.shape}; expected {(len(source), len(source))}, "
                "ordered like the dataset matrix labels."
            )
        pos = {lbl: k for k, lbl in enumerate(source)}
        idx = [pos[lbl] for lbl in nodes_df["label"]]
        return arr[np.ix_(idx, idx)]

    def _compute_layout(
        self,
        matrix: np.ndarray,
        layout: str,
        threshold: float,
        seed: int,
    ) -> np.ndarray:
        import networkx as nx

        adj = np.abs(matrix)
        adj[adj <= threshold] = 0
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_array(adj)
        # kamada_kawai reads the edge attribute as a distance, so stronger edges get shorter lengths
        nx.set_edge_attributes(G, {(u, v): 1.0 / w for u, v, w in G.edges(data="weight")}, "length")

        fn = {
            "spring":       lambda: nx.spring_layout(G, dim=3, seed=seed, weight="weight"),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(
                G, dim=3, weight="length", pos=nx.random_layout(G, dim=3, seed=seed)
            ),
            "spectral":     lambda: nx.spectral_layout(G, dim=3, weight="weight"),
            "forceatlas2":  lambda: nx.forceatlas2_layout(G, dim=3, weight="weight", seed=seed),
        }.get(layout)

        if fn is None:
            raise ValueError(
                f"layout='{layout}' not recognised. "
                "Choose from: 'spring', 'kamada_kawai', 'spectral', 'forceatlas2'."
            )

        pos = fn()
        n   = matrix.shape[0]
        coords = np.array([pos.get(i, np.zeros(3)) for i in range(n)])
        # scale to a range similar to MNI (roughly ±100 mm) for comfortable viewing
        span = np.ptp(coords, axis=0).max()
        if span > 0:
            coords = coords / span * 200
        return coords

    def _filter_hemisphere(
        self,
        nodes_df: pd.DataFrame,
        show_hemisphere: str,
    ) -> pd.DataFrame:
        if show_hemisphere == "both" or "hemisphere" not in nodes_df.columns:
            return nodes_df.copy()

        mask     = nodes_df["hemisphere"].str.upper() == show_hemisphere.upper()
        filtered = nodes_df[mask].reset_index(drop=True)

        if filtered.empty:
            raise DataValidationError(
                f"No nodes found for hemisphere='{show_hemisphere}'. "
                f"Check the 'hemisphere' column in your nodes file."
            )
        return filtered
