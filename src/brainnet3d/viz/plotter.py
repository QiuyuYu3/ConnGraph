"""
BrainNetPlotter: the main user-facing visualisation class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from brainnet3d.core.dataset import ConnectivityDataset

if TYPE_CHECKING:
    from brainnet3d.graph_theory.runner import GraphMetricsResult
from brainnet3d.viz.surface import load_surface
from brainnet3d.viz.nodes   import build_nodes
from brainnet3d.viz.edges   import build_edges


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

        from brainnet3d.graph_theory.metrics import METRIC_NAMES

        row    = result.node_df.loc[sid]
        chosen = metrics if metrics is not None else METRIC_NAMES

        # ROI labels may contain "_", so match known metric prefixes instead of splitting
        self._extra_cols = {}
        for metric in chosen:
            prefix = metric + "_"
            label_to_val = {
                col[len(prefix):]: float(row[col])
                for col in row.index
                if col.startswith(prefix)
            }
            if label_to_val:
                self._extra_cols[metric] = label_to_val

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
        node_cmap:           str                   = "Set3",
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
        node_cmap  : colormap for numeric node_color.
        node_alpha : node transparency.
        node_res   : sphere tessellation (higher = smoother).
        node_palette : {category_label: colour} override for categorical colouring.

        edge_threshold : edges with |weight| ≤ threshold are hidden.
        edge_width : float → uniform. "weight" → scaled to edge_width_range.
        edge_width_range : (min_w, max_w) in pixels.
        edge_color : "weight" → colormap. "node" → inherits node colour.
                     RGB tuple / colour name → uniform.
        edge_cmap  : colormap for edge_color="weight".
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
                 "kamada_kawai" → NetworkX Kamada-Kawai 3-D layout.
                 "spectral" → NetworkX spectral 3-D layout.
                 Brain surface is automatically hidden when a layout is used.
        layout_seed : random seed for "spring" layout reproducibility.
        """
        from vedo import Plotter

        nodes_df  = self._filter_hemisphere(self.dataset.nodes_df, show_hemisphere)
        if self._extra_cols:
            for col_name, label_to_val in self._extra_cols.items():
                nodes_df[col_name] = nodes_df["label"].map(label_to_val)
        matrix    = self._get_matrix(nodes_df)
        if highlight_edges is not None:
            highlight_edges = self._align_to_nodes(highlight_edges, nodes_df)

        if layout is not None:
            positions = self._compute_layout(matrix, layout, edge_threshold, layout_seed)
            show_surface = False
        else:
            positions = nodes_df[["x", "y", "z"]].values.astype(float)

        actors: list = []

        # Surface
        if show_surface and (surface_L or surface_R):
            meshes = load_surface(
                surface_L=surface_L,
                surface_R=surface_R,
                color=surface_color,
                alpha=surface_alpha,
            )
            actors.extend(meshes)

        # Nodes
        node_spheres = build_nodes(
            nodes_df           = nodes_df,
            node_size          = node_size,
            node_size_range    = node_size_range,
            node_color         = node_color,
            node_cmap          = node_cmap,
            node_colorvminvmax = node_colorvminvmax,
            node_alpha         = node_alpha,
            node_res           = node_res,
            palette            = node_palette,
            positions          = positions,
        )
        actors.extend(node_spheres)

        # Edges
        from brainnet3d.viz.nodes import _resolve_colors
        node_colors_list = _resolve_colors(
            node_color, node_cmap, node_colorvminvmax, nodes_df, len(nodes_df), node_palette
        )

        edge_lines = build_edges(
            matrix           = matrix,
            positions        = positions,
            threshold        = edge_threshold,
            threshold_dir    = edge_threshold_dir,
            edge_width       = edge_width,
            edge_width_range = edge_width_range,
            edge_color       = edge_color,
            edge_cmap        = edge_cmap,
            edge_alpha       = edge_alpha,
            node_colors      = node_colors_list,
            use_tube         = use_tube,
            highlight_edges  = highlight_edges,
            highlight_level  = highlight_level,
        )
        actors.extend(edge_lines)

        if arrowaxis is not None:
            from brainnet3d.viz.views import make_axis_arrows
            actors.extend(make_axis_arrows(axes=arrowaxis))

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

        fn = {
            "spring":       lambda: nx.spring_layout(G, dim=3, seed=seed, weight="weight"),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(G, dim=3, weight="weight"),
            "spectral":     lambda: nx.spectral_layout(G, dim=3, weight="weight"),
        }.get(layout)

        if fn is None:
            raise ValueError(
                f"layout='{layout}' not recognised. "
                "Choose from: 'spring', 'kamada_kawai', 'spectral'."
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
            raise ValueError(
                f"No nodes found for hemisphere='{show_hemisphere}'. "
                f"Check the 'hemisphere' column in your nodes file."
            )
        return filtered
