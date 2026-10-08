"""
Interactive window scene: nodes merged into one mesh and edges into a few, so rotation stays smooth.
"""

from __future__ import annotations

import numpy as np

_N_WIDTH_LEVELS = 8
_DIM_RGBA = (0.55, 0.55, 0.55, 0.06)


class WindowScene:
    """Merged node and edge meshes for the interactive window, with click highlighting and an edge threshold."""

    def __init__(self, nodes: list, edges: list, edge_key: np.ndarray | None = None):
        self.node_mesh = _merge_nodes(nodes) if nodes else None
        self.edge_meshes = _merge_edges(edges) if edges else []
        self._endpoints = np.array([e._endpoints for e in edges], dtype=int).reshape(-1, 2)
        self._orig_rgba = np.array([(*e._orig_color, e._orig_alpha) for e in edges], dtype=float).reshape(-1, 4)
        self._key = np.full(len(edges), np.inf) if edge_key is None else np.asarray(edge_key, dtype=float)
        self._threshold = -np.inf
        self._selected: int | None = None

    @classmethod
    def from_arrays(cls, nodes: list, paths, endpoints, rgba, widths, edge_key) -> WindowScene:
        """Edges given as arrays instead of per-edge actors, so tens of thousands of them stay cheap to build."""
        scene = cls(nodes, [])
        scene._endpoints = np.asarray(endpoints, dtype=int).reshape(-1, 2)
        scene._orig_rgba = np.asarray(rgba, dtype=float).reshape(-1, 4)
        scene._key = np.asarray(edge_key, dtype=float)
        if len(scene._key):
            scene.edge_meshes = _merge_lines(paths, np.asarray(widths, dtype=float), scene._orig_rgba, scene._endpoints)
        return scene

    def actors(self, surfaces: list, extras: list) -> list:
        nodes = [self.node_mesh] if self.node_mesh is not None else []
        return surfaces + nodes + self.edge_meshes + extras

    def node_at(self, actor, point) -> int | None:
        if actor is None or actor is not self.node_mesh or point is None:
            return None
        mesh = self.node_mesh
        # the picked point lies on the surface of the clicked sphere
        gap = np.abs(np.linalg.norm(mesh._node_centers - np.asarray(point, dtype=float), axis=1) - mesh._node_radii)
        return int(np.argmin(gap))

    def click(self, node_idx: int | None) -> None:
        self._selected = None if node_idx is None or node_idx == self._selected else node_idx
        self._refresh()

    def set_threshold(self, threshold: float) -> None:
        """Hide edges whose key is at or below threshold, as the static plot does."""
        self._threshold = threshold
        self._refresh()

    def _refresh(self) -> None:
        rgba = self._orig_rgba.copy()
        if self._selected is not None:
            touching = (self._endpoints == self._selected).any(axis=1)
            rgba[~touching] = _DIM_RGBA
            rgba[touching, 3] = np.minimum(rgba[touching, 3] * 1.4, 1.0)
        rgba[self._key <= self._threshold, 3] = 0.0
        for mesh in self.edge_meshes:
            mesh.cellcolors = _to_uint8(rgba[mesh._cell_edge])


def _to_uint8(rgba: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def _merge_nodes(spheres: list):
    from vedo import merge

    mesh = merge(spheres)
    counts = [s.ncells for s in spheres]
    rgba = np.array([(*s.color(), s.alpha()) for s in spheres])
    # set the index array before the colours, otherwise it becomes the active scalars
    mesh.celldata["node_idx"] = np.repeat([s._node_idx for s in spheres], counts)
    mesh.cellcolors = _to_uint8(np.repeat(rgba, counts, axis=0))
    mesh.properties.DeepCopy(spheres[0].properties)
    mesh.properties.SetOpacity(1.0)
    mesh._node_centers = np.array([s.center_of_mass() for s in spheres])
    mesh._node_radii = np.array([s.average_size() for s in spheres])
    return mesh


def _merge_edges(edges: list) -> list:
    from vedo import Tube, merge

    rgba = np.array([(*e._orig_color, e._orig_alpha) for e in edges])
    endpoints = np.array([e._endpoints for e in edges], dtype=int)
    if not isinstance(edges[0], Tube):
        widths = np.array([e.properties.GetLineWidth() for e in edges])
        return _merge_lines([e.vertices for e in edges], widths, rgba, endpoints)
    mesh = merge(edges)
    return [_edge_mesh(mesh, np.repeat(np.arange(len(edges)), [e.ncells for e in edges]), rgba, endpoints)]


def _merge_lines(paths, widths: np.ndarray, rgba: np.ndarray, endpoints: np.ndarray) -> list:
    """Polyline meshes grouped into a few line widths, since one mesh has a single width."""
    lo, hi = widths.min(), widths.max()
    levels = np.zeros(len(widths), dtype=int)
    if hi > lo:
        levels = np.minimum(((widths - lo) / (hi - lo) * _N_WIDTH_LEVELS).astype(int), _N_WIDTH_LEVELS - 1)
    out = []
    for level in np.unique(levels):
        members = np.flatnonzero(levels == level)
        mesh = _polylines([paths[k] for k in members])
        mesh.properties.LightingOff()
        mesh.lw(lo + (level + 0.5) * (hi - lo) / _N_WIDTH_LEVELS if hi > lo else lo)
        out.append(_edge_mesh(mesh, members, rgba, endpoints))
    return out


def _edge_mesh(mesh, cell_edge: np.ndarray, rgba: np.ndarray, endpoints: np.ndarray):
    mesh.properties.SetOpacity(1.0)
    mesh.cellcolors = _to_uint8(rgba[cell_edge])
    mesh._cell_edge = cell_edge
    mesh._cell_endpoints = endpoints[cell_edge]
    return mesh


def _polylines(paths: list):
    """One mesh holding each path as a single polyline cell."""
    from vedo import Mesh
    from vedo import vtkclasses as vtki
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = np.vstack(paths).astype(float)
    offsets = np.concatenate(([0], np.cumsum([len(p) for p in paths]))).astype(np.int64)
    cells = vtki.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True))
    vpts = vtki.vtkPoints()
    vpts.SetData(numpy_to_vtk(points, deep=True))
    poly = vtki.vtkPolyData()
    poly.SetPoints(vpts)
    poly.SetLines(cells)
    return Mesh(poly)


class HoverCard:
    """Card with a node's label and values, drawn next to the node under the mouse."""

    def __init__(self, nodes_df, node_colors: list, value_columns: list[str], centres: np.ndarray):
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkImageMapper, vtkPolyDataMapper2D
        from vedo import vtkclasses as vtki

        self._nodes_df = nodes_df.reset_index(drop=True)
        self._node_colors = node_colors
        self._columns = value_columns
        self._centres = np.asarray(centres, dtype=float)
        self._images: dict = {}
        self._shown: tuple | None = None

        self._mapper = vtkImageMapper()
        self._mapper.SetColorWindow(255)
        self._mapper.SetColorLevel(127.5)
        self.card = vtkActor2D()
        self.card.SetMapper(self._mapper)
        self.card._hover_lines = None

        self._leader_points = vtki.vtkPoints()
        self._leader_points.SetNumberOfPoints(2)
        line = vtki.vtkCellArray()
        line.InsertNextCell(2)
        line.InsertCellPoint(0)
        line.InsertCellPoint(1)
        poly = vtki.vtkPolyData()
        poly.SetPoints(self._leader_points)
        poly.SetLines(line)
        leader_mapper = vtkPolyDataMapper2D()
        leader_mapper.SetInputData(poly)
        self.leader = vtkActor2D()
        self.leader.SetMapper(leader_mapper)
        self.leader.GetProperty().SetColor(0.3, 0.3, 0.3)
        self.leader.GetProperty().SetLineWidth(1.5)
        self.hide()

    @property
    def actors(self) -> list:
        return [self.leader, self.card]

    def lines(self, node_idx: int) -> list:
        row = self._nodes_df.loc[node_idx]
        out: list = [str(row["label"])]
        if "network" in row.index:
            out.append(("Network", str(row["network"])))
        if "hemisphere" in row.index:
            out.append(("Hemisphere", {"L": "Left", "R": "Right"}.get(row["hemisphere"], str(row["hemisphere"]))))
        out += [(col, f"{float(row[col]):.4g}") for col in self._columns]
        return out

    def hide(self) -> bool:
        changed = self._shown is not None
        self._shown = None
        self.card.VisibilityOff()
        self.leader.VisibilityOff()
        self.card._hover_lines = None
        return changed

    def update(self, node_idx: int | None, renderer) -> bool:
        """Show the card for node_idx (None hides it); True when the window needs a render."""
        if node_idx is None:
            return self.hide()
        from vedo import vtkclasses as vtki

        coord = vtki.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue(*self._centres[node_idx])
        x, y = coord.GetComputedDoubleDisplayValue(renderer)
        state = (node_idx, round(x), round(y))
        if state == self._shown:
            return False
        self._shown = state

        image = self._image(node_idx)
        w, h, _ = image.GetDimensions()
        width, height = renderer.GetSize()
        # place the card up and right of the node, flipped when it would leave the window
        left = x + 18 if x + 18 + w <= width else x - 18 - w
        bottom = y + 14 if y + 14 + h <= height else y - 14 - h
        self.card.SetPosition(int(left), int(bottom))
        corner_x = left if left > x else left + w
        corner_y = bottom + 6 if bottom > y else bottom + h - 6
        self._leader_points.SetPoint(0, x, y, 0)
        self._leader_points.SetPoint(1, corner_x, corner_y, 0)
        self._leader_points.Modified()
        self._mapper.SetInputData(image)
        self.card._hover_lines = self.lines(node_idx)
        self.card.VisibilityOn()
        self.leader.VisibilityOn()
        return True

    def _image(self, node_idx: int):
        if node_idx not in self._images:
            from vedo import vtkclasses as vtki
            from vtkmodules.util.numpy_support import numpy_to_vtk

            title, *fields = self.lines(node_idx)
            rgba = _card_image(title, tuple(self._node_colors[node_idx]), fields)
            h, w = rgba.shape[:2]
            data = vtki.vtkImageData()
            data.SetDimensions(w, h, 1)
            scalars = numpy_to_vtk(np.flipud(rgba).reshape(-1, 4), deep=True)
            scalars.SetNumberOfComponents(4)
            data.GetPointData().SetScalars(scalars)
            self._images[node_idx] = data
        return self._images[node_idx]


def _card_image(title: str, dot_rgb: tuple, fields: list) -> np.ndarray:
    """Rounded white card with a coloured dot, a bold title and name/value rows, as RGBA pixels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, FancyBboxPatch

    longest_row = max((len(name) + len(value) for name, value in fields), default=0)
    w_px = max(230, 9 * len(title) + 60, 7 * longest_row + 60)
    h_px = 44 + 22 * len(fields)
    fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    fig.patch.set_alpha(0)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_px)
    ax.set_ylim(0, h_px)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((4, 4), w_px - 8, h_px - 8, boxstyle="round,pad=0,rounding_size=7",
                                fc="white", ec=(0.82, 0.82, 0.82), lw=0.5, alpha=0.97))
    top = h_px - 22
    ax.add_patch(Circle((18, top + 1), 5.5, color=dot_rgb))
    ax.text(30, top, title, fontsize=7.15, fontweight="bold", color="#222", va="center")
    for k, (name, value) in enumerate(fields):
        y = top - 26 - 22 * k
        ax.text(16, y, name, fontsize=6.5, color="#777", va="center")
        ax.text(w_px - 16, y, value, fontsize=6.5, color="#222", va="center", ha="right")
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


_SLIDER_TITLES = {"absabove": "Edge threshold |w|", "above": "Edge threshold w", "below": "Edge threshold -w"}


def add_threshold_slider(plt, on_change, lowest: float, highest: float, value: float, threshold_dir: str):
    """Thin grey slider centred at the bottom of the window; on_change(widget, event) runs while it moves."""
    widget = plt.add_slider(
        on_change, lowest, highest, value=min(max(value, lowest), highest), pos=[(0.30, 0.07), (0.70, 0.07)],
        title=_SLIDER_TITLES[threshold_dir], show_value=True,
    )
    rep = widget.representation
    rep.SetLabelFormat("%.2f")
    rep.SetSliderLength(0.012)
    rep.SetSliderWidth(0.022)
    rep.SetTubeWidth(0.004)
    rep.SetEndCapLength(0.0)
    rep.SetEndCapWidth(0.0)
    ink = (0.2, 0.2, 0.2) if np.mean(plt.renderer.GetBackground()) > 0.5 else (0.85, 0.85, 0.85)
    rep.GetTubeProperty().SetColor(0.6, 0.6, 0.6)
    rep.GetSliderProperty().SetColor(*ink)
    rep.GetSelectedProperty().SetColor(0.1, 0.4, 0.8)
    for text in (rep.GetTitleProperty(), rep.GetLabelProperty()):
        text.SetColor(*ink)
        text.SetFontFamilyToArial()
        text.ShadowOff()
        text.BoldOff()
        text.ItalicOff()
    rep.SetTitleHeight(0.018)
    rep.SetLabelHeight(0.018)
    return widget
