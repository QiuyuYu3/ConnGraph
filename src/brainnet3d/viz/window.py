"""
Interactive window scene: nodes merged into one mesh and edges into a few, so rotation stays smooth.
"""

from __future__ import annotations

import numpy as np

_N_WIDTH_LEVELS = 8
_DIM_RGBA = (0.55, 0.55, 0.55, 0.06)


class WindowScene:
    """Merged node and edge meshes built from the per-node and per-edge actors, with click highlighting."""

    def __init__(self, nodes: list, edges: list):
        self.node_mesh = _merge_nodes(nodes) if nodes else None
        self.edge_meshes = _merge_edges(edges) if edges else []
        self._endpoints = np.array([e._endpoints for e in edges], dtype=int).reshape(-1, 2)
        self._orig_rgba = np.array([(*e._orig_color, e._orig_alpha) for e in edges], dtype=float).reshape(-1, 4)
        self._selected: int | None = None

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
        if node_idx is None or node_idx == self._selected:
            self._selected = None
            rgba = self._orig_rgba
        else:
            self._selected = node_idx
            touching = (self._endpoints == node_idx).any(axis=1)
            rgba = np.tile(_DIM_RGBA, (len(self._orig_rgba), 1))
            rgba[touching, :3] = self._orig_rgba[touching, :3]
            rgba[touching, 3] = np.minimum(self._orig_rgba[touching, 3] * 1.4, 1.0)
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
    if isinstance(edges[0], Tube):
        mesh = merge(edges)
        cell_edge = np.repeat(np.arange(len(edges)), [e.ncells for e in edges])
        meshes = [(mesh, cell_edge)]
    else:
        widths = np.array([e.properties.GetLineWidth() for e in edges])
        lo, hi = widths.min(), widths.max()
        levels = np.zeros(len(edges), dtype=int)
        if hi > lo:
            levels = np.minimum(((widths - lo) / (hi - lo) * _N_WIDTH_LEVELS).astype(int), _N_WIDTH_LEVELS - 1)
        meshes = []
        for level in np.unique(levels):
            members = np.flatnonzero(levels == level)
            mesh = _polylines([edges[k].vertices for k in members])
            mesh.properties.DeepCopy(edges[members[0]].properties)
            mesh.lw(lo + (level + 0.5) * (hi - lo) / _N_WIDTH_LEVELS if hi > lo else lo)
            meshes.append((mesh, members))

    out = []
    for mesh, cell_edge in meshes:
        mesh.properties.SetOpacity(1.0)
        mesh.cellcolors = _to_uint8(rgba[cell_edge])
        mesh._cell_edge = cell_edge
        mesh._cell_endpoints = np.array([edges[k]._endpoints for k in cell_edge], dtype=int)
        out.append(mesh)
    return out


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
