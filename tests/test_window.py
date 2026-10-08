from types import SimpleNamespace

import numpy as np
import pytest

import brainnet3d as bnv

_ALL_EDGES = dict(edge_threshold=-1.0, edge_threshold_dir="above", edge_alpha=0.5, node_size=3.0)
_EDGE_STYLES = {
    "straight": {},
    "bundled": {"edge_bundling": True},
    "tube": {"use_tube": True},
}
_TOL = 1.0 / 255


def _actors_of(plt) -> list:
    actors = plt.renderer.GetActors()
    actors.InitTraversal()
    out = []
    for _ in range(actors.GetNumberOfItems()):
        a = actors.GetNextActor()
        out.append(a.retrieve_object() if hasattr(a, "retrieve_object") else a)
    return out


def _open_window(plotter, monkeypatch, clicks=(), **kwargs) -> list:
    """Run plot(interactive=True) off screen; return the actors in the window after each click."""
    import vedo

    snapshots = []

    class _Headless(vedo.Plotter):
        def __init__(self, *a, **kw):
            kw["offscreen"] = True
            super().__init__(*a, **kw)
            self._test_callback = None

        def add_callback(self, event_name, func, *a, **kw):
            assert event_name == "LeftButtonPress"
            self._test_callback = func

        def interactive(self):
            actors = _actors_of(self)
            snapshots.append(_states(actors))
            for target in clicks:
                self._test_callback(_click_event(actors, target))
                snapshots.append(_states(_actors_of(self)))
            return self

    monkeypatch.setattr(vedo, "Plotter", _Headless)
    plotter.plot(interactive=True, **kwargs)
    return snapshots


def _node_mesh_or_spheres(actors):
    merged = [a for a in actors if hasattr(a, "_node_centers")]
    return merged[0] if merged else {a._node_idx: a for a in actors if hasattr(a, "_node_idx")}


def _click_event(actors, target):
    if target is None:
        return SimpleNamespace(actor=None, object=None, picked3d=None)
    if target == "surface":
        surf = next(a for a in actors if hasattr(a, "_hemisphere"))
        return SimpleNamespace(actor=surf, object=surf, picked3d=np.asarray(surf.vertices[0]))
    nodes = _node_mesh_or_spheres(actors)
    if isinstance(nodes, dict):
        sphere = nodes[target]
        return SimpleNamespace(actor=sphere, object=sphere, picked3d=np.asarray(sphere.center_of_mass()))
    point = nodes._node_centers[target] + np.array([nodes._node_radii[target], 0.0, 0.0])
    return SimpleNamespace(actor=nodes, object=nodes, picked3d=point)


def _states(actors) -> dict:
    """{"edges": {(i, j): rgba}, "nodes": {idx: rgba}, "n_actors": int} from old or merged actors."""
    edges, nodes = {}, {}
    for a in actors:
        if hasattr(a, "_endpoints"):
            rgb = a.color()
            if a.mapper.GetScalarVisibility() and "TubeColors" in a.pointdata.keys():
                rgb = np.asarray(a.pointdata["TubeColors"][0][:3], dtype=float) / 255
            edges[tuple(a._endpoints)] = (*rgb, a.alpha())
        elif hasattr(a, "_cell_endpoints"):
            rgba = np.asarray(a.cellcolors, dtype=float) / 255
            for (i, j), c in zip(a._cell_endpoints, rgba):
                key = (int(i), int(j))
                assert key not in edges or np.allclose(edges[key], c), "cells of one edge differ in colour"
                edges[key] = tuple(c)
        elif hasattr(a, "_node_idx"):
            nodes[a._node_idx] = (*a.color(), a.alpha())
        elif hasattr(a, "_node_centers"):
            rgba = np.asarray(a.cellcolors, dtype=float) / 255
            for idx, c in zip(a.celldata["node_idx"], rgba):
                nodes[int(idx)] = tuple(c)
    return {"edges": edges, "nodes": nodes, "n_actors": len(actors)}


def _static_actors(plotter, monkeypatch, **kwargs) -> list:
    import brainnet3d.viz.views as views

    captured = {}
    real = views._finish_render
    monkeypatch.setattr(views, "_finish_render", lambda vp, actors, *a, **kw: captured.update(actors=actors))
    plotter.plot(**kwargs)
    monkeypatch.setattr(views, "_finish_render", real)
    return captured["actors"]


def _expected(original: dict, selected: int | None) -> dict:
    out = {}
    for (i, j), (rgb, alpha) in original.items():
        if selected is None:
            out[(i, j)] = (*rgb, alpha)
        elif selected in (i, j):
            out[(i, j)] = (*rgb, min(alpha * 1.4, 1.0))
        else:
            out[(i, j)] = (0.55, 0.55, 0.55, 0.06)
    return out


def _assert_edges(actual: dict, expected: dict):
    assert actual.keys() == expected.keys()
    for key, rgba in expected.items():
        np.testing.assert_allclose(actual[key], rgba, atol=_TOL, err_msg=str(key))


@pytest.fixture
def scene_kwargs(surfaces):
    return dict(surface_L=surfaces[0], surface_R=surfaces[1], **_ALL_EDGES)


@pytest.mark.parametrize("style", _EDGE_STYLES)
def test_window_has_few_actors(dataset, monkeypatch, scene_kwargs, style):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    static = _static_actors(plotter, monkeypatch, **scene_kwargs, **_EDGE_STYLES[style])
    assert sum(hasattr(a, "_endpoints") for a in static) > 100
    window = _open_window(plotter, monkeypatch, **scene_kwargs, **_EDGE_STYLES[style])[0]
    assert window["n_actors"] <= 2 + 1 + 8


@pytest.mark.parametrize("style", _EDGE_STYLES)
def test_window_keeps_node_and_edge_colours(dataset, monkeypatch, scene_kwargs, style):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    kwargs = dict(**scene_kwargs, **_EDGE_STYLES[style], highlight_nodes={"network": "Visual"}, highlight_level=0.6)
    static = _static_actors(plotter, monkeypatch, **kwargs)
    original = {tuple(a._endpoints): (a._orig_color, a._orig_alpha) for a in static if hasattr(a, "_endpoints")}
    nodes = {a._node_idx: (*a.color(), a.alpha()) for a in static if hasattr(a, "_node_idx")}
    window = _open_window(plotter, monkeypatch, **kwargs)[0]
    _assert_edges(window["edges"], _expected(original, None))
    assert window["nodes"].keys() == nodes.keys()
    for idx, rgba in nodes.items():
        np.testing.assert_allclose(window["nodes"][idx], rgba, atol=_TOL)


@pytest.mark.parametrize("style", _EDGE_STYLES)
@pytest.mark.parametrize("clicks, selected", [
    ((5,), [5]),
    ((5, 5), [5, None]),
    ((5, None), [5, None]),
    ((5, "surface"), [5, None]),
    ((5, 17), [5, 17]),
])
def test_click_highlights_edges_of_node(dataset, monkeypatch, scene_kwargs, style, clicks, selected):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    kwargs = dict(**scene_kwargs, **_EDGE_STYLES[style], highlight_on_click=True)
    static = _static_actors(plotter, monkeypatch, **kwargs)
    original = {tuple(a._endpoints): (a._orig_color, a._orig_alpha) for a in static if hasattr(a, "_endpoints")}
    snapshots = _open_window(plotter, monkeypatch, clicks=clicks, **kwargs)
    _assert_edges(snapshots[0]["edges"], _expected(original, None))
    for snap, sel in zip(snapshots[1:], selected):
        _assert_edges(snap["edges"], _expected(original, sel))


def test_window_line_widths_stay_close(dataset, monkeypatch, scene_kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    static = _static_actors(plotter, monkeypatch, **scene_kwargs)
    widths = {tuple(a._endpoints): a.properties.GetLineWidth() for a in static if hasattr(a, "_endpoints")}
    span = max(widths.values()) - min(widths.values())
    assert span > 1.0

    import vedo

    seen = {}

    class _Headless(vedo.Plotter):
        def __init__(self, *a, **kw):
            kw["offscreen"] = True
            super().__init__(*a, **kw)

        def interactive(self):
            for a in _actors_of(self):
                for key in (getattr(a, "_cell_endpoints", None) if hasattr(a, "_cell_endpoints") else []):
                    seen[(int(key[0]), int(key[1]))] = a.properties.GetLineWidth()
                if hasattr(a, "_endpoints"):
                    seen[tuple(a._endpoints)] = a.properties.GetLineWidth()
            return self

    monkeypatch.setattr(vedo, "Plotter", _Headless)
    plotter.plot(interactive=True, **scene_kwargs)
    assert seen.keys() == widths.keys()
    for key, w in widths.items():
        assert abs(seen[key] - w) <= span / 16 + 1e-9


def test_window_screenshot_matches_static_render(dataset, monkeypatch, scene_kwargs, tmp_path):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    plotter.plot(**scene_kwargs, screenshot=str(tmp_path / "static.png"))
    _open_window(plotter, monkeypatch, **scene_kwargs, screenshot=str(tmp_path / "window.png"))
    assert (tmp_path / "static.png").read_bytes() == (tmp_path / "window.png").read_bytes()
