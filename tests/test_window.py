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


def _open_window(plotter, monkeypatch, clicks=(), moves=(), slides=(), **kwargs) -> list:
    """Run plot(interactive=True) off screen; return the window state after each click, mouse move and slider value."""
    import vedo

    snapshots = []

    class _Headless(vedo.Plotter):
        def __init__(self, *a, **kw):
            kw["offscreen"] = True
            super().__init__(*a, **kw)
            self._test_callbacks = {}
            self._test_slider = None

        def add_callback(self, event_name, func, *a, **kw):
            self._test_callbacks[event_name] = func

        def add_slider(self, func, *a, **kw):
            widget = super().add_slider(func, *a, **kw)
            self._test_slider = (func, widget)
            return widget

        def interactive(self):
            actors = _actors_of(self)
            slider = None if self._test_slider is None else self._test_slider[1].range
            snapshots.append({
                **_states(actors), "hover": _hover_lines(self), "events": set(self._test_callbacks), "slider": slider,
                "legend": _legend_of(self),
            })
            for event_name, targets in (("LeftButtonPress", clicks), ("MouseMove", moves)):
                for target in targets:
                    self._test_callbacks[event_name](_click_event(actors, target, self))
                    snapshots.append({**_states(_actors_of(self)), "hover": _hover_lines(self)})
            for value in slides:
                func, widget = self._test_slider
                widget.value = value
                func(widget, "InteractionEvent")
                snapshots.append(_states(_actors_of(self)))
            return self

    monkeypatch.setattr(vedo, "Plotter", _Headless)
    plotter.plot(interactive=True, **kwargs)
    return snapshots


def _hover_lines(plt) -> list | None:
    props = plt.renderer.GetViewProps()
    props.InitTraversal()
    for _ in range(props.GetNumberOfItems()):
        p = props.GetNextProp()
        if hasattr(p, "_hover_lines") and p.GetVisibility():
            return p._hover_lines
    return None


def _node_mesh_or_spheres(actors):
    merged = [a for a in actors if hasattr(a, "_node_centers")]
    return merged[0] if merged else {a._node_idx: a for a in actors if hasattr(a, "_node_idx")}


def _legend_of(plt):
    props = plt.renderer.GetViewProps()
    props.InitTraversal()
    for _ in range(props.GetNumberOfItems()):
        p = props.GetNextProp()
        if hasattr(p, "_legend"):
            return p._legend
    return None


def _click_event(actors, target, plt=None):
    if isinstance(target, tuple) and target[0] == "legend":
        plt.render()
        return SimpleNamespace(actor=None, object=None, picked3d=None, picked2d=_legend_of(plt).row_center(target[1]))
    if target is None:
        return SimpleNamespace(actor=None, object=None, picked3d=None, picked2d=(-1, -1))
    if target == "surface":
        surf = next(a for a in actors if hasattr(a, "_hemisphere"))
        return SimpleNamespace(actor=surf, object=surf, picked3d=np.asarray(surf.vertices[0]), picked2d=(-1, -1))
    nodes = _node_mesh_or_spheres(actors)
    if isinstance(nodes, dict):
        sphere = nodes[target]
        return SimpleNamespace(actor=sphere, object=sphere, picked3d=np.asarray(sphere.center_of_mass()), picked2d=(-1, -1))
    point = nodes._node_centers[target] + np.array([nodes._node_radii[target], 0.0, 0.0])
    return SimpleNamespace(actor=nodes, object=nodes, picked3d=point, picked2d=(-1, -1))


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
                if c[3] == 0:
                    continue
                key = (int(i), int(j))
                assert key not in edges or np.allclose(edges[key], c), "cells of one edge differ in colour"
                edges[key] = tuple(c)
        elif hasattr(a, "_node_idx"):
            nodes[a._node_idx] = (*a.color(), a.alpha())
        elif hasattr(a, "_node_centers"):
            rgba = np.asarray(a.cellcolors, dtype=float) / 255
            spread = np.zeros(len(a._node_centers))
            np.maximum.at(spread, a._point_node, np.linalg.norm(a.vertices - a._node_centers[a._point_node], axis=1))
            for idx, c in zip(a.celldata["node_idx"], rgba):
                if spread[int(idx)] > 1e-3:
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


def test_hover_shows_card_for_node(dataset, monkeypatch, scene_kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    nodes = plotter.dataset.nodes_df.reset_index(drop=True)
    snaps = _open_window(plotter, monkeypatch, moves=(5, 5, None, 17, "surface"), **scene_kwargs)
    assert "MouseMove" in snaps[0]["events"]
    assert snaps[0]["hover"] is None

    def card(k):
        row = nodes.loc[k]
        return [row["label"], ("Network", row["network"]), ("Hemisphere", {"L": "Left", "R": "Right"}[row["hemisphere"]])]

    assert [s["hover"] for s in snaps[1:]] == [card(5), card(5), None, card(17), None]
    for s in snaps[1:]:
        assert s["edges"] == snaps[0]["edges"] and s["nodes"] == snaps[0]["nodes"]


def test_hover_lists_numeric_columns_used_for_style(dataset, monkeypatch, scene_kwargs):
    nodes = dataset.nodes_df.reset_index(drop=True).copy()
    size = np.linspace(1.0, 2.0, len(nodes))
    nodes["strength"] = size
    nodes["degree"] = np.arange(len(nodes)) * 1000.0
    plotter = bnv.BrainNetPlotter(bnv.load(next(iter(dataset.matrices.values())), nodes))
    snaps = _open_window(plotter, monkeypatch, moves=(3,), **{**scene_kwargs, "node_size": "strength", "node_color": "degree"})
    assert snaps[1]["hover"][3:] == [("strength", f"{size[3]:.4g}"), ("degree", "3000")]


def test_hover_can_be_turned_off(dataset, monkeypatch, scene_kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    snaps = _open_window(plotter, monkeypatch, hover_info=False, **scene_kwargs)
    assert "MouseMove" not in snaps[0]["events"]


def test_hover_card_image_has_rounded_transparent_corners():
    from brainnet3d.viz.window import _card_image

    img = _card_image("L_Default_1", (0.2, 0.4, 0.8), [("Network", "Default"), ("strength", "1.5")])
    assert img.dtype == np.uint8 and img.shape[2] == 4
    assert img[0, 0, 3] == 0 and img[-1, -1, 3] == 0
    assert img[img.shape[0] // 2, img.shape[1] // 2, 3] > 200


def test_plot_views_rejects_hover_info(dataset):
    with pytest.raises(TypeError, match="hover_info"):
        bnv.BrainNetPlotter(dataset, subject_id="mean").plot_views(hover_info=True)


def _static_edges(plotter, monkeypatch, **kwargs) -> dict:
    static = _static_actors(plotter, monkeypatch, **kwargs)
    return {tuple(a._endpoints): (a._orig_color, a._orig_alpha) for a in static if hasattr(a, "_endpoints")}


def test_slider_shows_the_edges_a_static_plot_would(dataset, monkeypatch, surfaces):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    base = dict(surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.3)
    snaps = _open_window(plotter, monkeypatch, slides=(0.1, 0.5, 0.3), **base)
    weights = np.abs(plotter._get_matrix(plotter.dataset.nodes_df)[np.triu_indices(len(plotter.dataset.nodes_df), 1)])
    assert snaps[0]["slider"] == pytest.approx((0.0, weights.max()))
    initial = _static_edges(plotter, monkeypatch, **base)
    _assert_edges(snaps[0]["edges"], _expected(initial, None))
    for snap, t in zip(snaps[1:], (0.1, 0.5, 0.3)):
        assert set(snap["edges"]) == set(_static_edges(plotter, monkeypatch, **{**base, "edge_threshold": t}))
    _assert_edges(snaps[-1]["edges"], _expected(initial, None))


def test_slider_keeps_click_highlight(dataset, monkeypatch, surfaces):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    base = dict(surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.3, highlight_on_click=True)
    snaps = _open_window(plotter, monkeypatch, clicks=(5,), slides=(0.5,), **base)
    high = _static_edges(plotter, monkeypatch, **{**base, "edge_threshold": 0.5})
    _assert_edges(snaps[-1]["edges"], _expected(high, 5))


@pytest.mark.parametrize("style", ["bundled", "tube"])
def test_slider_starts_at_plot_threshold_for_bundles_and_tubes(dataset, monkeypatch, surfaces, style):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    base = dict(surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.3, **_EDGE_STYLES[style])
    snaps = _open_window(plotter, monkeypatch, slides=(0.1, 0.5), **base)
    assert snaps[0]["slider"][0] == pytest.approx(0.3)
    assert set(snaps[1]["edges"]) == set(_static_edges(plotter, monkeypatch, **base))
    assert set(snaps[2]["edges"]) == set(_static_edges(plotter, monkeypatch, **{**base, "edge_threshold": 0.5}))


@pytest.mark.parametrize("direction", ["above", "below"])
def test_slider_follows_threshold_direction(dataset, monkeypatch, surfaces, direction):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    base = dict(surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.02, edge_threshold_dir=direction)
    snaps = _open_window(plotter, monkeypatch, slides=(0.0, 0.1), **base)
    for snap, t in zip(snaps[1:], (0.0, 0.1)):
        assert set(snap["edges"]) == set(_static_edges(plotter, monkeypatch, **{**base, "edge_threshold": t}))


def test_window_controls_can_be_turned_off(dataset, monkeypatch, scene_kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    snaps = _open_window(plotter, monkeypatch, window_controls=False, **scene_kwargs)
    assert snaps[0]["slider"] is None


def _legend_case(dataset, monkeypatch, surfaces, clicks=(), moves=(), slides=(), **kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    base = dict(surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.1, node_color="network", **kwargs)
    snaps = _open_window(plotter, monkeypatch, clicks=clicks, moves=moves, slides=slides, **base)
    nets = plotter.dataset.nodes_df["network"].astype(str).to_numpy()
    return snaps, nets, plotter, base


def test_legend_lists_networks_in_their_node_colours(dataset, monkeypatch, surfaces):
    snaps, nets, _, _ = _legend_case(dataset, monkeypatch, surfaces)
    legend = snaps[0]["legend"]
    assert legend.names == ["Default", "Salience", "Visual"]
    for name, rgb in zip(legend.names, legend.colors):
        node = int(np.flatnonzero(nets == name)[0])
        np.testing.assert_allclose(rgb, snaps[0]["nodes"][node][:3], atol=_TOL)


def test_legend_click_hides_and_restores_a_network(dataset, monkeypatch, surfaces):
    snaps, nets, plotter, base = _legend_case(
        dataset, monkeypatch, surfaces, clicks=(("legend", "Visual"), ("legend", "Visual")),
    )
    full, hidden, back = snaps
    visual = set(np.flatnonzero(nets == "Visual").tolist())
    assert visual <= set(full["nodes"])
    assert set(hidden["nodes"]) == set(full["nodes"]) - visual
    assert set(hidden["edges"]) == {e for e in full["edges"] if not set(e) & visual}
    assert {e for e in full["edges"] if set(e) & visual}
    assert back["edges"] == full["edges"] and back["nodes"] == full["nodes"]


def test_hidden_network_stays_hidden_when_sliding(dataset, monkeypatch, surfaces):
    snaps, nets, plotter, base = _legend_case(
        dataset, monkeypatch, surfaces, clicks=(("legend", "Default"),), slides=(0.0,),
    )
    default = set(np.flatnonzero(nets == "Default").tolist())
    static = _static_edges(plotter, monkeypatch, **{**base, "edge_threshold": 0.0})
    assert set(snaps[-1]["edges"]) == {e for e in static if not set(e) & default}


def test_hidden_nodes_show_no_hover_card(dataset, monkeypatch, surfaces):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    nets = plotter.dataset.nodes_df["network"].astype(str).to_numpy()
    node = int(np.flatnonzero(nets == "Salience")[0])
    snaps = _open_window(
        plotter, monkeypatch, clicks=(("legend", "Salience"),), moves=(node,),
        surface_L=surfaces[0], surface_R=surfaces[1], edge_threshold=0.1,
    )
    assert snaps[-1]["hover"] is None


def test_no_legend_without_network_column(dataset, monkeypatch, scene_kwargs):
    nodes = dataset.nodes_df.drop(columns="network")
    plotter = bnv.BrainNetPlotter(bnv.load(next(iter(dataset.matrices.values())), nodes))
    assert _open_window(plotter, monkeypatch, node_color="grey", **scene_kwargs)[0]["legend"] is None


def test_no_legend_without_window_controls(dataset, monkeypatch, scene_kwargs):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    assert _open_window(plotter, monkeypatch, window_controls=False, **scene_kwargs)[0]["legend"] is None
