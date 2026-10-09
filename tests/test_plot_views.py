import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import conngraph as bnv
from conngraph.exceptions import DataValidationError
from conngraph.viz.colormap import labels_to_colors


def _panels(fig):
    return [ax for ax in fig.axes if ax.images]


def _legends(fig):
    return {ax.get_label().removeprefix("legend:"): ax for ax in fig.axes if ax.get_label().startswith("legend:")}


def _views(dataset, **kwargs):
    kwargs.setdefault("views", [{"view": "L"}])
    return bnv.BrainNetPlotter(dataset, subject_id="mean").plot_views(panel_size=200, **kwargs)


def test_plot_views_returns_one_panel_per_view(dataset):
    fig = _views(dataset, views=[[{"view": "L"}, {"view": "R"}], [{"view": "S", "title": "Top"}]], edge_threshold=0.4)
    assert isinstance(fig, Figure)
    assert len(_panels(fig)) == 3
    assert "Top" in [t.get_text() for t in fig.texts]
    plt.close(fig)


def test_hemisphere_panel_keeps_only_that_hemisphere(dataset, surfaces):
    from conngraph.viz.panels import panel_actors

    left, right = surfaces
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    args = plotter._plot_args(surface_L=left, surface_R=right, edge_threshold=-1.0, edge_threshold_dir="above")
    scene = plotter._build_scene(args, "both")
    hemi = scene.nodes_df["hemisphere"].to_numpy()

    actors = panel_actors(scene, "L")
    nodes = [a for a in actors if hasattr(a, "_node_idx")]
    edges = [a for a in actors if hasattr(a, "_endpoints")]
    assert len(nodes) == (hemi == "L").sum() and all(hemi[a._node_idx] == "L" for a in nodes)
    assert edges and all(hemi[e._endpoints[0]] == "L" and hemi[e._endpoints[1]] == "L" for e in edges)
    assert [a._hemisphere for a in actors if hasattr(a, "_hemisphere")] == ["L"]


def test_default_legends_cover_styles_mapped_to_data(dataset):
    fig = _views(dataset, node_color="network", node_size="x", edge_threshold=0.3)
    assert set(_legends(fig)) == {"node_color", "node_size", "edge_color"}
    plt.close(fig)


def test_fixed_styles_get_no_legend(dataset):
    fig = _views(dataset, node_color="grey", node_size=3.0, edge_color="black")
    assert _legends(fig) == {}
    plt.close(fig)


def test_category_legend_matches_node_colours(dataset):
    fig = _views(dataset, node_color="network")
    legend = _legends(fig)["node_color"].get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    networks = dataset.nodes_df["network"].astype(str).tolist()
    expected = dict(zip(networks, labels_to_colors(networks)))
    assert labels == list(expected)
    np.testing.assert_allclose([p.get_facecolor()[:3] for p in legend.get_patches()], list(expected.values()))
    plt.close(fig)


def test_colorbars_use_drawn_limits(dataset):
    fig = _views(dataset, node_color="x", edge_threshold=0.3)
    legends = _legends(fig)
    x = dataset.nodes_df["x"].to_numpy(float)
    np.testing.assert_allclose(legends["node_color"].get_xlim(), (x.min(), x.max()))

    m = dataset.mean_matrix().to_numpy()
    w = m[np.triu_indices_from(m, 1)]
    peak = np.abs(w[np.abs(w) > 0.3]).max()
    np.testing.assert_allclose(legends["edge_color"].get_xlim(), (-peak, peak))
    plt.close(fig)


def test_edge_width_legend_on_request(dataset):
    fig = _views(dataset, legend=["edge_width"], edge_threshold=0.3)
    assert set(_legends(fig)) == {"edge_width"}
    plt.close(fig)


def test_legend_accepts_one_name(dataset):
    fig = _views(dataset, legend="node_color", node_color="network")
    assert set(_legends(fig)) == {"node_color"}
    plt.close(fig)


_THREE = [{"view": "L", "hemisphere": "L"}, {"view": "S"}, {"view": "R", "hemisphere": "L"}]


def test_width_sets_figure_width(dataset):
    fig = _views(dataset, views=_THREE, width=3.5, node_color="network")
    assert fig.get_size_inches()[0] == pytest.approx(3.5)
    assert max(ax.get_position().x1 for ax in _panels(fig)) <= 1.0
    plt.close(fig)


def test_width_wraps_legend_rows(dataset):
    names = ["node_color", "node_size", "edge_color", "edge_width"]
    one_row = _legends(_views(dataset, legend=names, node_color="network", node_size="x", edge_threshold=0.3))
    wrapped = _legends(_views(dataset, width=3.5, legend=names, node_color="network", node_size="x", edge_threshold=0.3))
    assert one_row["node_size"].get_position().y1 == pytest.approx(one_row["edge_width"].get_position().y1)
    assert wrapped["node_size"].get_position().y1 > wrapped["edge_width"].get_position().y1
    plt.close("all")


def test_width_keeps_one_title_size_and_print_line_widths(dataset):
    kwargs = dict(views=_THREE, edge_threshold=0.3, legend=["edge_width"])
    wide, narrow = _views(dataset, **kwargs), _views(dataset, width=3.5, **kwargs)
    assert len({t.get_fontsize() for t in narrow.texts}) == 1
    widths = [[line.get_linewidth() for line in _legends(f)["edge_width"].lines] for f in (wide, narrow)]
    np.testing.assert_allclose(widths[1], widths[0], rtol=0.15)
    plt.close("all")


def test_width_too_small_raises(dataset):
    with pytest.raises(ValueError, match="width"):
        _views(dataset, views=_THREE, width=0.5)


def test_legend_values_are_round_numbers():
    from conngraph.viz.panels import _number, _round_values

    np.testing.assert_allclose(_round_values(16.0, 1940.0), [500, 1000, 1500])
    np.testing.assert_allclose(_round_values(0.55, 0.705), [0.6, 0.65, 0.7])
    assert [_number(v) for v in (1500.0, 0.65, 1940.3, 0.5533)] == ["1,500", "0.65", "1,940", "0.553"]


def test_size_legend_shows_round_values(dataset):
    from conngraph.viz.panels import _number, _round_values

    fig = _views(dataset, legend=["node_size"], node_size="x")
    x = dataset.nodes_df["x"].to_numpy(float)
    texts = [t.get_text() for t in _legends(fig)["node_size"].texts]
    assert texts == ["x"] + [_number(v) for v in _round_values(x.min(), x.max())]
    plt.close(fig)


def test_legend_false_draws_none(dataset):
    fig = _views(dataset, legend=False, node_color="network", node_size="x")
    assert _legends(fig) == {}
    plt.close(fig)


def test_requested_legend_without_data_warns(dataset):
    with pytest.warns(UserWarning, match="node_size"):
        fig = _views(dataset, legend=["node_size"], node_size=3.0)
    plt.close(fig)


@pytest.mark.parametrize("views, match", [
    ([{"view": "X"}], "view"),
    ([{"view": "L", "hemi": "L"}], "hemi"),
    ([{"view": "L", "hemisphere": "both sides"}], "hemisphere"),
    ([], "no panels"),
    ([[{"view": "L"}], []], "empty row"),
])
def test_invalid_panels_raise(dataset, views, match):
    with pytest.raises(ValueError, match=match):
        _views(dataset, views=views)


def test_hemisphere_panel_needs_hemisphere_column(dataset):
    nodes_df = dataset.nodes_df.drop(columns="hemisphere")
    plotter = bnv.BrainNetPlotter(bnv.ConnectivityDataset(dataset.matrices, nodes_df), subject_id="mean")
    with pytest.raises(DataValidationError, match="hemisphere"):
        plotter.plot_views(views=[{"view": "R", "hemisphere": "L"}], panel_size=200)


@pytest.mark.parametrize("kwargs", [dict(show_hemisphere="L"), dict(screenshot="a.png"), dict(not_a_style=1)])
def test_unsupported_keywords_raise(dataset, kwargs):
    with pytest.raises(TypeError):
        _views(dataset, **kwargs)


def test_sign_legend_lists_edge_signs(dataset):
    fig = bnv.BrainNetPlotter(dataset, subject_id="sub-01").plot_views(
        views=[{"view": "L"}], panel_size=200, legend=["edge_color"], edge_threshold=0.02, edge_color="sign",
    )
    legend = _legends(fig)["edge_color"].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["positive", "negative"]
    np.testing.assert_allclose(
        [p.get_facecolor()[:3] for p in legend.get_patches()], [(1, 0.25, 0.25), (0.25, 0.25, 1)]
    )
    plt.close(fig)


def test_plot_views_accepts_highlight(dataset):
    fig = _views(dataset, views=[{"view": "L", "hemisphere": "L"}], highlight_nodes={"network": "Default"})
    assert len(_panels(fig)) == 1
    plt.close(fig)


def test_panel_render_is_not_blank(dataset, surfaces):
    # translucent surfaces exercise depth peeling, which a software OpenGL driver may render blank or black
    left, right = surfaces
    fig = _views(dataset, surface_L=left, surface_R=right, node_color="red", edge_threshold=0.3, legend=False)
    image = np.asarray(_panels(fig)[0].images[0].get_array())
    assert image.shape[-1] == 3 or (image[..., 3] == 255).all()
    r, g, b = np.moveaxis(image[..., :3].astype(float), -1, 0)
    drawn = (np.abs(np.stack([r, g, b]) - 255).max(axis=0) > 8).mean()
    red = ((r > 120) & (g < 0.5 * r) & (b < 0.5 * r)).mean()
    assert drawn > 0.1 and red > 0.01
    plt.close(fig)


def _tick_labels(fig, ax):
    fig.canvas.draw()
    lo, hi = sorted(ax.get_xlim())
    ticks = [t for t in ax.xaxis.get_major_ticks() if lo <= t.get_loc() <= hi and t.label1.get_visible()]
    return [t.label1 for t in ticks]


@pytest.mark.parametrize("width", [3.5, 10.5])
def test_colorbar_ticks_are_three_round_values_that_do_not_overlap(dataset, width):
    fig = _views(dataset, edge_threshold=0.4, node_color="x", edge_colorvminvmax=(-0.12, 0.12), width=width,
                 legend=["node_color", "edge_color"])
    for name in ("node_color", "edge_color"):
        labels = _tick_labels(fig, _legends(fig)[name])
        assert len(labels) == 3
        boxes = [lab.get_window_extent() for lab in labels]
        assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:]))
    edge_ticks = [t.get_loc() for t in _legends(fig)["edge_color"].xaxis.get_major_ticks()]
    assert np.allclose(sorted(edge_ticks)[:3], [-0.1, 0.0, 0.1])
    plt.close(fig)


def test_legend_titles_replace_default_titles(dataset):
    fig = _views(dataset, edge_threshold=0.4, legend=["node_color", "edge_color"],
                 legend_titles={"edge_color": "group difference", "node_color": "Network"})
    legends = _legends(fig)
    assert legends["edge_color"].get_title() == "group difference"
    assert legends["node_color"].get_legend().get_title().get_text() == "Network"
    plt.close(fig)


def test_legend_titles_rejects_unknown_names(dataset):
    with pytest.raises(ValueError, match="legend_titles"):
        _views(dataset, edge_threshold=0.4, legend_titles={"edge_colour": "x"})
