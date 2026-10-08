import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import brainnet3d as bnv
from brainnet3d.viz.colormap import labels_to_colors


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
    from brainnet3d.viz.panels import panel_actors

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
])
def test_invalid_panels_raise(dataset, views, match):
    with pytest.raises(ValueError, match=match):
        _views(dataset, views=views)


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
