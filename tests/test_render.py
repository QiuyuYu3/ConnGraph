import re
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import brainnet3d as bnv


def _net2color(network_labels):
    cmap = plt.get_cmap("Set2")
    return {n: cmap(i % cmap.N) for i, n in enumerate(sorted(set(network_labels)))}


def _assert_written(path):
    assert path.exists() and path.stat().st_size > 0


def _save_fig(fig, path):
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _assert_written(path)


def test_brainnet_plotter(dataset, out_dir):
    path = out_dir / "brainnet_basic.png"
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(
        node_color="network", edge_threshold=0.4, interactive=False, screenshot=str(path),
    )
    _assert_written(path)


def test_brainnet_threshold_dir(dataset, out_dir):
    path = out_dir / "brainnet_above.png"
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(
        node_color="network", edge_threshold=0.4, edge_threshold_dir="above",
        interactive=False, screenshot=str(path),
    )
    _assert_written(path)


def test_brainnet_absmax(dataset, out_dir):
    pytest.importorskip("bct")
    pytest.importorskip("topcorr")
    result = bnv.compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="node", n_jobs=1)

    p = bnv.BrainNetPlotter(dataset, subject_id="sub-01")
    p.attach_metrics(result)
    assert set(p._extra_cols) == {"clust_coeff", "btwn_cent", "strength", "ge_local"}
    assert set(p._extra_cols["strength"]) == set(dataset.nodes_df["label"])

    path = out_dir / "brainnet_absmax.png"
    p.plot(
        node_color="strength", node_cmap="RdBu_r", node_colorvminvmax="absmax",
        interactive=False, screenshot=str(path),
    )
    _assert_written(path)


def test_arrowaxis(dataset, out_dir):
    path = out_dir / "brainnet_arrows.png"
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(
        node_color="network", edge_threshold=0.4, arrowaxis="all",
        interactive=False, screenshot=str(path),
    )
    _assert_written(path)


def test_spring_plot(dataset, out_dir):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.4)
    network_labels = dataset.nodes_df["network"].tolist()
    fig, _ = bnv.spring_plot(G, dataset.nodes_df["label"].tolist(), network_labels, _net2color(network_labels))
    _save_fig(fig, out_dir / "spring_2d.png")


def _spring_plot_sizes(G):
    from matplotlib.collections import LineCollection, PathCollection

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, ax = bnv.spring_plot(G, list("abcd"), ["A", "A", "B", "B"], {"A": "red", "B": "blue"})
    widths = [w for c in ax.collections if isinstance(c, LineCollection) for w in c.get_linewidths()]
    sizes = [s for c in ax.collections if isinstance(c, PathCollection) for s in c.get_sizes()]
    plt.close(fig)
    return np.array(widths, dtype=float), np.array(sizes, dtype=float)


def test_axis_arrows_accept_single_axis_string():
    assert len(bnv.make_axis_arrows("LR")) == 2
    assert len(bnv.make_axis_arrows("all")) == 6
    with pytest.raises(ValueError, match="'XY'"):
        bnv.make_axis_arrows("XY")


def test_spring_plot_negative_weights():
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 0.8), (1, 2, -0.6), (2, 3, 0.5)])
    widths, sizes = _spring_plot_sizes(G)
    np.testing.assert_allclose(sorted(widths), sorted(np.power([0.8, 0.6, 0.5], 1.5)))
    assert np.isfinite(sizes).all() and sizes.argmax() == 1


def test_spring_plot_without_edges():
    _, sizes = _spring_plot_sizes(nx.empty_graph(4))
    np.testing.assert_array_equal(sizes, 50)


def test_spring_plot_3d(dataset, out_dir):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.4)
    path = out_dir / "spring_3d.png"
    bnv.spring_plot_3d(G, interactive=False, save_path=str(path))
    _assert_written(path)


def test_spring_communities(dataset, out_dir):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.35)
    ids = bnv.detect_communities(G)
    cmap = plt.get_cmap("tab10")
    path = out_dir / "spring_3d_communities.png"
    bnv.spring_plot_3d(G, node_colors=[cmap(i % cmap.N)[:3] for i in ids], interactive=False, save_path=str(path))
    _assert_written(path)


def test_circos_plot(dataset, out_dir):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.4)
    network_labels = dataset.nodes_df["network"].tolist()
    fig, _ = bnv.circos_plot(G, dataset.nodes_df["label"].tolist(), network_labels, _net2color(network_labels))
    _save_fig(fig, out_dir / "circos.png")


def test_circos_labels_face_outward(dataset):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.4)
    network_labels = dataset.nodes_df["network"].tolist()
    fig, ax = bnv.circos_plot(G, dataset.nodes_df["label"].tolist(), network_labels, _net2color(network_labels))
    checked = 0
    for text in ax.texts:
        x, _ = text.get_position()
        if abs(x) > 1e-6:
            assert text.get_horizontalalignment() == ("left" if x > 0 else "right"), text.get_text()
            checked += 1
    plt.close(fig)
    assert checked > 0


def test_matrix_heatmap(dataset, out_dir):
    fig, _ = bnv.matrix_heatmap(
        dataset.mean_matrix().values,
        labels=dataset.nodes_df["label"].tolist(),
        network_labels=dataset.nodes_df["network"].tolist(),
        title="Mock mean connectivity",
    )
    _save_fig(fig, out_dir / "matrix_heatmap.png")


def test_nbs(groups, out_dir):
    pytest.importorskip("bct")
    g1_mats, g2_mats = groups
    result = bnv.run_nbs(g1_mats, g2_mats, thresh=1.5, k=50, verbose=False)
    mean_g1 = np.stack([m.values for m in g1_mats.values()]).mean(axis=0)
    mean_g2 = np.stack([m.values for m in g2_mats.values()]).mean(axis=0)
    adj = result.adj.sum(axis=2) if result.adj.ndim == 3 else result.adj
    _save_fig(bnv.plot_nbs_matrices(mean_g1, mean_g2, adj, labels=result.labels), out_dir / "nbs_heatmap.png")


def test_nbs_highlight(dataset, groups, out_dir):
    pytest.importorskip("bct")
    g1_mats, g2_mats = groups
    result = bnv.run_nbs(g1_mats, g2_mats, thresh=1.5, k=50, verbose=False)
    adj = result.adj.sum(axis=2) if result.adj.ndim == 3 else result.adj
    path = out_dir / "nbs_highlight_3d.png"
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(
        node_color="network", edge_threshold=0.0, edge_threshold_dir="above",
        highlight_edges=adj, highlight_level=0.85, interactive=False, screenshot=str(path),
    )
    _assert_written(path)


def test_orbit_gif(dataset, out_dir):
    from vedo import Sphere

    spheres = [Sphere(pos, r=3, c="steelblue") for pos in dataset.nodes_df[["x", "y", "z"]].values]
    path = out_dir / "orbit.gif"
    bnv.save_orbit_gif(spheres, output_path=str(path), n_frames=8, fps=4, verbose=False)
    _assert_written(path)


def test_plot_without_window_renders_offscreen(dataset, out_dir, monkeypatch):
    import vedo

    created = []

    class RecordingPlotter(vedo.Plotter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(vedo, "Plotter", RecordingPlotter)
    path = out_dir / "brainnet_offscreen.png"
    if path.exists():
        path.unlink()
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(interactive=False, screenshot=str(path))

    assert created and created[0].offscreen
    _assert_written(path)


def _assert_image(img):
    assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] >= 3
    assert np.ptp(img) > 0


def test_plot_default_returns_image(dataset):
    _assert_image(bnv.BrainNetPlotter(dataset, subject_id="mean").plot())


def test_plot_writes_standalone_html(dataset, out_dir):
    pytest.importorskip("k3d")
    path = out_dir / "brainnet.html"
    if path.exists():
        path.unlink()
    img = bnv.BrainNetPlotter(dataset, subject_id="mean").plot(html=str(path))

    _assert_image(img)
    _assert_written(path)
    page = path.read_text(encoding="utf-8")
    assert "K3D" in page
    assert not re.search(r"""(?:src|href)=["']https?://""", page)


def test_spring_plot_3d_default_returns_image(dataset):
    G = bnv.threshold_graph(dataset.mean_matrix().values, threshold=0.4)
    _assert_image(bnv.spring_plot_3d(G))


def _plot_actors(plotter, monkeypatch, **kwargs):
    import brainnet3d.viz.views as views

    captured = {}
    monkeypatch.setattr(views, "_finish_render", lambda vp, actors, *a, **kw: captured.update(actors=actors))
    plotter.plot(**kwargs)
    return captured["actors"]


def test_layout_places_nodes_at_edge_endpoints(dataset, monkeypatch):
    actors = _plot_actors(
        bnv.BrainNetPlotter(dataset, subject_id="mean"), monkeypatch, layout="spring", edge_threshold=0.4,
    )
    centres = {a._node_idx: np.asarray(a.center_of_mass()) for a in actors if hasattr(a, "_node_idx")}
    edges = [a for a in actors if hasattr(a, "_endpoints")]
    assert edges
    for e in edges:
        i, j = e._endpoints
        ends = np.asarray(e.vertices)
        np.testing.assert_allclose(ends[0], centres[i], atol=1e-3)
        np.testing.assert_allclose(ends[-1], centres[j], atol=1e-3)


def test_kamada_kawai_puts_strong_edges_closer(dataset):
    ring = np.zeros((4, 4))
    for (i, j), w in {(0, 1): 0.95, (1, 2): 0.6, (2, 3): 0.95, (0, 3): 0.6}.items():
        ring[i, j] = ring[j, i] = w
    pos = bnv.BrainNetPlotter(dataset)._compute_layout(ring, "kamada_kawai", threshold=0.5, seed=0)
    d = lambda i, j: np.linalg.norm(pos[i] - pos[j])
    assert d(0, 1) < d(1, 2) and d(2, 3) < d(0, 3)


def test_kamada_kawai_is_reproducible_with_seed(dataset):
    matrix = dataset.mean_matrix().values
    plotter = bnv.BrainNetPlotter(dataset)
    a = plotter._compute_layout(matrix, "kamada_kawai", threshold=0.4, seed=3)
    b = plotter._compute_layout(matrix, "kamada_kawai", threshold=0.4, seed=3)
    np.testing.assert_array_equal(a, b)


def test_forceatlas2_layout_is_reproducible_with_seed(dataset):
    matrix = dataset.mean_matrix().values
    plotter = bnv.BrainNetPlotter(dataset)
    a = plotter._compute_layout(matrix, "forceatlas2", threshold=0.4, seed=3)
    b = plotter._compute_layout(matrix, "forceatlas2", threshold=0.4, seed=3)
    assert a.shape == (matrix.shape[0], 3) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_edge_colors_centre_on_zero_by_default(dataset, monkeypatch):
    kwargs = dict(edge_threshold=0.3, edge_threshold_dir="above", edge_color="weight", edge_cmap="RdBu_r")
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")

    centred = [a._orig_color for a in _plot_actors(plotter, monkeypatch, **kwargs) if hasattr(a, "_endpoints")]
    assert centred and all(r > b for r, _, b in centred)

    stretched = [
        a._orig_color for a in _plot_actors(plotter, monkeypatch, edge_colorvminvmax="minmax", **kwargs)
        if hasattr(a, "_endpoints")
    ]
    assert any(b > r for r, _, b in stretched)


def test_highlight_edges_follow_hemisphere_filter(dataset, monkeypatch):
    source = dataset.mean_matrix().columns.tolist()
    r_labels = dataset.nodes_df.loc[dataset.nodes_df["hemisphere"] == "R", "label"].tolist()
    a, b = r_labels[1], r_labels[3]
    hl = np.zeros((len(source), len(source)))
    hl[source.index(a), source.index(b)] = hl[source.index(b), source.index(a)] = 1

    actors = _plot_actors(
        bnv.BrainNetPlotter(dataset, subject_id="mean"), monkeypatch, show_hemisphere="R",
        edge_threshold=-1.0, edge_threshold_dir="above", edge_alpha=0.7, highlight_edges=hl,
    )
    bright = {
        frozenset((r_labels[e._endpoints[0]], r_labels[e._endpoints[1]]))
        for e in actors if hasattr(e, "_endpoints") and e._orig_alpha == 0.7
    }
    assert bright == {frozenset((a, b))}


def test_highlight_edges_shape_mismatch_raises(dataset):
    with pytest.raises(ValueError, match="highlight_edges"):
        bnv.BrainNetPlotter(dataset, subject_id="mean").plot(highlight_edges=np.zeros((3, 3)))


def test_show_hemisphere_hides_other_surface(dataset, surfaces, monkeypatch):
    left, right = surfaces
    actors = _plot_actors(
        bnv.BrainNetPlotter(dataset, subject_id="mean"), monkeypatch,
        surface_L=left, surface_R=right, show_hemisphere="L",
    )
    meshes = [a for a in actors if not hasattr(a, "_node_idx") and not hasattr(a, "_endpoints")]
    assert len(meshes) == 1 and meshes[0].center_of_mass()[0] < 0


def test_edge_colors_by_sign(dataset, monkeypatch):
    actors = _plot_actors(
        bnv.BrainNetPlotter(dataset, subject_id="sub-01"), monkeypatch,
        edge_threshold=0.02, edge_color="sign", edge_sign_colors=("orange", "purple"),
    )
    edges = [a for a in actors if hasattr(a, "_endpoints")]
    pos, neg = plt.matplotlib.colors.to_rgb("orange"), plt.matplotlib.colors.to_rgb("purple")
    assert {e._weight > 0 for e in edges} == {True, False}
    for e in edges:
        np.testing.assert_allclose(e._orig_color, pos if e._weight > 0 else neg)
