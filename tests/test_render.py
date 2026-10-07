import re

import matplotlib.pyplot as plt
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
