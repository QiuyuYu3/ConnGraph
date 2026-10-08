import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import brainnet3d as bnv
from brainnet3d.viz.matrix_style import matrix_order


def _blocks(sizes, within=0.6, seed=0):
    rng = np.random.default_rng(seed)
    nets = np.repeat([f"net{k}" for k in range(len(sizes))], sizes)
    m = np.where(nets[:, None] == nets[None, :], within, 0.0) + rng.normal(0, 0.05, (len(nets), len(nets)))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return m, nets


def _shuffle(m, nets, seed=1):
    perm = np.random.default_rng(seed).permutation(len(nets))
    return m[np.ix_(perm, perm)], nets[perm]


def _contiguous(seq):
    seen, prev = set(), None
    for x in seq:
        if x != prev and x in seen:
            return False
        seen.add(x)
        prev = x
    return True


def _image(ax):
    return ax.images[0].get_array()


def test_network_order_follows_given_sequence():
    nets = np.array(["b", "a", "b", "c", "a"])
    idx = matrix_order(np.eye(5), nets, "network", network_order=["c", "b"])
    assert nets[idx].tolist() == ["c", "b", "b", "a", "a"]
    with pytest.raises(ValueError, match="network_order"):
        matrix_order(np.eye(5), nets, "network", network_order=["x"])
    with pytest.raises(ValueError, match="more than once"):
        matrix_order(np.eye(5), nets, "network", network_order=["c", "b", "c"])


def test_cluster_order_groups_blocks():
    m, nets = _shuffle(*_blocks([6, 5, 7]))
    assert _contiguous(nets[matrix_order(m, nets, "cluster")])
    assert _contiguous(nets[matrix_order(m, None, "cluster")])


def test_network_cluster_order_keeps_networks_contiguous():
    m, nets = _shuffle(*_blocks([6, 5, 7]))
    idx = matrix_order(m, nets, "network_cluster")
    assert nets[idx].tolist() == sorted(nets.tolist())


def _gradient(seed=0):
    # networks sit on a line in an order unlike their names; connectivity decays along the line
    centre = {"c": 0.0, "a": 1.0, "d": 2.0, "b": 3.0}
    nets = np.repeat(list(centre), 6)
    pos = np.array([centre[s] for s in nets]) + np.tile(np.linspace(-0.4, 0.4, 6), 4)
    perm = np.random.default_rng(seed).permutation(len(nets))
    nets, pos = nets[perm], pos[perm]
    return np.exp(-np.abs(pos[:, None] - pos[None, :])) - 0.3, nets, pos


def test_network_chain_follows_connection_strength():
    m, nets, pos = _gradient()
    idx = matrix_order(m, nets, "network_chain")
    runs = [s for k, s in enumerate(nets[idx]) if k == 0 or s != nets[idx][k - 1]]
    assert runs in (["c", "a", "d", "b"], ["b", "d", "a", "c"])
    steps = np.diff(pos[idx])
    assert np.all(steps > 0) or np.all(steps < 0)


def test_network_chain_uses_the_sign_of_connections():
    m, nets, pos = _gradient()
    flipped = np.where(np.abs(pos[:, None] - pos[None, :]) > 1.5, 0.9, 0.0)
    idx = matrix_order(m - flipped, nets, "network_chain")
    runs = [s for k, s in enumerate(nets[idx]) if k == 0 or s != nets[idx][k - 1]]
    assert runs in (["c", "a", "d", "b"], ["b", "d", "a", "c"])


def test_network_chain_in_heatmap_and_nbs():
    m, nets, _ = _gradient()
    with pytest.raises(ValueError, match="network_order"):
        matrix_order(m, nets, "network_chain", network_order=["a"])
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets, order="network_chain")
    assert ax.get_legend() is None
    plt.close(fig)
    fig = bnv.plot_nbs_matrices(m, m, np.zeros_like(m), network_labels=nets, order="network_chain")
    plt.close(fig)


@pytest.mark.parametrize("kwargs, match", [
    (dict(order="alphabet"), "order"),
    (dict(order="network_cluster"), "network_labels"),
])
def test_bad_order_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        matrix_order(np.eye(3), None, **kwargs)


def test_heatmap_blanks_constant_diagonal_only():
    m, nets = _blocks([5, 5])
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets)
    img = _image(ax)
    assert all(tuple(img[i, i]) == (1, 1, 1, 1) for i in range(len(nets)))
    plt.close(fig)

    np.fill_diagonal(m, np.linspace(-0.5, 0.5, len(nets)))
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets)
    assert not any(tuple(_image(ax)[i, i]) == (1, 1, 1, 1) for i in range(len(nets)))
    plt.close(fig)


def test_heatmap_strips_and_network_names_for_large_matrices():
    m, nets = _blocks([20, 15, 10])
    labels = [f"roi{k}" for k in range(len(nets))]
    fig, ax = bnv.matrix_heatmap(m, labels=labels, network_labels=nets)
    assert len(fig.axes) == 4
    names = {t.get_text() for a in fig.axes for t in a.texts}
    assert {"net0", "net1", "net2"} <= names
    plt.close(fig)


def test_heatmap_roi_ticks_for_small_matrices(dataset):
    m = dataset.mean_matrix()
    nets = dataset.nodes_df.set_index("label").loc[m.columns, "network"].tolist()
    fig, ax = bnv.matrix_heatmap(m.values, labels=list(m.columns), network_labels=nets)
    ticks = {t.get_text() for a in fig.axes for t in a.get_yticklabels()}
    assert set(m.columns) <= ticks
    plt.close(fig)


def test_heatmap_without_networks_keeps_input_order_and_ticks(dataset):
    m = dataset.mean_matrix()
    fig, ax = bnv.matrix_heatmap(m.values, labels=list(m.columns))
    assert len(fig.axes) == 2
    assert [t.get_text() for t in ax.get_yticklabels()] == list(m.columns)
    plt.close(fig)


def test_heatmap_boundaries():
    m, nets = _blocks([5, 5, 5])
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets, network_boundaries="boxes")
    assert len([p for p in ax.patches if isinstance(p, Rectangle)]) == 3 and not ax.lines
    plt.close(fig)
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets)
    assert len(ax.lines) == 4 and not ax.patches
    plt.close(fig)


def test_heatmap_cluster_order_adds_network_legend():
    m, _ = _blocks([5, 5, 5])
    nets = np.array(["a", "b", "c"] * 5)
    fig, ax = bnv.matrix_heatmap(m, network_labels=nets, order="cluster")
    legend = ax.get_legend()
    assert legend is not None and {t.get_text() for t in legend.get_texts()} == set(nets)
    plt.close(fig)


def _nbs(sizes):
    m, nets = _blocks(sizes)
    adj = np.zeros_like(m)
    adj[0, -1] = adj[-1, 0] = adj[1, -2] = adj[-2, 1] = 1
    return m, m + 0.1, adj, nets


def test_nbs_outlines_small_matrices():
    g1, g2, adj, nets = _nbs([5, 5])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, network_labels=nets)
    sig_ax = [a for a in fig.axes if a.images][2]
    assert any(isinstance(c, LineCollection) for c in sig_ax.collections)
    plt.close(fig)


def test_nbs_fades_large_matrices():
    g1, g2, adj, nets = _nbs([40, 30])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, network_labels=nets)
    marked = _image([a for a in fig.axes if a.images][2])
    assert not any(isinstance(c, LineCollection) for c in fig.axes[2].collections)
    full = plt.get_cmap("RdBu_r")(0.0)
    np.testing.assert_allclose(marked[0, -1], full)
    assert marked[0, 1, :3].min() > min(full[:3])
    plt.close(fig)


def _colorbar_limits(fig):
    return [a.get_ylim() for a in fig.axes if hasattr(a, "_colorbar")]


def test_nbs_third_panel_shows_group_difference():
    g1, g2, adj, nets = _nbs([5, 5])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, network_labels=nets, group_names=("HC", "PT"))
    limits = _colorbar_limits(fig)
    assert len(limits) == 2
    np.testing.assert_allclose(limits[0], (-0.5, 0.5))
    np.testing.assert_allclose(limits[1], (-0.1, 0.1))
    titles = {a.get_title() for a in fig.axes}
    assert "HC − PT" in titles
    plt.close(fig)


def test_nbs_difference_range_can_be_set():
    g1, g2, adj, nets = _nbs([5, 5])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, diff_vmax=0.3)
    np.testing.assert_allclose(_colorbar_limits(fig)[1], (-0.3, 0.3))
    plt.close(fig)


def _panel_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    main, extra = fig.axes[:3], fig.axes[3:]
    groups = [[main[k], *extra[3 * k: 3 * k + 3]] for k in range(3)]
    return [Bbox.union([b for a in g if (b := a.get_tightbbox(renderer)) is not None]) for g in groups]


def test_nbs_panels_clear_long_roi_labels():
    g1, g2, adj, nets = _nbs([15, 15])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, labels=[f"L_CinguloOperc_{i}" for i in range(30)], network_labels=nets)
    boxes = _panel_boxes(fig)
    assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:]))
    plt.close(fig)


def test_nbs_width_unchanged_without_overlap():
    g1, g2, adj, nets = _nbs([40, 30])
    fig = bnv.plot_nbs_matrices(g1, g2, adj, network_labels=nets)
    assert fig.get_size_inches()[0] == pytest.approx(19)
    plt.close(fig)


def test_nbs_rejects_unknown_options():
    g1, g2, adj, nets = _nbs([5, 5])
    with pytest.raises(TypeError):
        bnv.plot_nbs_matrices(g1, g2, adj, colour="red")
    with pytest.raises(ValueError, match="sig_style"):
        bnv.plot_nbs_matrices(g1, g2, adj, sig_style="glow")
