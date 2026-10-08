import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

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


def test_cluster_order_groups_blocks():
    m, nets = _shuffle(*_blocks([6, 5, 7]))
    assert _contiguous(nets[matrix_order(m, nets, "cluster")])
    assert _contiguous(nets[matrix_order(m, None, "cluster")])


def test_network_cluster_order_keeps_networks_contiguous():
    m, nets = _shuffle(*_blocks([6, 5, 7]))
    idx = matrix_order(m, nets, "network_cluster")
    assert nets[idx].tolist() == sorted(nets.tolist())


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
    panels = [a for a in fig.axes if a.images]
    plain, marked = _image(panels[0]), _image(panels[2])
    assert not any(isinstance(c, LineCollection) for c in panels[2].collections)
    np.testing.assert_allclose(marked[0, -1], plain[0, -1])
    assert marked[0, 1, :3].min() > plain[0, 1, :3].min()
    plt.close(fig)


def test_nbs_rejects_unknown_options():
    g1, g2, adj, nets = _nbs([5, 5])
    with pytest.raises(TypeError):
        bnv.plot_nbs_matrices(g1, g2, adj, colour="red")
    with pytest.raises(ValueError, match="sig_style"):
        bnv.plot_nbs_matrices(g1, g2, adj, sig_style="glow")
