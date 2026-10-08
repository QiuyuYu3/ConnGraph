import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, PathPatch, Polygon, Wedge
from matplotlib.path import Path

import brainnet3d as bnv
from brainnet3d.viz.colormap import labels_to_colors, values_to_colors
from brainnet3d.viz.matrix_style import matrix_order, merge_heights

SIGN = ((1.0, 0.25, 0.25), (0.25, 0.25, 1.0))


@pytest.fixture
def signed_graph(dataset):
    m = dataset.matrices["sub-01"]
    nets = dataset.nodes_df.set_index("label").loc[m.columns, "network"].tolist()
    G = bnv.threshold_graph(m.to_numpy(), threshold=0.02, use_abs=True)
    assert any(w < 0 for _, _, w in G.edges(data="weight"))
    return G, list(m.columns), nets


def _chords(ax):
    return [p for p in ax.patches if isinstance(p, PathPatch)]


def test_circos_default_draws_curved_chords_coloured_by_weight(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets)
    chords = _chords(ax)
    assert len(chords) == G.number_of_edges()
    assert all(Path.CURVE3 in p.get_path().codes for p in chords)
    widths = [p.get_linewidth() for p in chords]
    assert widths == sorted(widths)
    reds = [p.get_edgecolor()[0] > p.get_edgecolor()[2] for p in chords]
    assert any(reds) and not all(reds)
    plt.close(fig)


def test_circos_straight_and_sign_colours(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets, edge_style="straight", edge_color="sign")
    chords = _chords(ax)
    assert all(Path.CURVE3 not in p.get_path().codes for p in chords)
    assert {tuple(np.round(p.get_edgecolor()[:3], 6)) for p in chords} == {SIGN[0], SIGN[1]}
    plt.close(fig)


def test_circos_plain_colour_and_width(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets, edge_color="grey", edge_width=0.5)
    chords = _chords(ax)
    assert {tuple(p.get_edgecolor()[:3]) for p in chords} == {to_rgb("grey")}
    assert {p.get_linewidth() for p in chords} == {0.5}
    plt.close(fig)


def test_circos_ring_replaces_legend(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets)
    assert len([p for p in ax.patches if isinstance(p, Wedge)]) == len(set(nets))
    assert set(nets) <= {t.get_text() for t in ax.texts}
    assert ax.get_legend() is None
    plt.close(fig)

    fig, ax = bnv.circos_plot(G, labels, nets, network_ring=False)
    assert not [p for p in ax.patches if isinstance(p, Wedge)]
    assert ax.get_legend() is not None
    plt.close(fig)


def test_circos_default_colours_match_3d_plot(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets)
    expected = dict(zip(nets, labels_to_colors(nets)))
    wedges = [p for p in ax.patches if isinstance(p, Wedge)]
    assert {tuple(np.round(w.get_facecolor()[:3], 6)) for w in wedges} == {
        tuple(np.round(c, 6)) for c in expected.values()
    }
    plt.close(fig)


def test_circos_keeps_input_order_within_network():
    labels, nets = ["b_10", "a_2", "b_1", "a_11"], ["B", "A", "B", "A"]
    fig, ax = bnv.circos_plot(nx.empty_graph(4), labels, nets)
    texts = [t for t in ax.texts if t.get_text() in labels]
    angles = [(np.arctan2(*t.get_position()[::-1]) - np.pi / 2) % (2 * np.pi) for t in texts]
    assert [texts[k].get_text() for k in np.argsort(angles)] == ["a_2", "a_11", "b_10", "b_1"]
    plt.close(fig)


def _ring_order(ax, labels):
    texts = [t for t in ax.texts if t.get_text() in labels]
    angles = [(np.arctan2(*t.get_position()[::-1]) - np.pi / 2) % (2 * np.pi) for t in texts]
    return [texts[k].get_text() for k in np.argsort(angles)]


def _gradient_graph(spread=0.4, seed=0):
    # networks on a line in an order unlike their names; a wide spread makes neighbouring networks overlap
    centre = {"c": 0.0, "a": 1.0, "d": 2.0, "b": 3.0}
    nets = np.repeat(list(centre), 6)
    pos = np.array([centre[s] for s in nets]) + np.tile(np.linspace(-spread, spread, 6), 4)
    perm = np.random.default_rng(seed).permutation(len(nets))
    nets, pos = nets[perm], pos[perm]
    m = np.exp(-np.abs(pos[:, None] - pos[None, :])) - 0.3
    np.fill_diagonal(m, 0.0)
    G = nx.from_numpy_array(np.where(m > 0.2, m, 0.0))
    return G, [f"r{i}" for i in range(len(nets))], nets.tolist(), m


@pytest.mark.parametrize("order", ["network_chain", "network_cluster", "cluster"])
def test_circos_orders_match_heatmap(order):
    G, labels, nets, m = _gradient_graph()
    fig, ax = bnv.circos_plot(G, labels, nets, order=order, order_matrix=m)
    assert _ring_order(ax, labels) == [labels[i] for i in matrix_order(m, nets, order)]
    plt.close(fig)


def test_circos_orders_by_graph_weights_without_matrix():
    G, labels, nets, _ = _gradient_graph()
    fig, ax = bnv.circos_plot(G, labels, nets, order="network_chain")
    adjacency = nx.to_numpy_array(G, nodelist=range(len(labels)))
    assert _ring_order(ax, labels) == [labels[i] for i in matrix_order(adjacency, nets, "network_chain")]
    plt.close(fig)


def test_circos_cluster_gaps_sit_at_the_largest_splits():
    G, labels, nets, m = _gradient_graph(spread=0.9)
    fig, ax = bnv.circos_plot(G, labels, nets, order="cluster", order_matrix=m)
    centres = np.array([p.center for p in ax.patches if isinstance(p, Circle)])
    steps = np.diff(np.unwrap(np.arctan2(centres[:, 1], centres[:, 0])))
    heights = merge_heights(m, matrix_order(m, nets, "cluster"))
    assert np.argmax(steps) == np.argmax(heights)
    assert (steps > steps.min() + 1e-9).sum() <= len(set(nets)) - 1
    plt.close(fig)


def test_circos_split_networks_use_a_legend():
    G, labels, nets, m = _gradient_graph(spread=0.9)
    ordered = [nets[i] for i in matrix_order(m, nets, "cluster")]
    runs = [s for k, s in enumerate(ordered) if k == 0 or s != ordered[k - 1]]
    assert len(runs) > len(set(nets))
    fig, ax = bnv.circos_plot(G, labels, nets, order="cluster", order_matrix=m)
    assert ax.get_legend() is not None
    assert not set(nets) & {t.get_text() for t in ax.texts}
    assert len([p for p in ax.patches if isinstance(p, Wedge)]) == len(runs)
    plt.close(fig)


def test_circos_labels_use_darker_network_colours():
    labels, nets = ["r0", "r1", "r2"], ["A", "A", "B"]
    net2color = {"A": (0.6, 0.9, 0.5), "B": (0.7, 0.8, 0.95)}
    fig, ax = bnv.circos_plot(nx.empty_graph(3), labels, nets, net2color=net2color)
    assert len(ax.texts) == len(labels) + len(net2color)
    for t in ax.texts:
        net = t.get_text() if t.get_text() in net2color else nets[labels.index(t.get_text())]
        ratio = np.array(to_rgb(t.get_color())) / np.array(net2color[net])
        assert ratio.max() < 1 and np.ptp(ratio) < 1e-6, t.get_text()
    plt.close(fig)


def test_circos_nodes_do_not_overlap():
    n = 300
    G = nx.empty_graph(n)
    fig, ax = bnv.circos_plot(G, [f"r{i}" for i in range(n)], ["a"] * n)
    nodes = [p for p in ax.patches if isinstance(p, Circle)]
    spacing = 2 * np.pi / n
    assert nodes and max(c.get_radius() for c in nodes) <= 0.5 * spacing
    plt.close(fig)


def test_circos_network_names_do_not_overlap():
    sizes = {"Alpha": 40, "Bb": 1, "Cc": 1, "Dd": 1, "Epsilon": 40}
    nets = [name for name, k in sizes.items() for _ in range(k)]
    G = nx.empty_graph(len(nets))
    fig, ax = bnv.circos_plot(G, [f"r{i}" for i in range(len(nets))], nets)
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in ax.texts if t.get_text() in sizes]
    assert len(boxes) == len(sizes)
    assert not any(a.overlaps(b) for a, b in itertools.combinations(boxes, 2))
    plt.close(fig)


def test_spring_hulls_are_optional(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.spring_plot(G, labels, nets)
    assert not [p for p in ax.patches if isinstance(p, Polygon)]
    plt.close(fig)

    fig, ax = bnv.spring_plot(G, labels, nets, network_hulls=True)
    hulls = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(hulls) == len(set(nets)) and all(h.get_zorder() < 1 for h in hulls)
    plt.close(fig)


def test_spring_edge_colours(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.spring_plot(G, labels, nets, edge_color="sign")
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    colours = {tuple(np.round(c[:3], 6)) for c in lines[0].get_edgecolor()}
    assert colours == {SIGN[0], SIGN[1]}
    plt.close(fig)


def test_spring_edge_colour_range(signed_graph):
    G, labels, nets = signed_graph
    fig, ax = bnv.spring_plot(G, labels, nets, edge_color="weight", edge_colorvminvmax="minmax")
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    np.testing.assert_allclose(lines[0].get_edgecolor()[:, :3], values_to_colors(w, cmap="RdBu_r", vmin=w.min(), vmax=w.max()))
    plt.close(fig)


def _colour_bars(ax):
    return [c for c in ax.child_axes if hasattr(c, "_colorbar")]


@pytest.mark.parametrize("vminvmax", ["absmax", "minmax", (0.0, 0.1)])
def test_circos_edge_colour_range_and_colour_bar(signed_graph, vminvmax):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets, edge_colorvminvmax=vminvmax)
    w = np.array(sorted((w for _, _, w in G.edges(data="weight")), key=abs))
    peak = np.abs(w).max()
    lo, hi = {"absmax": (-peak, peak), "minmax": (w.min(), w.max())}.get(vminvmax, vminvmax)
    np.testing.assert_allclose([p.get_edgecolor()[:3] for p in _chords(ax)], values_to_colors(w, "RdBu_r", lo, hi))
    bars = _colour_bars(ax)
    assert len(bars) == 1
    np.testing.assert_allclose(bars[0].get_xlim(), (lo, hi))
    plt.close(fig)


@pytest.mark.parametrize("kwargs", [dict(edge_colorbar=False), dict(edge_color="sign"), dict(edge_color="grey")])
def test_circos_colour_bar_only_for_weight_colours(signed_graph, kwargs):
    G, labels, nets = signed_graph
    fig, ax = bnv.circos_plot(G, labels, nets, **kwargs)
    assert not _colour_bars(ax)
    plt.close(fig)
