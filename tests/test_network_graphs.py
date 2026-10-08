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
from brainnet3d.viz.colormap import labels_to_colors

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
