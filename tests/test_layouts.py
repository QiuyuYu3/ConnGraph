import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.patches import Polygon

import conngraph as bnv
from conngraph.viz.layouts import grouped_layout


def _three_groups(seed=0):
    rng = np.random.default_rng(seed)
    groups = ["a"] * 12 + ["b"] * 10 + ["c"] * 8
    G = nx.empty_graph(len(groups))
    for i, j in itertools.combinations(range(len(groups)), 2):
        same = groups[i] == groups[j]
        if (same and rng.random() < 0.4) or ({groups[i], groups[j]} == {"a", "b"} and rng.random() < 0.15):
            G.add_edge(i, j, weight=rng.uniform(0.5, 0.9))
    return G, groups


def _centres_and_spreads(pos, groups):
    out = {}
    for g in dict.fromkeys(groups):
        pts = np.array([pos[i] for i, s in enumerate(groups) if s == g])
        centre = pts.mean(axis=0)
        out[g] = (centre, np.linalg.norm(pts - centre, axis=1).max())
    return out


@pytest.mark.parametrize("dim", [2, 3])
def test_grouped_layout_keeps_groups_apart(dim):
    G, groups = _three_groups()
    pos = grouped_layout(G, groups, dim=dim)
    assert all(len(p) == dim for p in pos.values())
    regions = _centres_and_spreads(pos, groups)
    for (c1, s1), (c2, s2) in itertools.combinations(regions.values(), 2):
        assert np.linalg.norm(c1 - c2) > s1 + s2


def test_grouped_layout_is_reproducible():
    G, groups = _three_groups()
    a, b = grouped_layout(G, groups, seed=3), grouped_layout(G, groups, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_connected_groups_sit_closer():
    G, groups = _three_groups()
    regions = _centres_and_spreads(grouped_layout(G, groups), groups)
    gap = lambda x, y: np.linalg.norm(regions[x][0] - regions[y][0]) - regions[x][1] - regions[y][1]
    assert gap("a", "b") < gap("a", "c")


def test_group_without_edges_fills_its_region():
    G = nx.empty_graph(40)
    pos = grouped_layout(G, ["only"] * 40)
    pts = np.array(list(pos.values()))
    radii = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    assert radii.min() < 0.3 * radii.max()


def test_grouped_layout_checks_group_count():
    with pytest.raises(ValueError, match="groups"):
        grouped_layout(nx.empty_graph(5), ["a", "b"])


def test_spring_plot_network_layout_draws_hulls_by_default():
    G, groups = _three_groups()
    labels = [f"r{i}" for i in range(len(groups))]
    fig, ax = bnv.spring_plot(G, labels, groups, layout="network")
    assert len([p for p in ax.patches if isinstance(p, Polygon)]) == 3
    plt.close(fig)
    fig, ax = bnv.spring_plot(G, labels, groups, layout="network", network_hulls=False)
    assert not [p for p in ax.patches if isinstance(p, Polygon)]
    plt.close(fig)
    with pytest.raises(ValueError, match="layout"):
        bnv.spring_plot(G, labels, groups, layout="circle")


def test_spring_plot_3d_network_layout():
    G, groups = _three_groups()
    img = bnv.spring_plot_3d(G, network_labels=groups, layout="network")
    assert isinstance(img, np.ndarray) and np.ptp(img) > 0
    with pytest.raises(ValueError, match="network_labels"):
        bnv.spring_plot_3d(G, layout="network")


def test_plot_network_layout_groups_nodes(dataset):
    plotter = bnv.BrainNetPlotter(dataset, subject_id="mean")
    matrix = dataset.mean_matrix().loc[dataset.nodes_df["label"], dataset.nodes_df["label"]].to_numpy()
    groups = dataset.nodes_df["network"].tolist()
    coords = plotter._compute_layout(matrix, "network", 0.4, 42, groups=groups)
    regions = _centres_and_spreads(dict(enumerate(coords)), groups)
    for (c1, s1), (c2, s2) in itertools.combinations(regions.values(), 2):
        assert np.linalg.norm(c1 - c2) > s1 + s2
    assert np.isfinite(bnv.BrainNetPlotter(dataset, subject_id="mean").plot(layout="network")).all()


def test_spring_plot_3d_colours_by_network_without_palette(monkeypatch):
    import conngraph.viz.views as views
    from conngraph.viz.colormap import labels_to_colors

    captured = {}
    monkeypatch.setattr(views, "_finish_render", lambda vp, actors, *a, **kw: captured.update(actors=actors))
    G, groups = _three_groups()
    bnv.spring_plot_3d(G, network_labels=groups, layout="network")
    spheres = captured["actors"][: len(groups)]
    np.testing.assert_allclose([s.color() for s in spheres], labels_to_colors(groups), atol=1e-6)
