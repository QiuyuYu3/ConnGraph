import networkx as nx
import numpy as np
import pytest

from brainnet3d.graph_theory.graph_utils import density_graph


def _random_corr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ts = rng.standard_normal((300, n)) + rng.standard_normal((300, 1))
    corr = np.corrcoef(ts.T)
    return (corr + corr.T) / 2


@pytest.mark.parametrize("n", [10, 60])
@pytest.mark.parametrize("density", [0.01, 0.05, 0.3, 1.0])
def test_density_graph_keeps_strongest_pairs(n, density):
    corr = _random_corr(n)
    G = density_graph(corr, density)

    n_pairs = n * (n - 1) // 2
    assert G.number_of_nodes() == n
    assert G.number_of_edges() == round(density * n_pairs)
    assert nx.number_of_selfloops(G) == 0

    kept = {tuple(sorted(e)) for e in G.edges()}
    rows, cols = np.triu_indices(n, k=1)
    kept_w = [corr[i, j] for i, j in zip(rows, cols) if (i, j) in kept]
    dropped_w = [corr[i, j] for i, j in zip(rows, cols) if (i, j) not in kept]
    if kept_w and dropped_w:
        assert min(kept_w) >= max(dropped_w)

    for u, v, w in G.edges(data="weight"):
        assert w == corr[u, v]


@pytest.mark.parametrize("density", [0, -0.1, 1.5])
def test_density_graph_rejects_invalid_density(density):
    with pytest.raises(ValueError):
        density_graph(_random_corr(10), density)

