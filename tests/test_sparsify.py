import networkx as nx
import numpy as np
import pytest

from brainnet3d.graph_theory.sparsify import GRAPH_METHODS, build_adjacency, inverse_distances, resolve_sign

PARAMS = {"absolute": {"threshold": 0.2}, "density": {"density": 0.1}, "knn": {"k": 3}, "mst_density": {"density": 0.1}}


def _corr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((200, 4))
    corr = np.corrcoef((factors[:, np.arange(n) % 4] + rng.standard_normal((200, n))).T)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 0)
    return corr


def _n_edges(A: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(A, 1)))


def _connected(A: np.ndarray) -> bool:
    return nx.is_connected(nx.from_numpy_array(A))


@pytest.mark.parametrize("method", GRAPH_METHODS)
def test_adjacency_keeps_input_weights_symmetrically(method):
    pytest.importorskip("bct")
    if method == "tmfg":
        pytest.importorskip("topcorr")
    W = _corr(20)
    A = build_adjacency(W, method, PARAMS.get(method))
    kept = A != 0
    assert np.array_equal(A, A.T)
    assert not np.diag(A).any()
    np.testing.assert_array_equal(A[kept], W[kept])


def test_tmfg_is_planar_with_3n_minus_6_edges():
    pytest.importorskip("topcorr")
    A = build_adjacency(_corr(20), "tmfg")
    assert _n_edges(A) == 3 * (20 - 2)
    assert nx.check_planarity(nx.from_numpy_array(A))[0]


def test_absolute_keeps_edges_at_or_above_threshold():
    pytest.importorskip("bct")
    W = _corr(20)
    A = build_adjacency(W, "absolute", {"threshold": 0.2})
    np.testing.assert_array_equal(A != 0, np.abs(W) >= 0.2)


@pytest.mark.parametrize("method, params, expected", [
    ("density", {"density": 0.1}, 19),
    ("eco", {}, 30),
])
def test_density_keeps_the_strongest_edges(method, params, expected):
    pytest.importorskip("bct")
    W = _corr(20)
    A = build_adjacency(W, method, params)
    assert _n_edges(A) == expected
    upper = np.triu(np.ones_like(W, dtype=bool), 1)
    assert np.abs(W[(A != 0) & upper]).min() > np.abs(W[(A == 0) & upper]).max()


def test_knn_keeps_each_nodes_strongest_edges():
    W = _corr(20)
    A = build_adjacency(W, "knn", {"k": 3})
    top = np.argsort(-np.abs(W), axis=1)[:, :3]
    expected = np.zeros_like(W, dtype=bool)
    expected[np.repeat(np.arange(20), 3), top.ravel()] = True
    np.testing.assert_array_equal(A != 0, expected | expected.T)


def test_mst_is_the_maximum_spanning_tree():
    W = _corr(20)
    A = build_adjacency(W, "mst")
    reference = nx.maximum_spanning_tree(nx.from_numpy_array(np.abs(W)))
    assert _n_edges(A) == 19 and _connected(A)
    assert np.abs(A).sum() / 2 == pytest.approx(reference.size(weight="weight"))


def test_mst_density_adds_strongest_edges_to_the_tree():
    W = _corr(20)
    tree = build_adjacency(W, "mst") != 0
    A = build_adjacency(W, "mst_density", {"density": 0.2})
    assert _n_edges(A) == 38
    assert (A[tree] != 0).all()
    with pytest.raises(ValueError, match="spanning tree"):
        build_adjacency(W, "mst_density", {"density": 0.05})


def test_omst_matches_reference_built_with_networkx_and_bct():
    bct = pytest.importorskip("bct")
    W = np.abs(_corr(40, seed=2))
    graph = nx.from_numpy_array(W)
    kept = np.zeros_like(W)
    best, best_score, n_trees = None, -np.inf, 0
    while graph.number_of_edges():
        tree = nx.maximum_spanning_tree(graph)
        graph.remove_edges_from(tree.edges)
        for i, j in tree.edges:
            kept[i, j] = kept[j, i] = W[i, j]
        score = bct.efficiency_wei(kept) / bct.efficiency_wei(W) - kept.sum() / W.sum()
        if score > best_score:
            best, best_score = kept.copy(), score
            n_trees = _n_edges(kept) // 39
    assert n_trees > 1
    np.testing.assert_array_equal(build_adjacency(W, "omst"), best)


def test_omst_stops_before_scoring_every_tree(monkeypatch):
    from brainnet3d.graph_theory import sparsify

    W = np.abs(_corr(60, seed=3))
    remaining, n_trees = W.copy(), 0
    while (tree := sparsify._spanning_mask(remaining, remaining > 0)).any():
        remaining[tree] = 0
        n_trees += 1

    calls = []
    real = sparsify.inverse_distances
    monkeypatch.setattr(sparsify, "inverse_distances", lambda X: calls.append(1) or real(X))
    build_adjacency(W, "omst")
    assert len(calls) - 1 < n_trees // 2  # one call is for the full graph


def test_percolation_keeps_the_highest_cutoff_that_stays_connected():
    W = _corr(20)
    A = build_adjacency(W, "percolation")
    cutoff = np.abs(A[A != 0]).min()
    np.testing.assert_array_equal(A != 0, np.abs(W) >= cutoff)
    assert _connected(A)
    assert not _connected(np.where(np.abs(W) > cutoff, W, 0.0))


def test_inverse_distances_give_bct_global_efficiency():
    bct = pytest.importorskip("bct")
    W = np.abs(build_adjacency(_corr(20), "knn", {"k": 2}))
    assert inverse_distances(W).sum() / (20 * 19) == pytest.approx(bct.efficiency_wei(W))


def test_resolve_sign_defaults_by_method():
    assert resolve_sign("tmfg", None) == "abs"
    assert resolve_sign(lambda m: m, None) == "abs"
    assert resolve_sign("density", None) == "positive"
    assert resolve_sign("density", "abs") == "abs"
    with pytest.raises(ValueError, match="sign"):
        resolve_sign("density", "both")
    with pytest.raises(ValueError, match="omst"):
        resolve_sign("omst", "signed")


def test_signed_density_matches_bct_on_the_raw_matrix():
    bct = pytest.importorskip("bct")
    W = _corr(30)
    W = W - W[np.triu_indices(30, 1)].mean()  # many negative weights
    np.fill_diagonal(W, 0)
    for density in (0.2, 0.6):
        np.testing.assert_array_equal(
            build_adjacency(W, "density", {"density": density}, signed=True),
            bct.threshold_proportional(W, density),
        )
    assert (build_adjacency(W, "density", {"density": 0.6}, signed=True) < 0).any()


def test_signed_tmfg_matches_topcorr_without_absolute():
    pytest.importorskip("topcorr")
    import topcorr as tpc

    W = _corr(20)
    W = W - W[np.triu_indices(20, 1)].mean()
    np.fill_diagonal(W, 0)
    expected = nx.to_numpy_array(tpc.tmfg(W, absolute=False, threshold_mean=True), nodelist=range(20))
    np.testing.assert_array_equal(build_adjacency(W, "tmfg", signed=True), expected)


def test_signed_mst_is_the_maximum_spanning_tree_of_signed_weights():
    W = _corr(20)
    W = W - W[np.triu_indices(20, 1)].mean()
    np.fill_diagonal(W, 0)
    reference = nx.maximum_spanning_tree(nx.from_numpy_array(W))
    np.testing.assert_array_equal(build_adjacency(W, "mst", signed=True) != 0, nx.to_numpy_array(reference) != 0)
