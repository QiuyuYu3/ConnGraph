"""
Sparsification methods that turn a connectivity matrix into a weighted adjacency matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_PARAMS = {
    "tmfg":        (),
    "full":        (),
    "absolute":    ("threshold",),
    "density":     ("density",),
    "eco":         (),
    "knn":         ("k",),
    "mst":         (),
    "mst_density": ("density",),
    "omst":        (),
    "percolation": (),
}
GRAPH_METHODS = tuple(_PARAMS)
SWEEP_PARAMS = {"absolute": "threshold", "density": "density", "mst_density": "density"}
SIGNS = ("abs", "signed", "positive", "negative")


def resolve_sign(method: str | Callable, sign: str | None) -> str:
    """Return the sign rule; None means "abs" for tmfg and callables, "positive" for the other methods."""
    if sign is None:
        return "abs" if callable(method) or method == "tmfg" else "positive"
    if sign not in SIGNS:
        raise ValueError(f"sign={sign!r} is not recognised. Choose one of {', '.join(SIGNS)}.")
    if sign == "signed" and method == "omst":
        raise ValueError("sign=\"signed\" is not available for graph_method=\"omst\", which needs positive edge lengths.")
    return sign


def apply_sign(W: np.ndarray, sign: str) -> np.ndarray:
    """Drop the weights the sign rule excludes; "negative" turns negative weights into positive magnitudes."""
    if sign == "positive":
        return np.clip(W, 0, None)
    if sign == "negative":
        return np.clip(-W, 0, None)
    return W


def check_graph_method(method: str | Callable, params: dict | None) -> tuple[str | None, np.ndarray | None]:
    """Validate a method and its params; return the swept parameter name and its sorted values, if any."""
    params = params or {}
    if callable(method):
        return None, None
    if method not in _PARAMS:
        raise ValueError(
            f"graph_method={method!r} is not recognised. "
            f"Choose one of {', '.join(GRAPH_METHODS)}, or pass a callable that takes a matrix."
        )
    expected = set(_PARAMS[method])
    if set(params) != expected:
        raise ValueError(
            f"graph_method={method!r} takes graph_params {sorted(expected) or 'none'}, got {sorted(params)}."
        )

    for name, value in params.items():
        if np.ndim(value) == 0:
            _check_value(method, name, value)
            continue
        if name != SWEEP_PARAMS.get(method):
            raise ValueError(f"graph_params[{name!r}] must be a single value for graph_method={method!r}.")
        values = np.unique(np.asarray(value, dtype=float))
        if values.size < 2:
            raise ValueError(f"graph_params[{name!r}] needs at least two distinct values to integrate over.")
        for v in values:
            _check_value(method, name, v)
        return name, values
    return None, None


def _check_value(method: str, name: str, value) -> None:
    if name == "density" and not 0 < value <= 1:
        raise ValueError(f"density must be in (0, 1], got {value}.")
    if name == "k" and not (float(value).is_integer() and value >= 1):
        raise ValueError(f"k must be a positive integer, got {value}.")
    if name == "threshold" and not np.isfinite(value):
        raise ValueError(f"threshold must be finite, got {value}.")


def requested_edges(n: int, method: str | Callable, params: dict) -> int | None:
    """Number of edges a fixed-size method asks for, or None when the method does not fix it."""
    if method in ("density", "mst_density"):
        return _n_edges(n, params["density"])
    if method == "eco":
        return _n_edges(n, min(1.0, 3 / (n - 1)))
    return None


def build_adjacency(
    W: np.ndarray, method: str | Callable, params: dict | None = None, signed: bool = False,
) -> np.ndarray:
    """Return the symmetric adjacency matrix (weights of kept edges); edges are ranked by w if signed, else by |w|."""
    params = params or {}
    if callable(method):
        graph = method(W, **params)
        if isinstance(graph, np.ndarray):
            return np.asarray(graph, dtype=float)
        import networkx as nx
        return nx.to_numpy_array(graph, nodelist=range(len(W)))
    return _BUILDERS[method](W, W if signed else np.abs(W), **params)


def inverse_distances(W: np.ndarray) -> np.ndarray:
    """Inverse shortest-path lengths with edge length 1/w; 0 on the diagonal and between disconnected nodes."""
    from scipy.sparse.csgraph import shortest_path

    dist = shortest_path(_lengths(W), method="D", directed=False)
    return np.divide(1.0, dist, out=np.zeros_like(dist), where=np.isfinite(dist) & (dist > 0))


def _lengths(W: np.ndarray) -> np.ndarray:
    return np.divide(1.0, W, out=np.zeros_like(W), where=W > 0)


def _n_edges(n: int, density: float) -> int:
    # Round half up, as bct.threshold_proportional does
    return int(np.floor(density * n * (n - 1) / 2 + 0.5))


def _keep(W: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Mirror the upper triangle so the result is exactly symmetric even if W is not
    upper = np.triu(np.where(mask & (W != 0), W, 0.0), 1)
    return upper + upper.T


def _keep_strongest(W: np.ndarray, S: np.ndarray, n_edges: int, base: np.ndarray) -> np.ndarray:
    """Keep the edges in `base`, then the highest-scoring remaining edges up to n_edges in total."""
    rows, cols = np.triu_indices(len(W), k=1)
    keep = base[rows, cols].copy()
    candidates = np.flatnonzero(~keep & (W[rows, cols] != 0))
    order = candidates[np.argsort(-S[rows, cols][candidates], kind="stable")]
    keep[order[:max(n_edges - int(keep.sum()), 0)]] = True

    mask = np.zeros(W.shape, dtype=bool)
    mask[rows[keep], cols[keep]] = True
    return _keep(W, mask | mask.T)


def _spanning_mask(S: np.ndarray, exists: np.ndarray) -> np.ndarray:
    """Symmetric mask of the spanning forest that keeps the highest scores among existing edges."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Kruskal depends only on edge order, so rank lengths work for signed scores too
    rows, cols = np.nonzero(np.triu(exists, 1))
    order = np.argsort(-S[rows, cols], kind="stable")
    lengths = np.zeros(S.shape)
    lengths[rows[order], cols[order]] = np.arange(1, order.size + 1)
    tree = minimum_spanning_tree(lengths).toarray() > 0
    return tree | tree.T


def _tmfg(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    import collections
    import collections.abc
    if not hasattr(collections, "Sized"):
        collections.Sized = collections.abc.Sized  # topcorr still uses the alias removed in Python 3.10
    import networkx as nx
    import topcorr as tpc

    graph = tpc.tmfg(S, absolute=False, threshold_mean=True)
    return _keep(W, nx.to_numpy_array(graph, nodelist=range(len(W))) != 0)


def _full(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    return _keep(W, np.ones(W.shape, dtype=bool))


def _absolute(W: np.ndarray, S: np.ndarray, threshold: float) -> np.ndarray:
    import bct
    return _keep(W, bct.threshold_absolute(S, threshold) != 0)


def _density(W: np.ndarray, S: np.ndarray, density: float) -> np.ndarray:
    import bct
    return _keep(W, bct.threshold_proportional(S, density) != 0)


def _eco(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    # Mean degree 3
    return _density(W, S, min(1.0, 3 / (len(W) - 1)))


def _knn(W: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    n = len(W)
    k = int(k)
    if k > n - 1:
        raise ValueError(f"k={k} is larger than the {n - 1} possible neighbours.")
    ranked = np.where(np.eye(n, dtype=bool), -np.inf, S)
    nearest = np.argsort(-ranked, axis=1, kind="stable")[:, :k]
    mask = np.zeros(W.shape, dtype=bool)
    mask[np.repeat(np.arange(n), k), nearest.ravel()] = True
    return _keep(W, mask | mask.T)


def _mst(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    return _keep(W, _spanning_mask(S, W != 0))


def _mst_density(W: np.ndarray, S: np.ndarray, density: float) -> np.ndarray:
    tree = _spanning_mask(S, W != 0)
    n_edges = _n_edges(len(W), density)
    n_tree = int(tree.sum()) // 2
    if n_edges < n_tree:
        raise ValueError(f"density={density} keeps {n_edges} edges, fewer than the {n_tree} of the spanning tree.")
    return _keep_strongest(W, S, n_edges, base=tree)


def _omst(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    # Add orthogonal spanning trees one at a time; keep the set maximising efficiency (relative to the full graph) minus cost
    total = S.sum()
    full_efficiency = inverse_distances(S).sum()
    remaining = S.copy()
    kept = np.zeros(W.shape, dtype=bool)
    best, best_score = kept, -np.inf
    while True:
        tree = _spanning_mask(remaining, remaining > 0)
        if not tree.any():
            break
        kept = kept | tree
        remaining[tree] = 0
        efficiency = inverse_distances(np.where(kept, S, 0.0)).sum() / full_efficiency
        cost = S[kept].sum() / total
        if efficiency - cost > best_score:
            best, best_score = kept, efficiency - cost
        if best_score >= 1 - cost:
            break  # relative efficiency is at most 1 and later trees cost more, so none can score higher
    return _keep(W, best)


def _percolation(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    # The highest cutoff that keeps the largest component whole is the weakest edge of its spanning tree
    from scipy.sparse.csgraph import connected_components

    exists = W != 0
    _, labels = connected_components(exists.astype(float), directed=False)
    giant = labels == np.bincount(labels).argmax()
    tree = _spanning_mask(S, exists & np.outer(giant, giant))
    if not tree.any():
        return np.zeros_like(W)
    return _keep(W, S >= S[tree].min())


_BUILDERS = {
    "tmfg":        _tmfg,
    "full":        _full,
    "absolute":    _absolute,
    "density":     _density,
    "eco":         _eco,
    "knn":         _knn,
    "mst":         _mst,
    "mst_density": _mst_density,
    "omst":        _omst,
    "percolation": _percolation,
}
