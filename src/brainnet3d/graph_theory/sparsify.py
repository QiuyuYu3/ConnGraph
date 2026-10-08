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


def resolve_sign(method: str | Callable, sign: str | None) -> str:
    """Return the sign rule; None means "abs" for tmfg and callables, "positive" for the other methods."""
    if sign is None:
        return "abs" if callable(method) or method == "tmfg" else "positive"
    if sign not in ("abs", "positive"):
        raise ValueError(f"sign={sign!r} is not recognised. Choose \"abs\" or \"positive\".")
    return sign


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


def build_adjacency(W: np.ndarray, method: str | Callable, params: dict | None = None) -> np.ndarray:
    """Return the symmetric adjacency matrix (signed weights of kept edges) for a zero-diagonal matrix."""
    params = params or {}
    if callable(method):
        graph = method(W, **params)
        if isinstance(graph, np.ndarray):
            return np.asarray(graph, dtype=float)
        import networkx as nx
        return nx.to_numpy_array(graph, nodelist=range(len(W)))
    return _BUILDERS[method](W, **params)


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


def _keep_strongest(W: np.ndarray, n_edges: int, base: np.ndarray | None = None) -> np.ndarray:
    """Keep the edges in `base`, then the strongest remaining edges by |w| up to n_edges in total."""
    n = len(W)
    rows, cols = np.triu_indices(n, k=1)
    weights = np.abs(W[rows, cols])
    keep = np.zeros(weights.size, dtype=bool) if base is None else base[rows, cols].copy()

    candidates = np.flatnonzero(~keep & (weights > 0))
    order = candidates[np.argsort(-weights[candidates], kind="stable")]
    keep[order[:max(n_edges - int(keep.sum()), 0)]] = True

    out = np.zeros_like(W)
    out[rows[keep], cols[keep]] = W[rows[keep], cols[keep]]
    return out + out.T


def _spanning_mask(absW: np.ndarray) -> np.ndarray:
    """Symmetric mask of the maximum spanning forest of |w| (the minimum spanning forest of 1/|w|)."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    tree = minimum_spanning_tree(_lengths(absW)).toarray() > 0
    return tree | tree.T


def _tmfg(W: np.ndarray) -> np.ndarray:
    import collections
    import collections.abc
    if not hasattr(collections, "Sized"):
        collections.Sized = collections.abc.Sized  # topcorr still uses the alias removed in Python 3.10
    import networkx as nx
    import topcorr as tpc

    return nx.to_numpy_array(tpc.tmfg(W, absolute=True, threshold_mean=True), nodelist=range(len(W)))


def _full(W: np.ndarray) -> np.ndarray:
    return W.copy()


def _absolute(W: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.abs(W) > threshold, W, 0.0)


def _density(W: np.ndarray, density: float) -> np.ndarray:
    return _keep_strongest(W, _n_edges(len(W), density))


def _eco(W: np.ndarray) -> np.ndarray:
    # Mean degree 3
    n = len(W)
    return _keep_strongest(W, _n_edges(n, min(1.0, 3 / (n - 1))))


def _knn(W: np.ndarray, k: int) -> np.ndarray:
    n = len(W)
    k = int(k)
    if k > n - 1:
        raise ValueError(f"k={k} is larger than the {n - 1} possible neighbours.")
    absW = np.abs(W)
    nearest = np.argsort(-absW, axis=1, kind="stable")[:, :k]
    mask = np.zeros(W.shape, dtype=bool)
    mask[np.repeat(np.arange(n), k), nearest.ravel()] = True
    mask = (mask | mask.T) & (absW > 0)
    return np.where(mask, W, 0.0)


def _mst(W: np.ndarray) -> np.ndarray:
    return np.where(_spanning_mask(np.abs(W)), W, 0.0)


def _mst_density(W: np.ndarray, density: float) -> np.ndarray:
    tree = _spanning_mask(np.abs(W))
    n_edges = _n_edges(len(W), density)
    n_tree = int(tree.sum()) // 2
    if n_edges < n_tree:
        raise ValueError(f"density={density} keeps {n_edges} edges, fewer than the {n_tree} of the spanning tree.")
    return _keep_strongest(W, n_edges, base=tree)


def _omst(W: np.ndarray) -> np.ndarray:
    # Add orthogonal spanning trees one at a time and keep the set that maximises global efficiency minus cost
    absW = np.abs(W)
    total = absW.sum()
    n = len(W)
    remaining = absW.copy()
    kept = np.zeros(W.shape, dtype=bool)
    best, best_score = kept, -np.inf
    while True:
        tree = _spanning_mask(remaining)
        if not tree.any():
            break
        kept = kept | tree
        remaining[tree] = 0
        efficiency = inverse_distances(np.where(kept, absW, 0.0)).sum() / (n * (n - 1))
        score = efficiency - absW[kept].sum() / total
        if score > best_score:
            best, best_score = kept, score
    return np.where(best, W, 0.0)


def _percolation(W: np.ndarray) -> np.ndarray:
    # The highest cutoff that keeps the largest component whole is the weakest edge of its maximum spanning tree
    from scipy.sparse.csgraph import connected_components

    absW = np.abs(W)
    _, labels = connected_components((absW > 0).astype(float), directed=False)
    giant = labels == np.bincount(labels).argmax()
    tree = _spanning_mask(np.where(np.outer(giant, giant), absW, 0.0))
    if not tree.any():
        return np.zeros_like(W)
    return np.where(absW >= absW[tree].min(), W, 0.0)


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
