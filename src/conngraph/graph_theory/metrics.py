"""
Single-subject graph-theory metrics on a sparsified connectivity matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from conngraph.graph_theory.randomize import randomize_signed
from conngraph.graph_theory.sparsify import (
    apply_sign,
    build_adjacency,
    check_graph_method,
    inverse_distances,
    requested_edges,
    resolve_sign,
)

METRIC_NAMES = ("clust_coeff", "btwn_cent", "strength", "ge_local", "eff_nodal", "participation", "module_z",
                "eig_cent", "close_cent")
# The first variant of each metric is its default
METRIC_VARIANTS = {
    "clust_coeff":   ("costantini", "zhang", "onnela", "bin"),
    "btwn_cent":     ("inv", "inv_norm", "log", "log_norm", "bin"),
    "strength":      ("abs", "pos", "neg", "bin"),
    "ge_local":      ("wang", "rubinov", "bin"),
    "eff_nodal":     ("wei", "bin"),
    "participation": ("pos", "neg"),
    "module_z":      ("abs", "bin"),
    "eig_cent":      ("wei", "bin"),
    "close_cent":    ("wei", "bin"),
}
# One value per graph
GLOBAL_VARIANTS = {
    "eff_global":  ("wei", "bin"),
    "char_path":   ("wei", "bin"),
    "clust_mean":  ("costantini", "zhang", "onnela", "bin"),
    "modularity":  ("wei",),
    "small_world": ("wei", "bin"),
}
PARTITION_METRICS = ("participation", "module_z", "modularity")
PARTITIONS = ("networks", "louvain")
SIGNED_FALLBACKS = ("abs", "positive")
LOUVAIN_RUNS = 100
_VARIANTS = {**METRIC_VARIANTS, **GLOBAL_VARIANTS}
# Clustering and path length whose ratios to random networks give each small-world index
_SMALL_WORLD = {"small_world.wei": ("clust_mean.onnela", "char_path.wei"),
                "small_world.bin": ("clust_mean.bin", "char_path.bin")}
_SUMMARIES = ("auc", "mean")


def is_global(name: str) -> bool:
    return name.split(".")[0] in GLOBAL_VARIANTS


def parse_metrics(metrics: str | list[str] | None) -> list[str]:
    """Expand metric specs to "metric.variant" names; a bare metric name means its default variant."""
    if metrics is None:
        return [f"{m}.{variants[0]}" for m, variants in _VARIANTS.items()]
    if isinstance(metrics, str) and metrics == "all":
        return [f"{m}.{v}" for m, variants in _VARIANTS.items() for v in variants]
    if isinstance(metrics, str):
        metrics = [metrics]

    names: list[str] = []
    for spec in metrics:
        metric, _, rest = spec.partition(".")
        variant, _, partition = rest.partition(".")
        if (metric not in _VARIANTS or (variant and variant not in _VARIANTS[metric])
                or (partition and (metric not in PARTITION_METRICS or partition not in PARTITIONS))):
            known = ", ".join(f"{m}.{v}" for m, vs in _VARIANTS.items() for v in vs)
            raise ValueError(f"Metric {spec!r} is not recognised. Choose from: {known}.")
        name = ".".join(x for x in (metric, variant or _VARIANTS[metric][0], partition) if x)
        if name not in names:
            names.append(name)
    return names


def check_options(
    metrics: str | list[str] | None,
    graph_method: str | Callable,
    graph_params: dict | None,
    sign: str | None,
    summary: str,
    n_random: int = 0,
    random_swaps: float = 10,
    signed_fallback: str = "abs",
    partitions: tuple[str, ...] = PARTITIONS,
    level: str = "node",
) -> list[str]:
    """Validate the options and return the metric names, one per partition for partition metrics.

    The defaults and "all" silently drop metrics the options rule out; named metrics raise instead.
    """
    implicit = metrics is None or metrics == "all"
    names = parse_metrics(metrics)
    check_graph_method(graph_method, graph_params)
    if summary not in _SUMMARIES:
        raise ValueError(f"summary={summary!r} is not recognised. Choose \"auc\" or \"mean\".")
    if not (float(n_random).is_integer() and n_random >= 0):
        raise ValueError(f"n_random must be a non-negative integer, got {n_random}.")
    if n_random and not random_swaps > 0:
        raise ValueError(f"random_swaps must be positive, got {random_swaps}.")
    resolved = resolve_sign(graph_method, sign)
    if signed_fallback not in SIGNED_FALLBACKS:
        raise ValueError(f"signed_fallback={signed_fallback!r} is not recognised. Choose \"abs\" or \"positive\".")
    if signed_fallback != "abs" and resolved in ("positive", "negative"):
        raise ValueError(f"signed_fallback=\"{signed_fallback}\" has no effect: sign=\"{resolved}\" leaves no "
                         "negative weights.")
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f"partitions={partitions!r} must name one or both of \"networks\" and \"louvain\".")

    if level != "node":
        kept = [m for m in names if m.split(".")[0] not in PARTITION_METRICS]
        if not kept:
            raise ValueError(f"{', '.join(names)} need modules of nodes, so only the node level has them.")
        names = kept
    names = [p for m in names for p in (
        [f"{m}.{part}" for part in partitions] if m.split(".")[0] in PARTITION_METRICS and m.count(".") == 1 else [m])]

    constant = {}
    if graph_method == "mst":
        for m in names:
            if m.split(".")[0] in ("clust_coeff", "ge_local", "clust_mean", "small_world"):
                constant[m] = "is always 0 on a spanning tree (graph_method=\"mst\")"
    if resolved in ("positive", "negative"):
        for m in ("strength.neg", *(n for n in names if n.startswith("participation.neg."))):
            if m in names:
                constant[m] = f"is always 0 when only one sign of weights is kept (sign=\"{resolved}\")"
    if not n_random:
        for m in names:
            if m.startswith("small_world."):
                constant[m] = "compares the graph with random networks, so it needs n_random"

    if constant and implicit:
        return [m for m in names if m not in constant]
    if constant:
        raise ValueError("; ".join(f"{m} {why}" for m, why in constant.items()))
    return names


def process_subject(
    subj: str,
    corrmat: np.ndarray,
    node_labels: list,
    metrics: str | list[str] | None = None,
    graph_method: str | Callable = "tmfg",
    graph_params: dict | None = None,
    sign: str | None = None,
    summary: str = "auc",
    return_curves: bool = False,
    normalize_weights: bool = False,
    n_random: int = 0,
    random_swaps: float = 10,
    random_seed=None,
    signed_fallback: str = "abs",
    partitions: tuple[str, ...] = PARTITIONS,
    modules=None,
) -> dict:
    """Return {"subj", "metric.variant": {label: value} or one number, ["curves"], ["shortfall"]} for one subject.

    modules gives each node's module for the "networks" partition, NaN for nodes left out of it; other options
    as in compute_graph_metrics.
    """
    names = check_options(metrics, graph_method, graph_params, sign, summary, n_random, random_swaps,
                          signed_fallback, partitions)
    params = dict(graph_params or {})
    sweep, values = check_graph_method(graph_method, params)

    mat = np.array(corrmat, dtype=float)
    np.fill_diagonal(mat, 0)
    resolved = resolve_sign(graph_method, sign)
    mat = apply_sign(mat, resolved)
    if normalize_weights and np.abs(mat).max() > 0:
        import bct
        mat = bct.weight_conversion(mat, "normalize")
    signed = resolved == "signed"
    settings = [params] if sweep is None else [{**params, sweep: v} for v in values]

    rng = np.random.default_rng(random_seed)
    curves = None
    shared = dict(n_random=n_random, swaps=random_swaps, rng=rng, signed_fallback=signed_fallback, modules=modules)
    if sweep is None:
        A = build_adjacency(mat, graph_method, params, signed)
        results = _node_values(names, A, **shared)
    else:
        stacked: dict[str, list] = {}
        for setting in settings:
            A = build_adjacency(mat, graph_method, setting, signed)
            for m, v in _node_values(names, A, **shared).items():
                stacked.setdefault(m, []).append(v)
        curves = {m: np.vstack(rows) for m, rows in stacked.items()}
        results = {m: _summarize(curve, values, summary) for m, curve in curves.items()}

    out: dict = {"subj": subj}
    for m, vals in results.items():
        out[m] = float(vals[0]) if is_global(m) else dict(zip(node_labels, vals))
    if return_curves and curves is not None:
        out["curves"] = _curves_frame(curves, values, node_labels)
    shortfall = _shortfall(mat, graph_method, settings)
    if shortfall:
        out["shortfall"] = shortfall
    return out


def normalized_metrics(names: list[str], n_random: int) -> list[str]:
    """Metrics that also get a ".norm" ratio to random networks.

    Randomizing keeps degree, so strength.bin never does; partition metrics and the small-world index do not either.
    """
    if not n_random:
        return []
    return [m for m in names if m != "strength.bin" and m.split(".")[0] not in (*PARTITION_METRICS, "small_world")]


def output_names(names: list[str], n_random: int) -> list[str]:
    return names + [f"{m}.norm" for m in normalized_metrics(names, n_random)]


def _node_values(
    names: list[str], A: np.ndarray, n_random: int, swaps: float, rng: np.random.Generator,
    signed_fallback: str = "abs", modules=None,
) -> dict[str, np.ndarray]:
    """Values of each metric on A, an array of one value for a global metric, plus the ratios to random networks."""
    needed = list(dict.fromkeys([m for m in names if m not in _SMALL_WORLD]
                                + [x for m in names if m in _SMALL_WORLD for x in _SMALL_WORLD[m]]))
    louvain = any(m.endswith(".louvain") for m in needed)
    graph = _Graph(A, signed_fallback, modules, int(rng.integers(2**31 - 1)) if louvain else None)
    values = {m: graph.value(m) for m in needed}
    norm = normalized_metrics(needed, n_random)
    if norm:
        sums = {m: np.zeros_like(values[m]) for m in norm}
        for _ in range(n_random):
            random_graph = _Graph(randomize_signed(A, swaps, rng), signed_fallback)
            for m in norm:
                sums[m] += random_graph.value(m)
        for m in norm:
            # NaN where the random networks give 0, e.g. clustering of a node with one neighbour
            values[f"{m}.norm"] = np.divide(values[m], sums[m] / n_random, out=np.full_like(values[m], np.nan),
                                            where=sums[m] != 0)
    for m in names:
        if m in _SMALL_WORLD:
            clustering, path = (values[f"{x}.norm"] for x in _SMALL_WORLD[m])
            values[m] = np.divide(clustering, path, out=np.full_like(path, np.nan), where=path != 0)
    keep = set(output_names(names, n_random))
    return {m: v for m, v in values.items() if m in keep}


def _shortfall(mat: np.ndarray, method: str | Callable, settings: list[dict]) -> str | None:
    """Describe settings that ask for more edges than the matrix has after the sign rule, if any."""
    if callable(method):
        return None
    available = int(np.count_nonzero(np.triu(mat, 1)))
    short = [s for s in settings if (requested_edges(len(mat), method, s) or 0) > available]
    if not short:
        return None
    asked = ", ".join(f"{requested_edges(len(mat), method, s)} ({', '.join(f'{k}={v:g}' for k, v in s.items()) or method})"
                      for s in short)
    return f"{available} edges available, asked for {asked}"


def compute_metric(name: str, A: np.ndarray, signed_fallback: str = "abs", modules=None, seed=None) -> np.ndarray:
    """Values of one metric name on a signed adjacency matrix with a zero diagonal: one per node, or one in all.

    signed_fallback: how metrics without a signed form read negative weights, as |w| ("abs") or dropped
    ("positive"). modules: each node's module for the "networks" partition, NaN for nodes left out of it.
    seed: for the Louvain partition.
    """
    return _Graph(A, signed_fallback, modules, seed).value(name)


def modularity_q(A: np.ndarray, modules) -> float:
    """Modularity of a partition; negative weights subtract their own modularity, scaled by their share of the weight."""
    same = np.equal.outer(np.asarray(modules), np.asarray(modules))
    q, total = 0.0, np.abs(A).sum()
    for part, scale in ((np.clip(A, 0, None), 1.0), (np.clip(-A, 0, None), -1.0)):
        v = part.sum()
        if v:
            k = part.sum(axis=1)
            q += scale * ((part - np.outer(k, k) / v) * same).sum() / (v if scale > 0 else total)
    return float(q)


class _Graph:
    """One adjacency matrix and the shortest paths and partitions that several metrics share."""

    def __init__(self, A: np.ndarray, signed_fallback: str = "abs", modules=None, seed=None):
        self.A = A
        self.W = np.abs(A) if signed_fallback == "abs" else np.clip(A, 0, None)
        self.B = (self.W != 0).astype(float)
        self.modules = None if modules is None else np.asarray(modules, dtype=object)
        self.seed = seed
        self._cache: dict = {}

    def value(self, name: str) -> np.ndarray:
        import bct

        metric, variant, *partition = name.split(".")
        A, W, B = self.A, self.W, self.B
        M = B if variant == "bin" else W
        if metric in PARTITION_METRICS:
            return self._partition_metric(metric, variant, partition[0])
        funcs = {
            "clust_coeff.costantini": lambda: _clustering_costantini(A),
            "clust_coeff.zhang":      lambda: bct.clustering_coef_wu_sign(W, coef_type="zhang")[0],
            "clust_coeff.onnela":     lambda: bct.clustering_coef_wu(W),
            "clust_coeff.bin":        lambda: bct.clustering_coef_bu(B),
            "btwn_cent.inv":          lambda: bct.betweenness_wei(bct.invert(W)),
            "btwn_cent.inv_norm":     lambda: bct.betweenness_wei(bct.invert(W)) / _n_pairs(A),
            "btwn_cent.log":          lambda: bct.betweenness_wei(_log_lengths(W)),
            "btwn_cent.log_norm":     lambda: bct.betweenness_wei(_log_lengths(W)) / _n_pairs(A),
            "btwn_cent.bin":          lambda: bct.betweenness_bin(B),
            "strength.abs":           lambda: np.abs(A).sum(axis=1),
            "strength.pos":           lambda: np.clip(A, 0, None).sum(axis=1),
            "strength.neg":           lambda: -np.clip(A, None, 0).sum(axis=1),
            "strength.bin":           lambda: B.sum(axis=1),
            "ge_local.wang":          lambda: bct.efficiency_wei(W, local=True),
            "ge_local.rubinov":       lambda: _local_efficiency_rubinov(W),
            "ge_local.bin":           lambda: bct.efficiency_bin(B, local=True),
            "eff_nodal":              lambda: self._inverse_distances(variant).sum(axis=1) / max(len(A) - 1, 1),
            "close_cent":             lambda: self._closeness(variant),
            "eig_cent":               lambda: bct.eigenvector_centrality_und(M),
            "eff_global":             lambda: [bct.charpath(self._distances(variant))[1]],
            "char_path":              lambda: [_finite_mean_path(self._distances(variant))],
            "clust_mean":             lambda: [self.value(f"clust_coeff.{variant}").mean()],
        }
        values = np.asarray((funcs.get(name) or funcs[metric])(), dtype=float)
        if name.startswith("clust_coeff."):
            values = np.nan_to_num(values, nan=0.0)  # nodes with fewer than two neighbours
        return values

    def _distances(self, variant: str) -> np.ndarray:
        """Shortest-path lengths, with edge length 1/w ("wei") or 1 ("bin"); inf between disconnected nodes."""
        if variant not in self._cache:
            from scipy.sparse.csgraph import shortest_path

            lengths = np.divide(1.0, self.W, out=np.zeros_like(self.W), where=self.W > 0)
            self._cache[variant] = shortest_path(lengths if variant == "wei" else self.B, method="D", directed=False,
                                                 unweighted=variant == "bin")
        return self._cache[variant]

    def _inverse_distances(self, variant: str) -> np.ndarray:
        D = self._distances(variant)
        return np.divide(1.0, D, out=np.zeros_like(D), where=np.isfinite(D) & (D > 0))

    def _closeness(self, variant: str) -> np.ndarray:
        # 1 / mean distance to the nodes it reaches; 0 for a node that reaches none
        D = self._distances(variant)
        reached = np.isfinite(D) & (D > 0)
        total = np.where(reached, D, 0).sum(axis=1)
        return np.divide(reached.sum(axis=1), total, out=np.zeros(len(D)), where=total > 0)

    def _partition(self, which: str) -> np.ndarray:
        """Module numbers from 1; nodes left out of the networks partition share one extra module."""
        if which in self._cache:
            return self._cache[which]
        if which == "networks":
            if self.modules is None:
                raise ValueError("The \"networks\" partition needs each node's module.")
            codes, _ = pd.factorize(pd.Series(self.modules))
            ci = np.where(codes < 0, codes.max() + 1, codes) + 1
        else:
            ci = self._louvain()
        self._cache[which] = ci
        return ci

    def _louvain(self) -> np.ndarray:
        import bct

        if not self.A.any():
            return np.ones(len(self.A), dtype=int)
        kind = "negative_asym" if (self.A < 0).any() else "modularity"
        best, best_q = None, -np.inf
        for seed in np.random.default_rng(self.seed).integers(2**31 - 1, size=LOUVAIN_RUNS):
            ci, q = bct.community_louvain(self.A, B=kind, seed=int(seed))
            if q > best_q:
                best, best_q = ci, q
        return best

    def _partition_metric(self, metric: str, variant: str, which: str) -> np.ndarray:
        import bct

        ci = self._partition(which)
        if metric == "modularity":
            return np.array([modularity_q(self.A, ci)])
        if metric == "participation":
            values = bct.participation_coef_sign(self.A, ci)[0 if variant == "pos" else 1]
        else:
            M = self.B if variant == "bin" else self.W
            sizes = np.bincount(ci)[ci]
            with np.errstate(divide="ignore", invalid="ignore"):
                # Sample rather than population standard deviation within each module
                values = bct.module_degree_zscore(M, ci, flag=0) * np.sqrt((sizes - 1) / sizes)
            values = np.where(sizes > 1, values, np.nan)
        values = np.asarray(values, dtype=float)
        if which == "networks" and self.modules is not None:
            values[pd.isna(pd.Series(self.modules)).to_numpy()] = np.nan
        return values


def _finite_mean_path(D: np.ndarray) -> float:
    import bct

    if not np.isfinite(D[~np.eye(len(D), dtype=bool)]).any():
        return np.nan
    return float(bct.charpath(D, include_infinite=False)[0])


def _clustering_costantini(A: np.ndarray) -> np.ndarray:
    # Closed form of bct's triple loop: signed triangles over the |w| products of neighbour pairs
    W = np.abs(A)
    before = np.zeros_like(W)
    before[:, 1:] = np.cumsum(W[:, :-1], axis=1)
    pairs = 2 * (W * before).sum(axis=1)  # avoids the cancellation of sum(|w|)**2 - sum(w**2)
    triangles = ((A @ A) * A).sum(axis=1)
    return np.divide(triangles, pairs, out=np.zeros(len(A)), where=triangles != 0)


def _n_pairs(A: np.ndarray) -> int:
    n = len(A)
    return max((n - 1) * (n - 2), 1)


def _log_lengths(W: np.ndarray) -> np.ndarray:
    if (W > 1).any():
        raise ValueError("btwn_cent.log needs weights in [0, 1].")
    L = np.zeros_like(W)
    nz = W > 0
    L[nz] = -np.log(np.minimum(W[nz], 1 - 1e-12))  # a weight of 1 would give a zero length, read as no edge
    return L


def _local_efficiency_rubinov(W: np.ndarray) -> np.ndarray:
    # Neighbour pairs weighted by cbrt(w_uj * w_uh / d_jh), with paths restricted to the neighbourhood
    n = len(W)
    E = np.zeros(n)
    cbrt_W = np.cbrt(W)
    for u in range(n):
        V = np.flatnonzero(W[u])
        k = V.size
        if k < 2:
            continue
        c = cbrt_W[u, V]
        E[u] = (np.outer(c, c) * np.cbrt(inverse_distances(W[np.ix_(V, V)]))).sum() / (k * (k - 1))
    return E


def _summarize(curve: np.ndarray, x: np.ndarray, summary: str) -> np.ndarray:
    auc = ((curve[1:] + curve[:-1]) / 2 * np.diff(x)[:, None]).sum(axis=0)
    return auc if summary == "auc" else auc / (x[-1] - x[0])


def _curves_frame(curves: dict[str, np.ndarray], x: np.ndarray, labels: list) -> pd.DataFrame:
    frames = []
    for m, curve in curves.items():
        names = ["global"] if is_global(m) else labels
        frames.append(pd.DataFrame({
            "metric": m,
            "node": np.tile(np.asarray(names, dtype=object), len(x)),
            "threshold": np.repeat(x, len(names)),
            "value": curve.ravel(),
        }))
    return pd.concat(frames, ignore_index=True)
