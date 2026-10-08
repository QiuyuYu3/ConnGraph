"""
Single-subject graph-theory metrics on a sparsified connectivity matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from brainnet3d.graph_theory.randomize import randomize_signed
from brainnet3d.graph_theory.sparsify import (
    apply_sign,
    build_adjacency,
    check_graph_method,
    inverse_distances,
    requested_edges,
    resolve_sign,
)

METRIC_NAMES = ("clust_coeff", "btwn_cent", "strength", "ge_local")
# The first variant of each metric is its default
METRIC_VARIANTS = {
    "clust_coeff": ("costantini", "zhang", "onnela", "bin"),
    "btwn_cent":   ("inv", "inv_norm", "log", "log_norm", "bin"),
    "strength":    ("abs", "pos", "neg", "bin"),
    "ge_local":    ("wang", "rubinov", "bin"),
}
_SUMMARIES = ("auc", "mean")


def parse_metrics(metrics: str | list[str] | None) -> list[str]:
    """Expand metric specs to "metric.variant" names; a bare metric name means its default variant."""
    if metrics is None:
        return [f"{m}.{variants[0]}" for m, variants in METRIC_VARIANTS.items()]
    if isinstance(metrics, str) and metrics == "all":
        return [f"{m}.{v}" for m, variants in METRIC_VARIANTS.items() for v in variants]
    if isinstance(metrics, str):
        metrics = [metrics]

    names: list[str] = []
    for spec in metrics:
        metric, _, variant = spec.partition(".")
        if metric not in METRIC_VARIANTS or (variant and variant not in METRIC_VARIANTS[metric]):
            known = ", ".join(f"{m}.{v}" for m, vs in METRIC_VARIANTS.items() for v in vs)
            raise ValueError(f"Metric {spec!r} is not recognised. Choose from: {known}.")
        name = f"{metric}.{variant or METRIC_VARIANTS[metric][0]}"
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
) -> list[str]:
    """Validate the options and return the metric names; "all" silently drops metrics the method makes constant."""
    names = parse_metrics(metrics)
    check_graph_method(graph_method, graph_params)
    if summary not in _SUMMARIES:
        raise ValueError(f"summary={summary!r} is not recognised. Choose \"auc\" or \"mean\".")
    if not (float(n_random).is_integer() and n_random >= 0):
        raise ValueError(f"n_random must be a non-negative integer, got {n_random}.")
    if n_random and not random_swaps > 0:
        raise ValueError(f"random_swaps must be positive, got {random_swaps}.")

    constant = {}
    if graph_method == "mst":
        for m in names:
            if m.split(".")[0] in ("clust_coeff", "ge_local"):
                constant[m] = "is always 0 on a spanning tree (graph_method=\"mst\")"
    resolved = resolve_sign(graph_method, sign)
    if resolved in ("positive", "negative") and "strength.neg" in names:
        constant["strength.neg"] = f"is always 0 when only one sign of weights is kept (sign=\"{resolved}\")"

    if constant and metrics == "all":
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
) -> dict:
    """Return {"subj", "metric.variant": {label: value}, ["curves"], ["shortfall"]} for one subject; options as in compute_graph_metrics."""
    names = check_options(metrics, graph_method, graph_params, sign, summary, n_random, random_swaps)
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
    if sweep is None:
        A = build_adjacency(mat, graph_method, params, signed)
        results = _node_values(names, A, n_random, random_swaps, rng)
    else:
        stacked: dict[str, list] = {}
        for setting in settings:
            A = build_adjacency(mat, graph_method, setting, signed)
            for m, v in _node_values(names, A, n_random, random_swaps, rng).items():
                stacked.setdefault(m, []).append(v)
        curves = {m: np.vstack(rows) for m, rows in stacked.items()}
        results = {m: _summarize(curve, values, summary) for m, curve in curves.items()}

    out: dict = {"subj": subj}
    for m, vals in results.items():
        out[m] = dict(zip(node_labels, vals))
    if return_curves and curves is not None:
        out["curves"] = _curves_frame(curves, values, node_labels)
    shortfall = _shortfall(mat, graph_method, settings)
    if shortfall:
        out["shortfall"] = shortfall
    return out


def normalized_metrics(names: list[str], n_random: int) -> list[str]:
    """Metrics that also get a ".norm" ratio to random networks; randomizing keeps degree, so strength.bin never does."""
    return [m for m in names if m != "strength.bin"] if n_random else []


def output_names(names: list[str], n_random: int) -> list[str]:
    return names + [f"{m}.norm" for m in normalized_metrics(names, n_random)]


def _node_values(
    names: list[str], A: np.ndarray, n_random: int, swaps: float, rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    values = {m: compute_metric(m, A) for m in names}
    norm = normalized_metrics(names, n_random)
    if not norm:
        return values
    sums = {m: np.zeros(len(A)) for m in norm}
    for _ in range(n_random):
        R = randomize_signed(A, swaps, rng)
        for m in norm:
            sums[m] += compute_metric(m, R)
    for m in norm:
        # NaN where the random networks give 0, e.g. clustering of a node with one neighbour
        values[f"{m}.norm"] = np.divide(values[m], sums[m] / n_random, out=np.full(len(A), np.nan), where=sums[m] != 0)
    return values


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


def compute_metric(name: str, A: np.ndarray) -> np.ndarray:
    """Node values of one "metric.variant" on a signed adjacency matrix with a zero diagonal."""
    import bct

    W = np.abs(A)
    B = (A != 0).astype(float)
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
        "strength.abs":           lambda: W.sum(axis=1),
        "strength.pos":           lambda: np.clip(A, 0, None).sum(axis=1),
        "strength.neg":           lambda: -np.clip(A, None, 0).sum(axis=1),
        "strength.bin":           lambda: B.sum(axis=1),
        "ge_local.wang":          lambda: bct.efficiency_wei(W, local=True),
        "ge_local.rubinov":       lambda: _local_efficiency_rubinov(W),
        "ge_local.bin":           lambda: bct.efficiency_bin(B, local=True),
    }
    values = np.asarray(funcs[name](), dtype=float)
    if name.startswith("clust_coeff."):
        values = np.nan_to_num(values, nan=0.0)  # nodes with fewer than two neighbours
    return values


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
    return pd.concat(
        [
            pd.DataFrame({
                "metric": m,
                "node": np.tile(np.asarray(labels, dtype=object), len(x)),
                "threshold": np.repeat(x, len(labels)),
                "value": curve.ravel(),
            })
            for m, curve in curves.items()
        ],
        ignore_index=True,
    )
