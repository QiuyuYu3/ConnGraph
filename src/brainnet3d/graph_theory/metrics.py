"""
Single-subject graph-theory metrics on a sparsified connectivity matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from brainnet3d.graph_theory.sparsify import (
    build_adjacency,
    check_graph_method,
    inverse_distances,
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
) -> list[str]:
    """Validate the options and return the metric names; "all" silently drops metrics the method makes constant."""
    names = parse_metrics(metrics)
    check_graph_method(graph_method, graph_params)
    if summary not in _SUMMARIES:
        raise ValueError(f"summary={summary!r} is not recognised. Choose \"auc\" or \"mean\".")

    constant = {}
    if graph_method == "mst":
        for m in names:
            if m.split(".")[0] in ("clust_coeff", "ge_local"):
                constant[m] = "is always 0 on a spanning tree (graph_method=\"mst\")"
    if resolve_sign(graph_method, sign) == "positive" and "strength.neg" in names:
        constant["strength.neg"] = "is always 0 when negative weights are removed (sign=\"positive\")"

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
) -> dict:
    """Return {"subj", "metric.variant": {label: value}, ["curves"]} for one subject; options as in compute_graph_metrics."""
    names = check_options(metrics, graph_method, graph_params, sign, summary)
    params = dict(graph_params or {})
    sweep, values = check_graph_method(graph_method, params)

    mat = np.array(corrmat, dtype=float)
    np.fill_diagonal(mat, 0)
    if resolve_sign(graph_method, sign) == "positive":
        mat[mat < 0] = 0

    curves = None
    if sweep is None:
        A = build_adjacency(mat, graph_method, params)
        results = {m: compute_metric(m, A) for m in names}
    else:
        stacked: dict[str, list] = {m: [] for m in names}
        for value in values:
            A = build_adjacency(mat, graph_method, {**params, sweep: value})
            for m in names:
                stacked[m].append(compute_metric(m, A))
        curves = {m: np.vstack(rows) for m, rows in stacked.items()}
        results = {m: _summarize(curve, values, summary) for m, curve in curves.items()}

    out: dict = {"subj": subj}
    for m, vals in results.items():
        out[m] = dict(zip(node_labels, vals))
    if return_curves and curves is not None:
        out["curves"] = _curves_frame(curves, values, node_labels)
    return out


def compute_metric(name: str, A: np.ndarray) -> np.ndarray:
    """Node values of one "metric.variant" on a signed adjacency matrix with a zero diagonal."""
    import bct

    W = np.abs(A)
    B = (A != 0).astype(float)
    funcs = {
        "clust_coeff.costantini": lambda: bct.clustering_coef_wu_sign(A, coef_type="costantini"),
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
