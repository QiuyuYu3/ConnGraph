"""
Top-level entry point: compute_graph_metrics.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import pandas as pd

from brainnet3d.graph_theory.aggregation import (
    build_net2rois,
    build_net_hemi2rois,
    compute_net_corr,
)
from brainnet3d.exceptions import DataValidationError
from brainnet3d.graph_theory.metrics import check_options, process_subject


@dataclass
class GraphMetricsResult:
    """Container for all outputs of compute_graph_metrics."""

    network_df:       pd.DataFrame | None = None
    node_df:          pd.DataFrame | None = None
    net_hemi_df:      pd.DataFrame | None = None
    net_corr_df:      pd.DataFrame | None = None
    net_hemi_corr_df: pd.DataFrame | None = None
    curves:           pd.DataFrame | None = None
    failed:           dict[str, dict[str, str]] = field(default_factory=dict)


def compute_graph_metrics(
    matrices: dict[str, pd.DataFrame],
    atlas: pd.DataFrame,
    level: str = "both",
    hemi_split: bool | str = True,
    metrics: str | list[str] | None = None,
    label_col: str = "label",
    network_col: str = "network_label",
    apply_fisher_z: bool = True,
    graph_method: str | Callable = "tmfg",
    graph_params: dict | None = None,
    sign: str | None = None,
    network_graph_method: str | Callable | None = None,
    network_graph_params: dict | None = None,
    summary: str = "auc",
    return_curves: bool = False,
    n_jobs: int = -1,
    output_dir: str | None = None,
    verbose: bool = True,
    hemi_col: str = "hemisphere",
) -> GraphMetricsResult:
    """Compute graph-theory metrics from pre-computed connectivity matrices.

    Parameters
    ----------
    matrices : dict[str, pd.DataFrame]
        {subject_id: N×N DataFrame} — square correlation matrices with ROI
        labels as both index and columns.  Keys become row IDs in the output.
    atlas : pd.DataFrame
        Must contain at minimum label_col (ROI name) and network_col (network
        assignment).  No path assumptions — pass the DataFrame directly.
    level : "network" | "node" | "both"
    hemi_split : network level graphs: True = one node per network and hemisphere (default),
        False = one node per network, "both" = both graphs.
    metrics : "metric" or "metric.variant" names, or "all". A bare name means its default variant.
        clust_coeff : costantini (signed, default), zhang, onnela, bin
        btwn_cent   : inv (length 1/|w|, default), inv_norm, log (length -log|w|), log_norm, bin
        strength    : abs (default), pos, neg, bin (degree)
        ge_local    : wang (default), rubinov, bin
        Defaults to the default variant of all four. "bin" variants ignore weights.
    label_col : atlas column with ROI labels (default "label").
    network_col : atlas column with network assignments (default "network_label").
    hemi_col : atlas column with L/R hemisphere labels (default "hemisphere"); used by hemi_split when present.
    apply_fisher_z : Fisher-z transform before averaging (recommended for r-matrices).
    graph_method : how edges are kept; graph_params gives each method's parameter.
        "tmfg" (default)  triangulated maximally filtered graph
        "full"            every edge
        "absolute"        |w| > {"threshold": r}
        "density"         strongest |w| at {"density": d}, d in (0, 1]
        "eco"             density giving a mean degree of 3
        "knn"             each node's {"k": k} strongest edges
        "mst"             maximum spanning tree of |w|
        "mst_density"     spanning tree plus the strongest edges up to {"density": d}
        "omst"            orthogonal spanning trees, as many as maximise global efficiency minus cost
        "percolation"     highest |w| cutoff that keeps the largest component connected
        or a callable ``f(matrix, **graph_params)`` returning an nx.Graph or an adjacency array.
        A list of values for "threshold" or "density" computes each metric at every value and
        reduces the curve with ``summary``.
    graph_params : parameters of graph_method, e.g. {"density": 0.1} or {"density": [0.2, 0.25, 0.3]}.
    sign : "abs" keeps negative weights and ranks edges by |w|; "positive" removes negative weights
        first. Default: "abs" for tmfg and callables, "positive" for the other methods.
    network_graph_method, network_graph_params : network level override; defaults to graph_method and graph_params.
    summary : how a parameter range is reduced: "auc" (trapezoidal area, default) or "mean" (area / range width).
    return_curves : also keep the per-value metrics of a parameter range in ``result.curves``.
    n_jobs : parallel workers for node-level computation; -1 = cpu_count - 1.
    output_dir : if given, saves CSV files there, one folder per level and one file per metric.
    verbose : print node-level progress and the names of saved files.

    Returns
    -------
    GraphMetricsResult
        .network_df       — subjects × (metric, network) columns (hemi_split False or "both")
        .node_df          — subjects × (metric, roi) columns
        .net_hemi_df      — subjects × (metric, hemisphere_network) columns (hemi_split True or "both")
        .net_corr_df      — wide-format pairwise network correlations
        .net_hemi_corr_df — like net_corr_df but hemisphere-split
        .curves           — long table (level, ID, metric, node, threshold, value) when return_curves is set
        .failed           — {level: {subject_id: error}} for subjects left as NaN rows
    """
    if not (isinstance(hemi_split, bool) or hemi_split == "both"):
        raise ValueError(f"hemi_split={hemi_split!r} is not recognised. Choose True, False or \"both\".")

    node_opts = dict(graph_method=graph_method, graph_params=graph_params, sign=sign,
                     summary=summary, return_curves=return_curves)
    net_opts = dict(node_opts)
    if network_graph_method is not None:
        net_opts.update(graph_method=network_graph_method, graph_params=network_graph_params)
    elif network_graph_params is not None:
        net_opts.update(graph_params=network_graph_params)

    want_node = level in ("node", "both")
    want_network = level in ("network", "both") and hemi_split in (False, "both")
    want_hemi = level in ("network", "both") and hemi_split in (True, "both")

    node_metrics = check_options(metrics, graph_method, graph_params, sign, summary) if want_node else []
    net_metrics = (
        check_options(metrics, net_opts["graph_method"], net_opts["graph_params"], sign, summary)
        if want_network or want_hemi else []
    )
    if want_node:
        _check_node_matrices(matrices)

    subject_ids = list(matrices.keys())
    result = GraphMetricsResult()
    curves: list[pd.DataFrame] = []

    if want_network:
        net2rois = build_net2rois(atlas, label_col, network_col)
        result.network_df, result.net_corr_df = _network_level(
            "network", matrices, net2rois, apply_fisher_z, net_metrics, net_opts, result, curves,
        )

    if want_node:
        atlas_rois = atlas[label_col].dropna().tolist()
        node_df = pd.DataFrame(
            index=subject_ids,
            columns=pd.MultiIndex.from_product([node_metrics, atlas_rois], names=["metric", "roi"]),
        )

        workers = max(1, (os.cpu_count() or 2) - 1) if n_jobs == -1 else max(1, n_jobs)
        if verbose:
            print(f"[graph_theory] Node-level: {len(subject_ids)} subjects, {workers} workers")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_subject,
                    sid,
                    matrices[sid].values.astype(float),
                    matrices[sid].columns.tolist(),
                    node_metrics,
                    **node_opts,
                ): sid
                for sid in subject_ids
                if sid in matrices
            }
            total = len(futures)
            done = 0
            for future in as_completed(futures):
                done += 1
                sid = futures[future]
                if verbose:
                    print(f"  [{done}/{total}] {sid}")
                try:
                    res = future.result()
                except Exception as e:
                    _record_failure(result, "node", sid, e)
                    continue
                for m in node_metrics:
                    for roi in atlas_rois:
                        if roi in res[m]:
                            node_df.at[sid, (m, roi)] = res[m][roi]
                _collect_curves(curves, "node", sid, res)

        result.node_df = node_df.astype(float)

    if want_hemi:
        net_hemi2rois = build_net_hemi2rois(atlas, label_col, network_col, hemi_col)
        result.net_hemi_df, result.net_hemi_corr_df = _network_level(
            "network_hemi", matrices, net_hemi2rois, apply_fisher_z, net_metrics, net_opts, result, curves,
        )

    if curves:
        result.curves = pd.concat(curves, ignore_index=True)

    if result.failed:
        lines = [
            f"  {lvl} / {sid}: {msg}"
            for lvl, subs in result.failed.items()
            for sid, msg in subs.items()
        ]
        warnings.warn(
            "Graph metrics failed for some subjects; their rows are left as NaN:\n" + "\n".join(lines),
            stacklevel=2,
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save(result, output_dir, verbose)

    return result


def _network_level(
    level: str,
    matrices: dict[str, pd.DataFrame],
    net2rois: dict,
    apply_fisher_z: bool,
    metrics: list[str],
    opts: dict,
    result: GraphMetricsResult,
    curves: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_net_corr = compute_net_corr(matrices, net2rois, apply_fisher_z)
    nets = list(next(iter(all_net_corr.values())).columns)
    net_df = pd.DataFrame(
        index=list(matrices.keys()),
        columns=pd.MultiIndex.from_product([metrics, nets], names=["metric", "network"]),
    )
    for sub_id, corr in all_net_corr.items():
        corrmat = corr.values.astype(float)
        if apply_fisher_z:
            corrmat = np.tanh(corrmat)
        try:
            res = process_subject(sub_id, corrmat, nets, metrics, **opts)
        except Exception as e:
            _record_failure(result, level, sub_id, e)
            continue
        for m in metrics:
            for n in nets:
                net_df.at[sub_id, (m, n)] = res[m].get(n, np.nan)
        _collect_curves(curves, level, sub_id, res)
    return net_df.astype(float), _net_corr_to_wide(all_net_corr)


def _collect_curves(curves: list[pd.DataFrame], level: str, sub_id: str, res: dict) -> None:
    if "curves" in res:
        curves.append(res["curves"].assign(level=level, ID=sub_id)[
            ["level", "ID", "metric", "node", "threshold", "value"]
        ])


def _record_failure(result: GraphMetricsResult, level: str, sub_id: str, exc: Exception) -> None:
    result.failed.setdefault(level, {})[sub_id] = f"{type(exc).__name__}: {exc}"


def _check_node_matrices(matrices: dict[str, pd.DataFrame]) -> None:
    # The diagonal is zeroed before graph construction, so only off-diagonal values must be finite
    problems = []
    for sub_id, df in matrices.items():
        mat = df.values.astype(float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            problems.append(f"{sub_id}: not square, shape {mat.shape}")
            continue
        off_diag = mat[~np.eye(len(mat), dtype=bool)]
        n_nan = int(np.isnan(off_diag).sum())
        n_inf = int(np.isinf(off_diag).sum())
        if n_nan:
            problems.append(f"{sub_id}: {n_nan} NaN value(s) off the diagonal")
        if n_inf:
            problems.append(f"{sub_id}: {n_inf} Inf value(s) off the diagonal")
    if problems:
        raise DataValidationError("Node-level input check failed:\n  " + "\n  ".join(problems))


def _net_corr_to_wide(all_corr: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten {sub_id: net×net DataFrame} to a wide DataFrame (upper triangle)."""
    net_names = list(next(iter(all_corr.values())).columns)
    rows = []
    for sub_id, df in all_corr.items():
        row: dict = {"ID": sub_id}
        for i, n1 in enumerate(net_names):
            for j in range(i, len(net_names)):
                n2 = net_names[j]
                row[f"{n1}__{n2}"] = df.loc[n1, n2]
        rows.append(row)
    return pd.DataFrame(rows).set_index("ID")


def _save(result: GraphMetricsResult, output_dir: str, verbose: bool) -> None:
    def _write(df: pd.DataFrame, *parts: str, index: bool = True) -> None:
        path = os.path.join(output_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = df.reset_index().rename(columns={"index": "ID"}) if index else df
        out.to_csv(path, index=False)
        if verbose:
            print(f"  Saved {'/'.join(parts)}")

    for folder, df in (("node", result.node_df), ("network", result.network_df), ("network_hemi", result.net_hemi_df)):
        if df is not None:
            for metric in df.columns.get_level_values(0).unique():
                _write(df[metric], folder, f"{metric}.csv")
    if result.net_corr_df is not None:
        _write(result.net_corr_df, "correlation", "network.csv")
    if result.net_hemi_corr_df is not None:
        _write(result.net_hemi_corr_df, "correlation", "network_hemi.csv")
    if result.curves is not None:
        for level, df in result.curves.groupby("level", sort=False):
            _write(df.drop(columns="level"), level, "curves.csv", index=False)
