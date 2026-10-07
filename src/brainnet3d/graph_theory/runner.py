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
from brainnet3d.graph_theory.metrics import METRIC_NAMES, _check_graph_method, process_subject


@dataclass
class GraphMetricsResult:
    """Container for all outputs of compute_graph_metrics."""

    network_df:       pd.DataFrame | None = None
    node_df:          pd.DataFrame | None = None
    net_hemi_df:      pd.DataFrame | None = None
    net_corr_df:      pd.DataFrame | None = None
    net_hemi_corr_df: pd.DataFrame | None = None
    failed:           dict[str, dict[str, str]] = field(default_factory=dict)


def compute_graph_metrics(
    matrices: dict[str, pd.DataFrame],
    atlas: pd.DataFrame,
    level: str = "both",
    hemi_split: bool = True,
    metrics: list[str] | None = None,
    label_col: str = "label",
    network_col: str = "network_label",
    apply_fisher_z: bool = True,
    graph_method: str | Callable = "tmfg",
    n_jobs: int = -1,
    output_dir: str | None = None,
    verbose: bool = True,
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
    hemi_split : compute network metrics split by hemisphere (L_/R_ prefix rule).
    metrics : list from {"clust_coeff", "btwn_cent", "strength", "ge_local"}.
        Defaults to all four.
    label_col : atlas column with ROI labels (default "label").
    network_col : atlas column with network assignments (default "network_label").
    apply_fisher_z : Fisher-z transform before averaging (recommended for r-matrices).
    graph_method : "tmfg" (default) or a callable ``f(corrmat) -> nx.Graph``.
        Applied during graph construction in both network- and node-level steps.
    n_jobs : parallel workers for node-level computation; -1 = cpu_count - 1.
    output_dir : if given, saves CSV files there (directory is created if needed).
    verbose : print node-level progress and the names of saved files.

    Returns
    -------
    GraphMetricsResult
        .network_df       — subjects × (metric_network) flat columns
        .node_df          — subjects × (metric_roi) flat columns
        .net_hemi_df      — like network_df but hemisphere-split (hemi_split=True)
        .net_corr_df      — wide-format pairwise network correlations
        .net_hemi_corr_df — like net_corr_df but hemisphere-split
        .failed           — {level: {subject_id: error}} for subjects left as NaN rows
    """
    if metrics is None:
        metrics = list(METRIC_NAMES)
    _check_graph_method(graph_method)
    if level in ("node", "both"):
        _check_node_matrices(matrices)

    subject_ids = list(matrices.keys())
    result = GraphMetricsResult()

    # Network-level
    if level in ("network", "both"):
        net2rois = build_net2rois(atlas, label_col, network_col)
        all_net_corr = compute_net_corr(matrices, net2rois, apply_fisher_z)

        nets = list(next(iter(all_net_corr.values())).columns)
        net_df = pd.DataFrame(
            index=subject_ids,
            columns=pd.MultiIndex.from_product([metrics, nets]),
        )
        for sub_id in subject_ids:
            if sub_id not in all_net_corr:
                continue
            corrmat = all_net_corr[sub_id].values.astype(float)
            if apply_fisher_z:
                corrmat = np.tanh(corrmat)
            try:
                res = process_subject(sub_id, corrmat, list(nets), metrics, graph_method)
            except Exception as e:
                _record_failure(result, "network", sub_id, e)
                continue
            for m in metrics:
                for n in nets:
                    net_df.at[sub_id, (m, n)] = res.get(m, {}).get(n, np.nan)

        net_df.columns = ["_".join(c) for c in net_df.columns]
        result.network_df = net_df
        result.net_corr_df = _net_corr_to_wide(all_net_corr)

    # Node-level
    if level in ("node", "both"):
        atlas_rois = atlas[label_col].dropna().tolist()
        node_df = pd.DataFrame(
            index=subject_ids,
            columns=pd.MultiIndex.from_product([metrics, atlas_rois]),
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
                    metrics,
                    graph_method,
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
                for m in metrics:
                    if m not in res:
                        continue
                    for roi in atlas_rois:
                        if roi in res[m]:
                            node_df.at[sid, (m, roi)] = res[m][roi]

        node_df.columns = ["_".join(c) for c in node_df.columns]
        result.node_df = node_df

    # Hemi-split network-level
    if hemi_split and level in ("network", "both"):
        net_hemi2rois = build_net_hemi2rois(atlas, label_col, network_col)
        all_net_hemi_corr = compute_net_corr(matrices, net_hemi2rois, apply_fisher_z)

        nets_hemi = list(next(iter(all_net_hemi_corr.values())).columns)
        net_hemi_df = pd.DataFrame(
            index=subject_ids,
            columns=pd.MultiIndex.from_product([metrics, nets_hemi]),
        )
        for sub_id in subject_ids:
            if sub_id not in all_net_hemi_corr:
                continue
            corrmat = all_net_hemi_corr[sub_id].values.astype(float)
            if apply_fisher_z:
                corrmat = np.tanh(corrmat)
            try:
                res = process_subject(sub_id, corrmat, list(nets_hemi), metrics, graph_method)
            except Exception as e:
                _record_failure(result, "network_hemi", sub_id, e)
                continue
            for m in metrics:
                for n in nets_hemi:
                    net_hemi_df.at[sub_id, (m, n)] = res.get(m, {}).get(n, np.nan)

        net_hemi_df.columns = ["_".join(c) for c in net_hemi_df.columns]
        result.net_hemi_df = net_hemi_df
        result.net_hemi_corr_df = _net_corr_to_wide(all_net_hemi_corr)

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
    def _write(df: pd.DataFrame | None, name: str) -> None:
        if df is not None:
            out = df.reset_index().rename(columns={"index": "ID"})
            out.to_csv(os.path.join(output_dir, name), index=False)
            if verbose:
                print(f"  Saved {name}")

    _write(result.network_df,       "network_graph_theory.csv")
    _write(result.node_df,          "node_graph_theory.csv")
    _write(result.net_hemi_df,      "network_graph_theory_hemi.csv")
    _write(result.net_corr_df,      "network_correlation.csv")
    _write(result.net_hemi_corr_df, "network_correlation_hemi.csv")
