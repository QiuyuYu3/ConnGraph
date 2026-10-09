"""
Top-level entry point: compute_graph_metrics.
"""

from __future__ import annotations

import json
import os
import platform
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from collections.abc import Callable

import numpy as np
import pandas as pd

from conngraph.graph_theory.aggregation import (
    build_net2rois,
    build_net_hemi2rois,
    compute_net_corr,
)
from conngraph.exceptions import DataValidationError
from conngraph.loaders import INPUT_ATTR
from conngraph.graph_theory.metrics import PARTITIONS, check_options, is_global, output_names, process_subject
from conngraph.graph_theory.sparsify import requested_edges, resolve_sign


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
    params:           dict = field(default_factory=dict)
    nodes:            pd.DataFrame | None = None
    mean_matrix:      pd.DataFrame | None = None
    global_df:        pd.DataFrame | None = None

    def save_report(self, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                    static_brain: bool = True, interactive_brain: bool = False) -> None:
        """Write an HTML report (plotly loads from its CDN); nodes overrides the atlas, e.g. to add x, y, z."""
        from conngraph.report.pages import save_graph_report

        save_graph_report(self, path, nodes, surfaces, static_brain, interactive_brain)


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
    n_jobs: int = -1,
    output_dir: str | None = None,
    verbose: bool = True,
    hemi_col: str = "hemisphere",
    graph_params: dict | None = None,
    sign: str | None = None,
    network_graph_method: str | Callable | None = None,
    network_graph_params: dict | None = None,
    summary: str = "auc",
    return_curves: bool = False,
    normalize_weights: bool = False,
    n_random: int = 0,
    random_swaps: float = 10,
    random_seed: int | None = None,
    exclude_networks: str | list[str] | tuple[str, ...] | None = ("None",),
    signed_fallback: str = "abs",
    partitions: tuple[str, ...] | None = None,
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
        "absolute"        |w| >= {"threshold": r}
        "density"         strongest |w| at {"density": d}, d in (0, 1]
        "eco"             density giving a mean degree of 3
        "knn"             each node's {"k": k} strongest edges
        "mst"             maximum spanning tree of |w|
        "mst_density"     spanning tree plus the strongest edges up to {"density": d}
        "omst"            orthogonal spanning trees, as many as maximise global efficiency minus cost
        "percolation"     highest |w| cutoff that keeps the largest component connected
        "disparity"       edges whose share of either end's strength is significant at {"alpha": a}
        "pmfg"            planar maximally filtered graph; much slower than tmfg on large graphs
        or a callable ``f(matrix, **graph_params)`` returning an undirected nx.Graph or a symmetric adjacency array.
        A list of values for "threshold", "density" or "alpha" computes each metric at every value and
        reduces the curve with ``summary``.
    graph_params : parameters of graph_method, e.g. {"density": 0.1} or {"density": [0.2, 0.25, 0.3]}.
    sign : how negative weights are treated.
        "abs"       rank edges by |w| and keep the sign (signed variants use it, the others use |w|)
        "signed"    rank edges by w, so negative edges are kept only when needed; not for omst or disparity
        "positive"  remove negative weights first
        "negative"  keep only negative weights, as positive magnitudes
        Default: "abs" for tmfg and callables, "positive" for the other methods.
    normalize_weights : divide each matrix by its largest |w| after the sign rule (default False).
    network_graph_method, network_graph_params : network level override; defaults to graph_method and graph_params.
    summary : how a parameter range is reduced: "auc" (trapezoidal area, default) or "mean" (area / range width).
    return_curves : also keep the per-value metrics of a parameter range in ``result.curves``.
    n_random : random networks per graph (default 0, off); adds "metric.variant.norm", the value divided by
        its mean over the random networks, at every parameter value before ``summary``. Random networks keep
        each node's positive and negative degree and redistribute the weights within each sign, so
        strength.bin (the degree) gets no ".norm". The signed clust_coeff.costantini averages near 0 on random
        networks, so its ratio is unstable and can change sign.
    random_swaps : average number of swaps per edge when randomizing (default 10).
    random_seed : seed for reproducible random networks; when None, the seed drawn is recorded in ``result.params``.
    exclude_networks : network labels whose ROIs are left out of the network level, e.g. unassigned parcels
        (default "None"); the node level keeps them. None or () keeps every label.
    n_jobs : parallel workers for node-level computation; -1 = cpu_count - 1.
    output_dir : if given, saves TSV files there, one folder per level and one file per metric, the node table as
        nodes.tsv, and ``result.params`` as parameters.json.
    verbose : print each level's graph method and sign rule, node-level progress and the names of saved files.

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
        .params           — options, graph method and sign of each level, package versions and data summary
        .nodes            — the atlas table as passed
        .mean_matrix      — group mean node-level connectivity (Fisher z averaged when apply_fisher_z)
        Call .save_report(path) to write an HTML report.
    """
    if not (isinstance(hemi_split, bool) or hemi_split == "both"):
        raise ValueError(f"hemi_split={hemi_split!r} is not recognised. Choose True, False or \"both\".")

    has_networks = network_col in atlas.columns and atlas[network_col].notna().any()
    if partitions is None:
        partitions = PARTITIONS if has_networks else ("louvain",)
    elif "networks" in partitions and not has_networks:
        raise ValueError(f"partitions includes \"networks\", but the atlas has no {network_col!r} column.")
    node_opts = dict(graph_method=graph_method, graph_params=graph_params, sign=sign,
                     summary=summary, return_curves=return_curves, normalize_weights=normalize_weights,
                     n_random=n_random, random_swaps=random_swaps, signed_fallback=signed_fallback,
                     partitions=tuple(partitions))
    net_opts = dict(node_opts)
    if network_graph_method is not None:
        net_opts.update(graph_method=network_graph_method, graph_params=network_graph_params)
    elif network_graph_params is not None:
        net_opts.update(graph_params=network_graph_params)

    want_node = level in ("node", "both")
    want_network = level in ("network", "both") and hemi_split in (False, "both")
    want_hemi = level in ("network", "both") and hemi_split in (True, "both")

    node_metrics = (
        check_options(metrics, graph_method, graph_params, sign, summary, n_random, random_swaps, signed_fallback,
                      node_opts["partitions"]) if want_node else []
    )
    net_metrics = (
        check_options(metrics, net_opts["graph_method"], net_opts["graph_params"], sign, summary, n_random, random_swaps,
                      signed_fallback, level="network")
        if want_network or want_hemi else []
    )
    if want_node:
        _check_node_matrices(matrices)
    net_atlas, excluded = (
        _drop_networks(atlas, network_col, exclude_networks, verbose) if want_network or want_hemi else (atlas, {})
    )
    net2rois = build_net2rois(net_atlas, label_col, network_col) if want_network else {}
    net_hemi2rois = build_net_hemi2rois(net_atlas, label_col, network_col, hemi_col) if want_hemi else {}
    sizes = {}
    if want_node:
        sizes["node"] = min((len(df) for df in matrices.values()), default=0)
    if want_network:
        sizes["network"] = len(net2rois)
    if want_hemi:
        sizes["network_hemi"] = len(net_hemi2rois)
    _check_spanning_tree_fits(sizes, node_opts, net_opts)
    _check_knn_fits(sizes, node_opts, net_opts)

    subject_ids = list(matrices.keys())
    result = GraphMetricsResult(nodes=atlas)
    if want_node:
        result.mean_matrix = _mean_matrix(matrices, apply_fisher_z)
    curves: list[pd.DataFrame] = []
    shortfalls: list[str] = []
    global_values: dict[tuple[str, str], dict] = {}
    # Seeds follow the subject ID, so a subject's random networks do not depend on the other subjects or workers
    root_seed = np.random.SeedSequence(random_seed)
    seeds = {
        lvl: {sid: np.random.SeedSequence([root_seed.entropy, k, *str(sid).encode()]) for sid in subject_ids}
        for k, lvl in enumerate(("network", "node", "network_hemi"))
    }

    if want_network:
        if verbose:
            print(f"[graph_theory] Network-level: {len(net2rois)} networks, {_describe_graph(net_opts)}")
        result.network_df, result.net_corr_df = _network_level(
            "network", matrices, net2rois, apply_fisher_z, net_metrics, net_opts, result, curves, shortfalls,
            seeds["network"], global_values,
        )

    if want_node:
        atlas_rois = atlas[label_col].dropna().tolist()
        node_columns = [m for m in output_names(node_metrics, n_random) if not is_global(m)]
        global_columns = [m for m in output_names(node_metrics, n_random) if is_global(m)]
        module_of = _modules(atlas, label_col, network_col, exclude_networks) if has_networks else {}
        node_df = pd.DataFrame(
            index=subject_ids,
            columns=pd.MultiIndex.from_product([node_columns, atlas_rois], names=["metric", "roi"]),
        )

        workers = max(1, (os.cpu_count() or 2) - 1) if n_jobs == -1 else max(1, n_jobs)
        if verbose:
            print(f"[graph_theory] Node-level: {len(subject_ids)} subjects, {workers} workers, "
                  f"{_describe_graph(node_opts)}")

        calls = {
            sid: partial(
                process_subject,
                sid,
                matrices[sid].values.astype(float),
                matrices[sid].columns.tolist(),
                node_metrics,
                **node_opts,
                random_seed=seeds["node"][sid],
                modules=[module_of.get(label, np.nan) for label in matrices[sid].columns] if module_of else None,
            )
            for sid in subject_ids
            if sid in matrices
        }
        total = len(calls)
        done = 0
        for sid, future in _completed(calls, workers):
            done += 1
            if verbose:
                print(f"  [{done}/{total}] {sid}")
            try:
                res = future.result()
            except Exception as e:
                _record_failure(result, "node", sid, e)
                continue
            for m in node_columns:
                for roi in atlas_rois:
                    if roi in res[m]:
                        node_df.at[sid, (m, roi)] = res[m][roi]
            for m in global_columns:
                global_values.setdefault(("node", m), {})[sid] = res[m]
            _collect_curves(curves, "node", sid, res)
            _collect_shortfall(shortfalls, "node", sid, res)

        result.node_df = node_df.astype(float) if node_columns else None

    if want_hemi:
        if verbose:
            print(f"[graph_theory] Network-level (hemisphere split): {len(net_hemi2rois)} networks, "
                  f"{_describe_graph(net_opts)}")
        result.net_hemi_df, result.net_hemi_corr_df = _network_level(
            "network_hemi", matrices, net_hemi2rois, apply_fisher_z, net_metrics, net_opts, result, curves, shortfalls,
            seeds["network_hemi"], global_values,
        )

    if curves:
        result.curves = pd.concat(curves, ignore_index=True)
    if global_values:
        order = {"node": 0, "network": 1, "network_hemi": 2}
        columns = sorted(global_values, key=lambda key: order[key[0]])
        result.global_df = pd.DataFrame(
            {key: [global_values[key].get(sid, np.nan) for sid in subject_ids] for key in columns},
            index=subject_ids, dtype=float,
        )
        result.global_df.columns = pd.MultiIndex.from_tuples(columns, names=["level", "metric"])

    if shortfalls:
        warnings.warn(
            "Fewer edges than requested, because the sign rule left too few edges:\n" + "\n".join(shortfalls),
            stacklevel=2,
        )

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

    levels = {
        lvl: _describe_level(opts, output_names(names, n_random), sizes[lvl])
        for lvl, opts, names, wanted in (
            ("node", node_opts, node_metrics, want_node),
            ("network", net_opts, net_metrics, want_network),
            ("network_hemi", net_opts, net_metrics, want_hemi),
        )
        if wanted
    }
    result.params = _jsonable({
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "options": {
            "level": level, "hemi_split": hemi_split, "label_col": label_col, "network_col": network_col,
            "hemi_col": hemi_col, "apply_fisher_z": apply_fisher_z, "summary": summary,
            "return_curves": return_curves, "normalize_weights": normalize_weights, "n_random": n_random,
            "random_swaps": random_swaps, "random_seed": root_seed.entropy if n_random else None,
            "exclude_networks": exclude_networks, "signed_fallback": signed_fallback,
            "partitions": list(node_opts["partitions"]),
        },
        "levels": levels,
        "input": atlas.attrs.get(INPUT_ATTR, {}),
        "subjects": subject_ids,
        "excluded_rois": excluded,
        "failed": result.failed,
        "warnings": shortfalls,
    })

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
    shortfalls: list[str],
    seeds: dict,
    global_values: dict,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    all_net_corr = compute_net_corr(matrices, net2rois, apply_fisher_z)
    nets = list(next(iter(all_net_corr.values())).columns)
    columns = [m for m in output_names(metrics, opts["n_random"]) if not is_global(m)]
    global_columns = [m for m in output_names(metrics, opts["n_random"]) if is_global(m)]
    net_df = pd.DataFrame(
        index=list(matrices.keys()),
        columns=pd.MultiIndex.from_product([columns, nets], names=["metric", "network"]),
    )
    for sub_id, corr in all_net_corr.items():
        corrmat = corr.values.astype(float)
        if apply_fisher_z:
            corrmat = np.tanh(corrmat)
        try:
            res = process_subject(sub_id, corrmat, nets, metrics, **opts, random_seed=seeds[sub_id])
        except Exception as e:
            _record_failure(result, level, sub_id, e)
            continue
        for m in columns:
            for n in nets:
                net_df.at[sub_id, (m, n)] = res[m].get(n, np.nan)
        for m in global_columns:
            global_values.setdefault((level, m), {})[sub_id] = res[m]
        _collect_curves(curves, level, sub_id, res)
        _collect_shortfall(shortfalls, level, sub_id, res)
    return (net_df.astype(float) if columns else None), _net_corr_to_wide(all_net_corr)


def _mean_matrix(matrices: dict[str, pd.DataFrame], fisher: bool) -> pd.DataFrame:
    # A running sum, so a large cohort is never stacked in memory
    labels = next(iter(matrices.values())).index
    total = np.zeros((len(labels), len(labels)))
    count = np.zeros_like(total)
    for mat in matrices.values():
        a = mat.reindex(index=labels, columns=labels).to_numpy(dtype=float)
        if fisher:
            a = np.arctanh(np.clip(a, -1 + 1e-7, 1 - 1e-7))
        present = ~np.isnan(a)
        total += np.where(present, a, 0.0)
        count += present
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    np.fill_diagonal(mean, 0)
    return pd.DataFrame(np.tanh(mean) if fisher else mean, index=labels, columns=labels)


def _modules(atlas: pd.DataFrame, label_col: str, network_col: str, exclude) -> dict:
    """Each ROI's network for the "networks" partition; NaN for ROIs whose label is excluded."""
    excluded = set() if exclude is None else ({exclude} if isinstance(exclude, str) else set(exclude))
    return {label: (np.nan if net in excluded else net) for label, net in zip(atlas[label_col], atlas[network_col])}


def _drop_networks(atlas: pd.DataFrame, network_col: str, exclude, verbose: bool) -> tuple[pd.DataFrame, dict]:
    """Return the atlas without the excluded network labels, and the number of ROIs left out per label."""
    if exclude is None:
        return atlas, {}
    labels = [exclude] if isinstance(exclude, str) else list(exclude)
    drop = atlas[network_col].isin(labels)
    counts = {str(k): int(v) for k, v in atlas.loc[drop, network_col].value_counts(sort=False).items()}
    if verbose and drop.any():
        names = ", ".join(repr(x) for x in sorted(atlas.loc[drop, network_col].unique(), key=str))
        print(f"[graph_theory] Network-level: leaving out {int(drop.sum())} ROIs labelled {names}")
    return atlas[~drop], counts


def _method_name(method: str | Callable) -> str:
    return getattr(method, "__name__", repr(method)) if callable(method) else method


def _describe_graph(opts: dict) -> str:
    return f"graph_method={_method_name(opts['graph_method'])}, sign={resolve_sign(opts['graph_method'], opts['sign'])}"


def _describe_level(opts: dict, metrics: list[str], n_nodes: int) -> dict:
    return {
        "graph_method": _method_name(opts["graph_method"]),
        "graph_params": opts["graph_params"] or {},
        "sign": resolve_sign(opts["graph_method"], opts["sign"]),
        "metrics": metrics,
        "n_nodes": n_nodes,
    }


def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    import conngraph

    versions: dict[str, str | None] = {"conngraph": conngraph.__version__}
    for name in ("numpy", "scipy", "pandas", "networkx", "bctpy", "topcorr", "nilearn", "scikit-learn"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(value):
    """Convert numpy values and tuples to plain JSON types; anything else unknown becomes its repr."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _collect_curves(curves: list[pd.DataFrame], level: str, sub_id: str, res: dict) -> None:
    if "curves" in res:
        curves.append(res["curves"].assign(level=level, ID=sub_id)[
            ["level", "ID", "metric", "node", "threshold", "value"]
        ])


def _collect_shortfall(shortfalls: list[str], level: str, sub_id: str, res: dict) -> None:
    if "shortfall" in res:
        shortfalls.append(f"  {level} / {sub_id}: {res['shortfall']}")


def _completed(calls: dict, workers: int):
    """(key, finished future) pairs, run in this process when there is one worker, else in a process pool."""
    if workers == 1:
        for key, call in calls.items():
            future = Future()
            try:
                future.set_result(call())
            except Exception as e:
                future.set_exception(e)
            yield key, future
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call): key for key, call in calls.items()}
        for future in as_completed(futures):
            yield futures[future], future


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


def _check_spanning_tree_fits(sizes: dict[str, int], node_opts: dict, net_opts: dict) -> None:
    # mst_density starts from a spanning tree, so every density must keep at least n - 1 edges
    problems = {}
    for level, n in sizes.items():
        opts = node_opts if level == "node" else net_opts
        if opts["graph_method"] != "mst_density":
            continue
        low = [d for d in np.atleast_1d(opts["graph_params"]["density"])
               if requested_edges(n, "mst_density", {"density": d}) < n - 1]
        if low:
            problems[level] = (
                f"{level} ({n} nodes): a spanning tree needs {n - 1} edges, i.e. density >= 2/{n} = {2 / n:.3g}; "
                f"got {', '.join(f'{d:g}' for d in low)}"
            )
    _raise_size_problems("graph_method=\"mst_density\" is too sparse", problems)


def _check_knn_fits(sizes: dict[str, int], node_opts: dict, net_opts: dict) -> None:
    problems = {}
    for level, n in sizes.items():
        opts = node_opts if level == "node" else net_opts
        if opts["graph_method"] == "knn" and int(opts["graph_params"]["k"]) > n - 1:
            k = int(opts["graph_params"]["k"])
            problems[level] = f"{level} ({n} nodes): k={k} is larger than the {n - 1} possible neighbours"
    _raise_size_problems("graph_method=\"knn\" asks for too many neighbours", problems)


def _raise_size_problems(header: str, problems: dict[str, str]) -> None:
    if not problems:
        return
    hint = "" if list(problems) == ["node"] else (
        "\nSet the network level separately with network_graph_method or network_graph_params."
    )
    raise ValueError(header + ":\n  " + "\n  ".join(problems.values()) + hint)


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


_LEVEL_FOLDERS = (("node_df", "node", "roi"), ("network_df", "network", "network"),
                  ("net_hemi_df", "network_hemi", "network"))
_CONNECTIVITY_FILES = (("net_corr_df", "network.tsv"), ("net_hemi_corr_df", "network_hemi.tsv"))
_MEAN_MATRIX_FILE = ("correlation", "node_mean.tsv")


def _save(result: GraphMetricsResult, output_dir: str, verbose: bool) -> None:
    def _write(df: pd.DataFrame, *parts: str, index: bool = True, label: str = "ID") -> None:
        path = os.path.join(output_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = df.rename_axis(label).reset_index() if index else df
        out.to_csv(path, sep="\t", index=False)
        if verbose:
            print(f"  Saved {'/'.join(parts)}")

    for attr, folder, _ in _LEVEL_FOLDERS:
        df = getattr(result, attr)
        if df is not None:
            for metric in df.columns.get_level_values(0).unique():
                _write(df[metric], folder, f"{metric}.tsv")
    for attr, name in _CONNECTIVITY_FILES:
        if getattr(result, attr) is not None:
            _write(getattr(result, attr), "correlation", name)
    if result.mean_matrix is not None:
        _write(result.mean_matrix, *_MEAN_MATRIX_FILE, label="label")
    if result.global_df is not None:
        for level in dict.fromkeys(result.global_df.columns.get_level_values(0)):
            _write(result.global_df[level], "global", f"{level}.tsv")
    if result.curves is not None:
        for level, df in result.curves.groupby("level", sort=False):
            _write(df.drop(columns="level"), level, "curves.tsv", index=False)
    if result.nodes is not None:
        _write(result.nodes, "nodes.tsv", index=False)
    with open(os.path.join(output_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(result.params, f, indent=2)
    if verbose:
        print("  Saved parameters.json")


def _read_table(path: str, index: str | None = None, text: tuple[str, ...] = ()) -> pd.DataFrame:
    # Only empty cells are missing, so a network called "None" keeps its name
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""],
                        dtype={c: str for c in (index, *text) if c})
    return table.set_index(index) if index else table


def _load(output_dir: str) -> GraphMetricsResult:
    """The result _save wrote to output_dir, read back from its files."""
    def path(*parts: str) -> str:
        return os.path.join(output_dir, *parts)

    with open(path("parameters.json"), encoding="utf-8") as f:
        params = json.load(f)
    result = GraphMetricsResult(params=params, failed=params.get("failed", {}))
    for attr, folder, row in _LEVEL_FOLDERS:
        names = params["levels"].get(folder, {}).get("metrics", [])
        tables = {m: _read_table(path(folder, f"{m}.tsv"), "ID") for m in names if os.path.isfile(path(folder, f"{m}.tsv"))}
        if tables:
            setattr(result, attr, pd.concat(tables, axis=1, names=["metric", row]))
    for attr, name in _CONNECTIVITY_FILES:
        if os.path.isfile(path("correlation", name)):
            setattr(result, attr, _read_table(path("correlation", name), "ID"))
    if os.path.isfile(path(*_MEAN_MATRIX_FILE)):
        result.mean_matrix = _read_table(path(*_MEAN_MATRIX_FILE), "label")
    levels = [lv for lv in params["levels"] if os.path.isfile(path("global", f"{lv}.tsv"))]
    if levels:
        result.global_df = pd.concat({lv: _read_table(path("global", f"{lv}.tsv"), "ID") for lv in levels}, axis=1,
                                     names=["level", "metric"])
    curves = [_read_table(path(lv, "curves.tsv"), text=("ID", "node")).assign(level=lv) for lv in params["levels"]
              if os.path.isfile(path(lv, "curves.tsv"))]
    if curves:
        result.curves = pd.concat(curves, ignore_index=True)[["level", "ID", "metric", "node", "threshold", "value"]]
    if os.path.isfile(path("nodes.tsv")):
        options = params.get("options", {})
        result.nodes = _read_table(path("nodes.tsv"), text=(options.get("label_col", "label"),))
    return result
