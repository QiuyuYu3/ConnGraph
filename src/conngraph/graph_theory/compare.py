"""
Two-group comparison of graph metrics, network blocks and edges with mass-univariate permutation tests.
"""

from __future__ import annotations

import platform
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

COMPARISONS = ("metrics", "blocks", "edges")
# Result attribute, file label and row name of each level
_METRIC_LEVELS = (("node_df", "node", "node"), ("network_df", "network", "network"),
                  ("net_hemi_df", "networkhemi", "network"))
_BLOCK_LEVELS = (("net_corr_df", "network"), ("net_hemi_corr_df", "networkhemi"))
_GLOBAL_LEVELS = {"node": "node", "network": "network", "network_hemi": "networkhemi"}
_STATS = ["t", "p", "p_fdr", "p_fwe", "mean_group1", "mean_group2", "n_group1", "n_group2"]


@dataclass
class GroupComparisonResult:
    """Output of compare_groups(): one table per family of tests, and the parameters of the run."""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def permuted_t_test(values: pd.DataFrame, group1: pd.Series, covariates: pd.DataFrame | None = None,
                    n_perms: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Group 1 vs group 2 for each column, one family: t, parametric p, FDR p and max-T permutation FWE p."""
    from nilearn.mass_univariate import permuted_ols
    from scipy import stats

    X = values.to_numpy(dtype=float)
    g = group1.reindex(values.index).to_numpy(dtype=bool)
    conf = None if covariates is None or covariates.shape[1] == 0 else covariates.reindex(values.index).to_numpy(float)
    dof = len(g) - 2 - (0 if conf is None else conf.shape[1])
    if dof < 1:
        raise ValueError(f"{len(g)} participants are too few for the groups and {len(g) - 2 - dof} covariate column(s)")
    out = pd.DataFrame(np.nan, index=values.columns, columns=_STATS)
    out["mean_group1"], out["mean_group2"] = X[g].mean(axis=0), X[~g].mean(axis=0)
    out["n_group1"], out["n_group2"] = int(g.sum()), int((~g).sum())
    with np.errstate(invalid="ignore"):
        # a constant column has no variance to test, and permuted_ols returns rounding noise for it
        tested = ~np.isnan(X).any(axis=0) & (np.ptp(X, axis=0) > 1e-12 * np.abs(X).max(axis=0))
    if tested.any():
        # one job: nilearn's permutations change with the job count
        res = permuted_ols(g[:, None].astype(float), X[:, tested], conf, model_intercept=True, n_perm=n_perms,
                           two_sided_test=True, random_state=seed, n_jobs=1, verbose=0)
        t = res["t"][0]
        p = 2 * stats.t.sf(np.abs(t), dof)
        out.loc[tested, "t"], out.loc[tested, "p"] = t, p
        out.loc[tested, "p_fdr"] = stats.false_discovery_control(p)
        out.loc[tested, "p_fwe"] = 10 ** -res["logp_max_t"][0]
    out[["n_group1", "n_group2"]] = out[["n_group1", "n_group2"]].astype(int)
    return out


def compare_groups(
    groups: Mapping[str, str],
    contrast: Sequence[str],
    metrics=None,
    matrices: Mapping[str, pd.DataFrame] | None = None,
    compare: Sequence[str] = ("metrics", "blocks"),
    covariates: pd.DataFrame | None = None,
    n_perms: int = 5000,
    seed: int | None = None,
    apply_fisher_z: bool = True,
    verbose: bool = True,
) -> GroupComparisonResult:
    """
    Compare two groups on graph metrics, network blocks and edges, with each family corrected on its own.

    Parameters
    ----------
    groups       : {participant ID: group name}, keyed as in metrics and matrices.
    contrast     : (group 1, group 2); a positive t means group 1 is higher.
    metrics      : GraphMetricsResult for "metrics" (each metric at each level, and the whole-graph metrics of each
                   level) and "blocks" (connectivity within and between network nodes).
    matrices     : {participant ID: N×N DataFrame} for "edges".
    compare      : any of "metrics", "blocks", "edges".
    covariates   : participants × covariates; text columns become indicator columns, and participants with a
                   missing value are left out.
    n_perms      : permutations for the family-wise p-values.
    seed         : random seed; when None, the seed drawn is recorded in ``result.params``.
    apply_fisher_z : Fisher z-transform the edges before testing.

    Returns
    -------
    GroupComparisonResult with tables "metrics_<level>", "global_<level>", "blocks_<level>" and "edges", each with
    t, p, p_fdr, p_fwe, the group means and sizes; columns with missing or constant values are not tested.
    """
    from conngraph.graph_theory.runner import _jsonable, _package_versions

    compare = list(dict.fromkeys(compare))
    unknown = [c for c in compare if c not in COMPARISONS]
    if unknown or not compare:
        raise ValueError(f"compare must name some of {', '.join(COMPARISONS)}, got {compare}")
    if len(contrast) != 2 or contrast[0] == contrast[1]:
        raise ValueError(f"contrast must name two different groups, got {list(contrast)}")
    if {"metrics", "blocks"} & set(compare) and metrics is None:
        raise ValueError("comparing metrics or blocks needs a graph metrics result")
    if "edges" in compare and matrices is None:
        raise ValueError("comparing edges needs the matrices")
    sources = []
    if "metrics" in compare or "blocks" in compare:
        sources.append(set(_metric_ids(metrics)))
    if "edges" in compare:
        sources.append(set(matrices))
    chosen = [sid for sid, name in groups.items() if name in contrast]
    no_data = [sid for sid in chosen if not all(sid in s for s in sources)]
    ids = [sid for sid in chosen if sid not in no_data]
    if no_data:
        warnings.warn(f"Leaving out {len(no_data)} participant(s) without data: {', '.join(map(str, no_data))}",
                      stacklevel=2)

    design, columns, missing_cov = _covariates(covariates, ids)
    if missing_cov:
        warnings.warn(f"Leaving out {len(missing_cov)} participant(s) with a missing covariate: "
                      f"{', '.join(map(str, missing_cov))}", stacklevel=2)
        ids = [sid for sid in ids if sid not in missing_cov]
    group1 = pd.Series([groups[sid] == contrast[0] for sid in ids], index=ids)
    for name, inside in zip(contrast, (group1, ~group1)):
        if inside.sum() < 1:
            raise ValueError(f"no participant left in group {name!r}")
    if design is not None:
        full = np.column_stack([group1.to_numpy(float), np.ones(len(ids)), design.loc[ids].to_numpy(float)])
        if np.linalg.matrix_rank(full) < full.shape[1]:
            raise ValueError("the covariates are constant, or collinear with the groups or each other: "
                             f"{', '.join(columns)}")
        design = design.loc[ids]
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])  # a seed that can be recorded and passed back

    families = []
    if "metrics" in compare:
        families += _metric_families(metrics)
        if not families:
            raise ValueError("the graph metrics result holds no metrics")
    if "blocks" in compare:
        blocks = _block_families(metrics)
        if not blocks:
            raise ValueError("comparing blocks needs the network level")
        families += blocks
    if "edges" in compare:
        families.append(_edge_family(matrices, ids, apply_fisher_z))

    tables, untested = {}, {}
    for name, rows, wide in families:
        if verbose:
            print(f"Comparing {name}: {wide.shape[1]} tests, {n_perms} permutations")
        table = permuted_t_test(wide.loc[ids], group1, design, n_perms=n_perms, seed=seed)
        skipped = rows[table["t"].isna().to_numpy()]
        if len(skipped):
            untested.setdefault(name, []).extend(" / ".join(map(str, r)) for r in skipped.itertuples(index=False))
        tables.setdefault(name, []).append(pd.concat([rows.reset_index(drop=True), table.reset_index(drop=True)],
                                                     axis=1))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}

    params = _jsonable({
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "options": {"compare": compare, "n_perms": n_perms, "seed": seed, "apply_fisher_z": apply_fisher_z,
                    "covariates": [] if covariates is None else list(map(str, covariates.columns))},
        "contrast": list(contrast),
        "groups": {"g1": list(group1.index[group1]), "g2": list(group1.index[~group1])},
        "covariate_columns": columns,
        "left_out": {"no_data": no_data, "covariates": missing_cov},
        "untested": untested,
    })
    return GroupComparisonResult(tables=tables, params=params)


def _metric_ids(metrics) -> list[str]:
    for attr in ("node_df", "network_df", "net_hemi_df", "global_df", "net_corr_df", "net_hemi_corr_df"):
        df = getattr(metrics, attr, None)
        if df is not None:
            return list(df.index)
    return []


def _covariates(covariates: pd.DataFrame | None, ids: list[str]) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Numeric design columns (text columns as indicators), their names, and participants with a missing value."""
    if covariates is None or covariates.shape[1] == 0:
        return None, [], []
    absent = [sid for sid in ids if sid not in covariates.index]
    table = covariates.reindex(ids)
    missing = [sid for sid in ids if sid in absent or table.loc[sid].isna().any()]
    table = table.drop(index=missing)
    parts = []
    for col in table.columns:
        numbers = pd.to_numeric(table[col], errors="coerce")
        if numbers.notna().all():
            parts.append(numbers.rename(str(col)).astype(float))
        else:
            parts.append(pd.get_dummies(table[col].astype(str), prefix=str(col), drop_first=True, dtype=float))
    design = pd.concat(parts, axis=1)
    return design, list(design.columns), missing


def _metric_families(metrics) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    families = []
    for attr, name, row in _METRIC_LEVELS:
        df = getattr(metrics, attr, None)
        if df is None:
            continue
        for metric in dict.fromkeys(df.columns.get_level_values(0)):
            wide = df[metric]
            rows = pd.DataFrame({"metric": metric, row: [str(c) for c in wide.columns]})
            families.append((f"metrics_{name}", rows, wide))
    if metrics.global_df is not None:
        for level in dict.fromkeys(metrics.global_df.columns.get_level_values(0)):
            wide = metrics.global_df[level]
            families.append((f"global_{_GLOBAL_LEVELS[level]}", pd.DataFrame({"metric": list(wide.columns)}), wide))
    return families


def _block_families(metrics) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    families = []
    for attr, name in _BLOCK_LEVELS:
        wide = getattr(metrics, attr, None)
        if wide is not None:
            pairs = [col.split("__") for col in wide.columns]
            rows = pd.DataFrame(pairs, columns=["network_a", "network_b"])
            families.append((f"blocks_{name}", rows, wide))
    return families


def _edge_family(matrices: Mapping[str, pd.DataFrame], ids: list[str], fisher: bool):
    labels = list(matrices[ids[0]].columns)
    iu = np.triu_indices(len(labels), 1)
    values = np.vstack([matrices[sid].loc[labels, labels].to_numpy(dtype=float)[iu] for sid in ids])
    if fisher:
        values = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
    rows = pd.DataFrame({"roi_a": [labels[i] for i in iu[0]], "roi_b": [labels[j] for j in iu[1]]})
    return "edges", rows, pd.DataFrame(values, index=ids)
