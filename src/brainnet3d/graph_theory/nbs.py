from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class NBSResult:
    """Output of run_nbs()."""
    pval: np.ndarray
    adj:  np.ndarray
    null: np.ndarray
    labels: list[str] | None = None
    params: dict = field(default_factory=dict)
    mean_g1: np.ndarray | None = None
    mean_g2: np.ndarray | None = None

    def save_report(self, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                    static_brain: bool = True, label_col: str = "label", network_col: str = "network") -> None:
        """Write an HTML report (plotly loads from its CDN); nodes adds networks and, with x, y, z, brain figures."""
        from brainnet3d.report.pages import save_nbs_report

        save_nbs_report(self, path, nodes, surfaces, static_brain, label_col, network_col)


def run_nbs(
    matrices_g1: dict[str, pd.DataFrame],
    matrices_g2: dict[str, pd.DataFrame],
    thresh: float,
    k: int = 1000,
    tail: str = "both",
    paired: bool = False,
    seed: int | None = None,
    verbose: bool = True,
) -> NBSResult:
    """
    Run Network-Based Statistics comparing two groups of connectivity matrices.

    Parameters
    ----------
    matrices_g1, matrices_g2 : {subject_id: N×N DataFrame} for each group.
        Both groups must share the same ROI labels (index/columns).
    thresh   : t-statistic threshold for initial edge selection.
    k        : number of permutations.
    tail     : "both" | "left" | "right".
    paired   : True for paired t-test (groups must be the same size).
    seed     : random seed for reproducibility; when None, the seed drawn is recorded in ``result.params``.
    verbose  : print permutation progress.

    Returns
    -------
    NBSResult with pval, adj, null arrays, the ROI label list, params (options, groups, package versions) and the
    two group mean matrices; .save_report(path) writes an HTML report.
    """
    try:
        from bct import BCTParamError, get_components
        from bct.utils import get_rng
    except ImportError:
        raise ImportError("run_nbs needs bctpy: pip install bctpy") from None

    if tail not in ("both", "left", "right"):
        raise BCTParamError("Tail must be both, left, right")

    ids1 = list(matrices_g1.keys())
    ids2 = list(matrices_g2.keys())
    labels = matrices_g1[ids1[0]].columns.tolist()

    X = np.stack([matrices_g1[s].values.astype(float) for s in ids1], axis=2)
    Y = np.stack([matrices_g2[s].values.astype(float) for s in ids2], axis=2)

    n, nx, ny = X.shape[0], X.shape[2], Y.shape[2]
    if not X.shape[0] == X.shape[1] == Y.shape[0] == Y.shape[1]:
        raise BCTParamError("Population matrices are of inconsistent size")
    if paired and nx != ny:
        raise BCTParamError("Population matrices must be an equal size")

    # same edge order and random draws as bct.nbs_bct, with the t-test run on all edges at once
    iu = np.triu_indices(n, 1)
    x_edges, y_edges = X[iu], Y[iu]
    stat = _paired_t if paired else _two_sample_t

    above = stat(x_edges, y_edges, tail) > thresh
    if not above.any():
        raise BCTParamError("Unsuitable threshold")
    edge_comp, sizes = _component_edge_counts(above, iu, n, get_components)
    adj = np.zeros((n, n))
    adj[iu[0][above], iu[1][above]] = edge_comp + 1
    adj = adj + adj.T
    max_size = sizes.max()
    if verbose:
        print(f"max component size is {int(max_size)}")
        print(f"estimating null distribution with {k} permutations")

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])  # a seed that can be recorded and passed back
    rng = get_rng(seed)
    pooled = np.hstack((x_edges, y_edges))
    null = np.zeros(k)
    hits = 0
    for u in range(k):
        if paired:
            flip = np.sign(0.5 - rng.rand(1, nx))
            d = pooled * np.hstack((flip, flip))
        else:
            d = pooled[:, rng.permutation(nx + ny)]
        perm_sizes = _component_edge_counts(stat(d[:, :nx], d[:, nx:], tail) > thresh, iu, n, get_components)[1]
        null[u] = perm_sizes.max() if perm_sizes.size else 0
        hits += null[u] >= max_size
        if verbose and (u % max(k // 10, 1) == 0 or u == k - 1):
            print(f"permutation {u} of {k}.  p-value so far is {hits / (u + 1):.3f}")

    pval = np.array([np.count_nonzero(null >= s) / k for s in sizes])
    from brainnet3d.graph_theory.runner import _jsonable, _package_versions

    params = _jsonable({
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "options": {"thresh": thresh, "k": k, "tail": tail, "paired": paired, "seed": seed},
        "groups": {"g1": ids1, "g2": ids2},
        "n_nodes": n,
    })
    return NBSResult(pval=pval, adj=adj, null=null, labels=labels, params=params,
                     mean_g1=X.mean(axis=2), mean_g2=Y.mean(axis=2))


def _two_sample_t(a: np.ndarray, b: np.ndarray, tail: str) -> np.ndarray:
    n1, n2 = a.shape[1], b.shape[1]
    pooled_sd = np.sqrt(((n1 - 1) * a.var(axis=1, ddof=1) + (n2 - 1) * b.var(axis=1, ddof=1)) / (n1 + n2 - 2))
    denom = pooled_sd * np.sqrt(1 / n1 + 1 / n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom == 0, 0.0, (a.mean(axis=1) - b.mean(axis=1)) / denom)
    return _apply_tail(t, tail)


def _paired_t(a: np.ndarray, b: np.ndarray, tail: str) -> np.ndarray:
    diff = a - b
    n = diff.shape[1]
    sd = np.sqrt((np.sum(diff ** 2, axis=1) - np.sum(diff, axis=1) ** 2 / n) / (n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = diff.mean(axis=1) / sd * np.sqrt(n)
    return _apply_tail(t, tail)


def _apply_tail(t: np.ndarray, tail: str) -> np.ndarray:
    if tail == "both":
        return np.abs(t)
    return -t if tail == "left" else t


def _component_edge_counts(above: np.ndarray, iu: tuple, n: int, get_components) -> tuple[np.ndarray, np.ndarray]:
    """Component index of each suprathreshold edge and the edge count of each component."""
    graph = np.zeros((n, n))
    graph[iu[0][above], iu[1][above]] = 1
    comp, node_counts = get_components(graph + graph.T)
    kept = np.flatnonzero(node_counts > 1) + 1
    position = np.full(node_counts.size + 1, -1)
    position[kept] = np.arange(kept.size)
    edge_comp = position[comp[iu[0][above]]]
    return edge_comp, np.bincount(edge_comp, minlength=kept.size).astype(float)
