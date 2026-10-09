from __future__ import annotations

import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    # result files this was read from, for the report to link
    files: dict = field(default_factory=dict)

    def save_report(self, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                    static_brain: bool = True, label_col: str = "label", network_col: str = "network",
                    interactive_brain: bool = False) -> None:
        """Write an HTML report (plotly loads from its CDN); nodes adds networks and, with x, y, z, brain figures."""
        from conngraph.report.pages import save_nbs_report

        save_nbs_report(self, path, nodes, surfaces, static_brain, label_col, network_col, interactive_brain)


def run_nbs(
    matrices_g1: dict[str, pd.DataFrame],
    matrices_g2: dict[str, pd.DataFrame],
    thresh: float,
    k: int = 1000,
    tail: str = "both",
    paired: bool = False,
    seed: int | None = None,
    verbose: bool = True,
    n_jobs: int = -1,
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
    n_jobs   : worker processes for the permutations; -1 = cpu_count - 1. Results do not depend on it.

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
    # every draw is made here in bct's order, so the worker count cannot change the result
    rng = get_rng(seed)
    if paired:
        draws = np.array([np.sign(0.5 - rng.rand(1, nx))[0] for _ in range(k)])
    else:
        draws = np.array([rng.permutation(nx + ny) for _ in range(k)])
    # column-major so each permutation copies whole subject columns; the arithmetic, and so the result, is unchanged
    null = _null_distribution(np.asfortranarray(np.hstack((x_edges, y_edges))), draws, (nx, n, thresh, tail, paired),
                              max_size, n_jobs, verbose)

    pval = np.array([np.count_nonzero(null >= s) / k for s in sizes])
    from conngraph.graph_theory.runner import _jsonable, _package_versions

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


_MIN_PER_CHUNK = 25
_WORKER: dict = {}


def _null_distribution(pooled, draws, setup, observed_max, n_jobs, verbose) -> np.ndarray:
    k = len(draws)
    workers = max(1, (os.cpu_count() or 2) - 1) if n_jobs == -1 else max(1, n_jobs)
    chunk = max(_MIN_PER_CHUNK, -(-k // (4 * workers)))
    starts = list(range(0, k, chunk))
    workers = min(workers, len(starts))
    null = np.zeros(k)
    report = _Progress(k, observed_max, verbose)
    if workers == 1:
        _init_worker(pooled, setup)
        try:
            for s in starts:
                null[s:s + chunk] = _null_chunk(draws[s:s + chunk])
                report(null[:s + chunk])
        finally:
            _WORKER.clear()
        return null
    # spawn on every platform: forking a parent that already runs threads can deadlock
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(pooled, setup),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_null_chunk, draws[s:s + chunk]): s for s in starts}
        done = np.zeros(k, dtype=bool)
        for f in as_completed(futures):
            s = futures[f]
            null[s:s + chunk] = f.result()
            done[s:s + chunk] = True
            report(null[done])
    return null


class _Progress:
    """Prints about ten progress lines with the p-value of the largest component so far."""

    def __init__(self, k: int, observed_max: float, verbose: bool):
        self.k, self.observed_max, self.verbose, self.next = k, observed_max, verbose, 0

    def __call__(self, finished: np.ndarray) -> None:
        if not self.verbose or (finished.size < self.next and finished.size < self.k):
            return
        self.next = finished.size + max(self.k // 10, 1)
        p = np.count_nonzero(finished >= self.observed_max) / finished.size
        print(f"permutation {finished.size} of {self.k}.  p-value so far is {p:.3f}")


def _init_worker(pooled: np.ndarray, setup: tuple) -> None:
    _WORKER["pooled"], _WORKER["setup"] = pooled, setup


def _null_chunk(draws: np.ndarray) -> np.ndarray:
    from bct import get_components

    pooled = _WORKER["pooled"]
    nx, n, thresh, tail, paired = _WORKER["setup"]
    stat = _paired_t if paired else _two_sample_t
    iu = np.triu_indices(n, 1)
    out = np.zeros(len(draws))
    for i, draw in enumerate(draws):
        d = pooled * np.concatenate((draw, draw)) if paired else pooled[:, draw]
        sizes = _component_edge_counts(stat(d[:, :nx], d[:, nx:], tail) > thresh, iu, n, get_components)[1]
        out[i] = sizes.max() if sizes.size else 0
    return out


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
