"""
Signed null-model networks for normalizing graph metrics.
"""

from __future__ import annotations

import numpy as np

_BATCH = 10  # weights handed out between re-rankings


def randomize_signed(A: np.ndarray, swaps: float = 10, rng=None) -> np.ndarray:
    """Random network keeping each node's positive and negative degree; weights are redistributed within each sign."""
    rng = np.random.default_rng(rng)
    upper = np.sign(np.triu(A, 1)).astype(np.int8)
    classes = upper + upper.T
    _swap_classes(classes, swaps, rng)
    R = np.zeros(A.shape)
    for sign in (1, -1):
        _assign_weights(R, A, classes, sign, rng)
    return R + R.T


def _swap_classes(classes: np.ndarray, swaps: float, rng: np.random.Generator) -> None:
    # Exchange the classes (positive, negative, none) of pairs a-b, c-d with those of a-d, c-b, so every node keeps its degree per class
    rows, cols = np.nonzero(np.triu(classes, 1))
    ends_a, ends_b = rows.tolist(), cols.tolist()
    n_edges = len(ends_a)
    if n_edges < 2:
        return
    C = classes.tolist()
    target = int(round(swaps * n_edges))
    done = tries = 0
    while done < target and tries < 10 * target:
        picks = rng.integers(n_edges, size=(2, 4096)).tolist()
        flips = (rng.random(4096) < 0.5).tolist()
        for x, y, flip in zip(picks[0], picks[1], flips):
            tries += 1
            a, b, c, d = ends_a[x], ends_b[x], ends_a[y], ends_b[y]
            if flip:
                c, d = d, c
            if a == c or a == d or b == c or b == d:
                continue
            old, new = C[a][b], C[a][d]
            if C[c][d] != old or C[c][b] != new or new == old:
                continue
            C[a][b] = C[b][a] = C[c][d] = C[d][c] = new
            C[a][d] = C[d][a] = C[c][b] = C[b][c] = old
            if new == 0:  # the edges moved; otherwise only their signs changed
                ends_b[x], ends_a[y], ends_b[y] = d, c, b
            done += 1
            if done == target:
                break
    classes[:] = C


def _assign_weights(R: np.ndarray, A: np.ndarray, classes: np.ndarray, sign: int, rng: np.random.Generator) -> None:
    # Match weights to edges by rank: larger weights go where both ends still lack more of their original strength
    W = np.where(np.sign(A) == sign, np.abs(A), 0.0)
    weights = np.sort(W[np.triu(W != 0, 1)])
    rows, cols = np.nonzero(np.triu(classes == sign, 1))
    lacking = W.sum(axis=1)
    remaining = np.arange(rows.size)
    while remaining.size:
        need = np.clip(lacking, 0, None)
        ranked = remaining[np.argsort(need[rows[remaining]] * need[cols[remaining]], kind="stable")]
        picks = rng.choice(ranked.size, size=min(_BATCH, ranked.size), replace=False)
        chosen, w = ranked[picks], weights[picks]
        R[rows[chosen], cols[chosen]] = sign * w
        np.subtract.at(lacking, rows[chosen], w)
        np.subtract.at(lacking, cols[chosen], w)
        keep = np.ones(ranked.size, dtype=bool)
        keep[picks] = False
        remaining, weights = ranked[keep], weights[keep]
