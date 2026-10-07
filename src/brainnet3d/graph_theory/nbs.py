from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NBSResult:
    """Output of run_nbs()."""
    pval: np.ndarray
    adj:  np.ndarray
    null: np.ndarray
    labels: list[str] | None = None


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
    seed     : random seed for reproducibility.
    verbose  : print permutation progress.

    Returns
    -------
    NBSResult with pval, adj, null arrays and the ROI label list.
    """
    try:
        from bct import nbs_bct
    except ImportError:
        raise ImportError("run_nbs needs bctpy: pip install 'brainnet3d[graph]'") from None

    ids1 = list(matrices_g1.keys())
    ids2 = list(matrices_g2.keys())
    labels = matrices_g1[ids1[0]].columns.tolist()

    X = np.stack([matrices_g1[s].values.astype(float) for s in ids1], axis=2)
    Y = np.stack([matrices_g2[s].values.astype(float) for s in ids2], axis=2)

    pval, adj, null = nbs_bct(
        X, Y, thresh,
        k=k, tail=tail, paired=paired,
        verbose=verbose, seed=seed,
    )
    return NBSResult(pval=pval, adj=adj, null=null, labels=labels)
