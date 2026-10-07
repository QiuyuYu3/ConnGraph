"""
Single-subject graph-theory metrics via TMFG + BCT.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

METRIC_NAMES = ("clust_coeff", "btwn_cent", "strength", "ge_local")


def process_subject(
    subj: str,
    corrmat: np.ndarray,
    node_labels: list,
    metrics: list[str],
    graph_method: str | Callable = "tmfg",
) -> dict | None:
    """Compute graph-theory metrics for one subject.

    Parameters
    ----------
    subj : subject identifier (used in error messages only)
    corrmat : square numpy array; diagonal is set to 0 internally
    node_labels : ROI or network names corresponding to corrmat rows/cols
    metrics : any subset of {"clust_coeff", "btwn_cent", "strength", "ge_local"}
    graph_method : "tmfg" (default) or any callable that takes a square
        numpy array and returns a networkx Graph.  Use this to plug in
        alternative sparsification algorithms without modifying the package.

        Example::

            def my_method(corrmat):
                import networkx as nx
                # ... build and return a nx.Graph
                return G

            results = compute_graph_metrics(..., graph_method=my_method)

    Returns
    -------
    dict with "subj" key plus one {label: value} dict per metric, or None on failure.
    """
    import bct
    import networkx as nx

    try:
        mat = corrmat.copy()
        np.fill_diagonal(mat, 0)

        A = nx.to_numpy_array(_build_graph(mat, graph_method))

        out: dict = {"subj": subj}

        if "clust_coeff" in metrics:
            try:
                clust = bct.clustering_coef_wu_sign(A, coef_type="costantini")
            except Exception as e:
                print(f"[{subj}] clustering_coef_wu_sign failed: {e}")
                clust = np.full(len(node_labels), np.nan)
            out["clust_coeff"] = pd.Series(clust, index=node_labels).fillna(0).to_dict()

        if "btwn_cent" in metrics:
            A_dist = 1.0 / (np.abs(A) + 1e-6)
            out["btwn_cent"] = pd.Series(
                bct.betweenness_wei(A_dist), index=node_labels
            ).to_dict()

        if "strength" in metrics:
            out["strength"] = pd.Series(
                bct.strengths_und(np.abs(A)), index=node_labels
            ).to_dict()

        if "ge_local" in metrics:
            out["ge_local"] = pd.Series(
                bct.efficiency_wei(np.abs(A), local=True), index=node_labels
            ).to_dict()

        return out

    except Exception as e:
        print(f"[{subj}] failed: {e}")
        return None


def _build_graph(corrmat: np.ndarray, method: str | Callable):
    """Return a networkx Graph from a correlation matrix using the given method."""
    if callable(method):
        return method(corrmat)

    if method == "tmfg":
        import collections
        import collections.abc
        if not hasattr(collections, "Sized"):
            collections.Sized = collections.abc.Sized  # topcorr still uses the alias removed in Python 3.10
        import topcorr as tpc
        return tpc.tmfg(corrmat, absolute=True, threshold_mean=True)

    raise ValueError(
        f"graph_method='{method}' is not recognised. "
        "Pass \"tmfg\" or a callable that takes a corrmat and returns a nx.Graph."
    )
