from __future__ import annotations

import numpy as np
import networkx as nx


def threshold_graph(matrix: np.ndarray, threshold: float, use_abs: bool = False) -> nx.Graph:
    """
    Threshold a correlation matrix by a fixed value.

    Parameters
    ----------
    matrix    : symmetric (N, N) matrix; diagonal is ignored.
    threshold : keep edges where weight > threshold (or |weight| > threshold
                when use_abs=True).
    use_abs   : if True, threshold on |weight|, retaining both positive and
                negative correlations whose magnitude exceeds the cutoff.
    """
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    G = nx.from_numpy_array(mat)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_edges_from([
        (u, v) for u, v, a in G.edges(data=True)
        if (abs(a["weight"]) if use_abs else a["weight"]) <= threshold
    ])
    return G


def density_graph(matrix: np.ndarray, density: float) -> nx.Graph:
    """Keep the strongest `density` fraction of node pairs (by signed weight) from a symmetric matrix."""
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")

    n = matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    weights = matrix[rows, cols]
    n_keep = int(round(density * weights.size))

    strongest = np.argsort(-weights, kind="stable")[:n_keep]
    strongest = strongest[weights[strongest] != 0]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(zip(rows[strongest].tolist(), cols[strongest].tolist(), weights[strongest].tolist()))
    return G


def detect_communities(
    G: nx.Graph,
    method: str = "louvain",
    seed: int = 42,
) -> list[int]:
    """
    Detect communities in a graph and return a per-node integer label.

    Parameters
    ----------
    G      : nx.Graph (weighted edges recommended for better results).
    method : "louvain" (default) | "greedy" | "label_propagation".
    seed   : random seed (used by louvain and label_propagation).

    Returns
    -------
    List of length N where each value is a 0-indexed community id.
    Pass this list as node_colors (via a colormap) to spring_plot_3d, or
    set it as a column in nodes_df for use with BrainNetPlotter.

    Example
    -------
    >>> community_ids = detect_communities(G)
    >>> spring_plot_3d(G, node_colors=labels_to_colors(community_ids))
    """
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    if method == "louvain":
        communities = nx.community.louvain_communities(G, seed=seed)
    elif method == "greedy":
        communities = nx.community.greedy_modularity_communities(G)
    elif method == "label_propagation":
        communities = nx.community.label_propagation_communities(G)
    else:
        raise ValueError(f"method='{method}' not recognised. Choose: louvain, greedy, label_propagation.")

    labels = [0] * len(nodes)
    for community_id, members in enumerate(communities):
        for node in members:
            labels[node_index[node]] = community_id
    return labels
