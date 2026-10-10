"""
From matrices to graph metrics
==============================

Compute node-level and network-level graph metrics for a group of participants, then draw the group mean node strength on the brain.
"""

# %%
# Simulated data
# --------------
# ``make_mock_dataset`` returns one correlation matrix per participant on the Gordon atlas (333 regions), the node table and a participants table. Group B is more connected within the Default network and less within the Visual network.

from conngraph.datasets import make_mock_dataset

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
participants.head()

# %%
# The node table gives each region's network, hemisphere and MNI coordinates.

nodes.head()

# %%
# Graph metrics
# -------------
# ``compute_graph_metrics`` keeps the edges of each matrix that form a triangulated maximally filtered graph (TMFG) and computes the metrics on that graph. The node level has a row per participant and a column per metric and region.
#
# ``n_jobs=1`` keeps the computation in one process. With more workers, a script needs its code under ``if __name__ == "__main__":``, since each worker starts a fresh Python process that imports the script.

import conngraph

result = conngraph.compute_graph_metrics(
    matrices,
    nodes,
    network_col="network",
    graph_method="tmfg",
    metrics=["strength", "clust_coeff"],
    n_jobs=1,
    verbose=False,
)
result.node_df["strength.abs"].iloc[:5, :5].round(2)

# %%
# The network level averages the connectivity within and between networks in each hemisphere, which gives a graph with one node per network and hemisphere, and computes the same metrics on it.

result.net_hemi_df["strength.abs"].iloc[:5, :5].round(2)

# %%
# Strength on the brain
# ---------------------
# The group mean strength of each region goes into the node table as a new column, which ``plot_views`` uses for the colour and the size of the nodes. Edges are drawn where the absolute group mean connectivity reaches ``edge_threshold``, here set to keep the strongest 0.2% of the connections.

import numpy as np

mean_strength = result.node_df["strength.abs"].mean()
shown = nodes.assign(strength=mean_strength[nodes["label"]].to_numpy())
mean = result.mean_matrix.to_numpy()
threshold = np.quantile(np.abs(mean[np.triu_indices(len(mean), 1)]), 0.998)
left, right = conngraph.get_fsLR_surface()

plotter = conngraph.BrainNetPlotter(conngraph.load(result.mean_matrix, shown))
fig = plotter.plot_views(
    node_color="strength",
    node_size="strength",
    node_cmap="plasma",
    node_size_range=(1.0, 5.0),
    edge_threshold=threshold,
    surface_L=left,
    surface_R=right,
    legend=["node_color"],
    width=10.5,
)
