"""
Connectivity matrices by network
================================

Draw one participant's connectivity matrix and the group mean, with the regions ordered and coloured by network.
"""

# %%
# ``load_group`` collects the simulated matrices and the node table into one dataset; ``mean_matrix`` averages the participants' correlations.

import conngraph
from conngraph.datasets import make_mock_dataset

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
dataset = conngraph.load_group(matrices, nodes)
labels = nodes["label"].tolist()
networks = nodes["network"].tolist()

# %%
# One participant
# ---------------
# ``matrix_heatmap`` orders the regions by network, draws a line between networks and a coloured bar beside each.

fig, ax = conngraph.matrix_heatmap(
    dataset.get_matrix("sub-01"),
    labels,
    networks,
    vmin=-0.6,
    vmax=0.6,
    title="sub-01",
)

# %%
# Group mean
# ----------

fig, ax = conngraph.matrix_heatmap(
    dataset.mean_matrix().to_numpy(),
    labels,
    networks,
    vmin=-0.6,
    vmax=0.6,
    title="Group mean",
)
