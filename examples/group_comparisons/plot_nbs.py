"""
Network-based statistic
=======================

Find connected sets of edges that differ between two groups, then draw the group means, their difference and the significant edges on the brain.
"""

# %%
# NBS tests every edge, so it takes the matrices themselves. The correlations are Fisher z-transformed first, as the ``conngraph`` command does.

import numpy as np
import pandas as pd

import conngraph
from conngraph.datasets import make_mock_dataset

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
group = dict(zip(participants["participant_id"], participants["group"]))
labels = nodes["label"].tolist()

z = {}
for sid, matrix in matrices.items():
    values = np.arctanh(np.clip(matrix.to_numpy(), -0.999999, 0.999999))
    np.fill_diagonal(values, 0)
    z[sid] = pd.DataFrame(values, index=labels, columns=labels)
group_b = {sid: m for sid, m in z.items() if group[sid] == "B"}
group_a = {sid: m for sid, m in z.items() if group[sid] == "A"}

# %%
# Running NBS
# -----------
# The edges whose t exceeds ``thresh`` form connected components, and the size of each is compared with the largest component found in ``k`` permutations of the group labels. The simulated difference is large, so a high threshold keeps the component small enough to draw.

nbs = conngraph.run_nbs(group_b, group_a, thresh=5.0, k=1000, seed=0, n_jobs=1, verbose=False)
components = np.flatnonzero(nbs.pval < 0.05) + 1
significant = np.isin(nbs.adj, components)
print(f"{len(components)} significant component(s) with {int(np.triu(significant, 1).sum())} edges")

# %%
# Group means and their difference
# --------------------------------
# ``plot_nbs_matrices`` draws the two group means and their difference, with the significant edges marked.

fig = conngraph.plot_nbs_matrices(
    nbs.mean_g1,
    nbs.mean_g2,
    significant,
    labels,
    group_names=("B", "A"),
    network_labels=nodes["network"].tolist(),
)

# %%
# Significant edges on the brain
# ------------------------------
# The difference of the significant edges as a matrix, drawn with the nodes coloured by network.

difference = np.where(significant, nbs.mean_g1 - nbs.mean_g2, 0)
left, right = conngraph.get_fsLR_surface()
plotter = conngraph.BrainNetPlotter(conngraph.load(pd.DataFrame(difference, index=labels, columns=labels), nodes))
fig = plotter.plot_views(
    node_color="network",
    edge_threshold=1e-12,
    edge_color="weight",
    surface_L=left,
    surface_R=right,
    legend=["node_color", "edge_color"],
    legend_titles={"edge_color": "B - A"},
    width=10.5,
)

# sphinx_gallery_thumbnail_number = 2
