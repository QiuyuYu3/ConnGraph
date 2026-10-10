"""
Brain views of each hemisphere
==============================

Draw the nodes coloured by network and the strongest connections of the group mean, from the lateral and the medial side of each hemisphere.
"""

# %%
# The group mean matrix gives the edges; ``edge_threshold`` keeps those whose absolute weight reaches it, here the strongest 0.2%.

import numpy as np

import conngraph
from conngraph.datasets import make_mock_dataset

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
dataset = conngraph.load_group(matrices, nodes)
mean = dataset.mean_matrix().to_numpy()
threshold = np.quantile(np.abs(mean[np.triu_indices(len(mean), 1)]), 0.998)
left, right = conngraph.get_fsLR_surface()

# %%
# Four panels
# -----------
# Each panel is a dict: ``"view"`` is the side the camera looks from and ``"hemisphere"`` keeps one hemisphere's nodes, surface and edges. A list of dicts is one row. Seen from the right, the left hemisphere shows its medial side; a more opaque surface (``surface_alpha``) fades the nodes behind it.

plotter = conngraph.BrainNetPlotter(conngraph.load(dataset.mean_matrix(), nodes))
fig = plotter.plot_views(
    views=[
        [{"view": "L", "hemisphere": "L"}, {"view": "R", "hemisphere": "L"}],
        [{"view": "R", "hemisphere": "R"}, {"view": "L", "hemisphere": "R"}],
    ],
    node_color="network",
    edge_threshold=threshold,
    surface_L=left,
    surface_R=right,
    surface_alpha=0.5,
    width=8,
)
