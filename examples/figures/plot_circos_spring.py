"""
Circos and spring plots
=======================

Draw the TMFG graph of the group mean around a circle, as chords and bundled through the networks, and in a spring layout grouped by network.
"""

# %%
# ``build_adjacency`` keeps the edges of the group mean matrix that form a triangulated maximally filtered graph (TMFG), the graph the metrics are computed on. It links every region, so it holds edges between networks as well as within them.

import networkx as nx

import conngraph
from conngraph.datasets import make_mock_dataset
from conngraph.graph_theory.sparsify import build_adjacency

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
mean = conngraph.load_group(matrices, nodes).mean_matrix().to_numpy()
graph = nx.from_numpy_array(build_adjacency(mean, "tmfg"))
labels = nodes["label"].tolist()
networks = nodes["network"].tolist()

# %%
# Circos plot
# -----------
# ``circos_plot`` returns two figures: chords coloured by weight, and the same edges bundled through their networks and coloured by the networks they join.

(chords, _), (bundled, _) = conngraph.circos_plot(graph, labels, networks, figsize=(8, 8), label_fontsize=3)

# %%
# Spring layout
# -------------
# With ``layout="network"`` each network gets its own disc, placed near the networks it shares the most weight with.

fig, ax = conngraph.spring_plot(graph, labels, networks, figsize=(8, 8), layout="network", label_fontsize=3)

# sphinx_gallery_thumbnail_number = 2
