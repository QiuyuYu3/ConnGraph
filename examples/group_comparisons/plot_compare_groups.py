"""
Comparing two groups
====================

Compare two groups on node strength and on the connectivity within and between networks, adjusting for age.
"""

# %%
# The comparison starts from the graph metrics of every participant, computed as in the first example; here only strength.

import conngraph
from conngraph.datasets import make_mock_dataset

matrices, nodes, participants = make_mock_dataset(n_per_group=10)
result = conngraph.compute_graph_metrics(
    matrices,
    nodes,
    network_col="network",
    graph_method="tmfg",
    metrics=["strength"],
    n_jobs=1,
    verbose=False,
)

# %%
# Permutation t-tests
# -------------------
# ``compare_groups`` takes each participant's group and the two groups to compare; a positive t means the first is higher. ``"metrics"`` tests each metric at each level and ``"blocks"`` the mean connectivity within and between networks. Each family of tests is corrected on its own, and ``correction`` picks the p-value that marks a result significant.

groups = dict(zip(participants["participant_id"], participants["group"]))
covariates = participants.set_index("participant_id")[["age"]]
comparison = conngraph.compare_groups(
    groups,
    ("B", "A"),
    result,
    compare=["metrics", "blocks"],
    covariates=covariates,
    n_perms=1000,
    seed=0,
    correction="fdr",
    verbose=False,
)
list(comparison.tables)

# %%
# Each table has a row per test. Strength at the network level, smallest p-values first:

columns = ["network", "t", "p_fdr", "mean_group1", "mean_group2", "significant"]
comparison.tables["metrics_networkhemi"].sort_values("p")[columns].head().round(3)

# %%
# Network blocks
# --------------
# The block table has a row per pair of networks. Laid out as a matrix, its t values show where group B is more connected (red) or less (blue).

blocks = comparison.tables["blocks_networkhemi"]
t = blocks.pivot(index="network_a", columns="network_b", values="t")
t = t.combine_first(t.T)
fig, ax = conngraph.matrix_heatmap(t.to_numpy(), t.index.tolist(), order=None, vmin=-8, vmax=8, title="t, B vs A")
