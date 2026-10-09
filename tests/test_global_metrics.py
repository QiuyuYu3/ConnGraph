import bct
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conngraph.graph_theory import compute_graph_metrics
from conngraph.graph_theory.metrics import check_options, compute_metric, modularity_q, parse_metrics, process_subject
from conngraph.graph_theory.sparsify import build_adjacency


def _corr(n=30, n_networks=3, seed=0):
    rng = np.random.default_rng(seed)
    nets = np.arange(n) % n_networks
    factors = rng.standard_normal((200, n_networks))
    factors[:, -1] = -factors[:, 0] + rng.standard_normal(200)  # anticorrelated networks give negative edges
    ts = factors[:, nets] + rng.standard_normal((200, n))
    c = np.corrcoef(ts.T)
    np.fill_diagonal(c, 0)
    return c, nets


def _signed_graph(seed=0):
    c, nets = _corr(seed=seed)
    return build_adjacency(c, "density", {"density": 0.3}, False), nets


def _positive_graph(seed=0):
    A, nets = _signed_graph(seed)
    return np.clip(A, 0, None), nets


def _disconnected(seed=0):
    A, nets = _positive_graph(seed)
    A[:5, 5:] = A[5:, :5] = 0
    return A, nets


def _bct_lengths(W):
    return bct.distance_wei(bct.invert(W))[0]


def test_global_efficiency_matches_bct():
    W, _ = _positive_graph()
    assert compute_metric("eff_global.wei", W)[0] == pytest.approx(bct.efficiency_wei(W))
    assert compute_metric("eff_global.bin", W)[0] == pytest.approx(bct.efficiency_bin((W > 0).astype(float)))


@pytest.mark.parametrize("graph", [_positive_graph, _disconnected])
def test_path_length_averages_the_connected_pairs_as_bct(graph):
    W, _ = graph()
    expected = bct.charpath(_bct_lengths(W), include_infinite=False)[0]
    assert compute_metric("char_path.wei", W)[0] == pytest.approx(expected)
    binary = bct.charpath(bct.distance_bin((W > 0).astype(float)), include_infinite=False)[0]
    assert compute_metric("char_path.bin", W)[0] == pytest.approx(binary)


def test_nodal_efficiency_averages_to_global_efficiency():
    W, _ = _disconnected()
    nodal = compute_metric("eff_nodal.wei", W)
    assert nodal.shape == (len(W),) and nodal.mean() == pytest.approx(bct.efficiency_wei(W))


@pytest.mark.parametrize("graph", [_positive_graph, _disconnected])
def test_closeness_matches_networkx(graph):
    W, _ = graph()
    G = nx.from_numpy_array(np.divide(1, W, out=np.zeros_like(W), where=W > 0))
    expected = nx.closeness_centrality(G, distance="weight", wf_improved=False)
    assert np.allclose(compute_metric("close_cent.wei", W), [expected[i] for i in range(len(W))])
    binary = nx.closeness_centrality(nx.from_numpy_array((W > 0).astype(int)), wf_improved=False)
    assert np.allclose(compute_metric("close_cent.bin", W), [binary[i] for i in range(len(W))])


def test_eigenvector_centrality_matches_bct():
    W, _ = _positive_graph()
    assert np.allclose(compute_metric("eig_cent.wei", W), bct.eigenvector_centrality_und(W))
    B = (W > 0).astype(float)
    assert np.allclose(compute_metric("eig_cent.bin", W), bct.eigenvector_centrality_und(B))


def test_mean_clustering_is_the_mean_of_the_node_values():
    A, _ = _signed_graph()
    for variant in ("costantini", "zhang", "onnela", "bin"):
        assert compute_metric(f"clust_mean.{variant}", A)[0] == pytest.approx(
            compute_metric(f"clust_coeff.{variant}", A).mean())


def test_participation_matches_bct_on_the_given_networks():
    A, nets = _signed_graph()
    pos, neg = bct.participation_coef_sign(A, nets + 1)
    assert np.allclose(compute_metric("participation.pos.networks", A, modules=nets), pos)
    assert np.allclose(compute_metric("participation.neg.networks", A, modules=nets), neg)


def test_module_z_uses_the_sample_standard_deviation():
    W, nets = _positive_graph()
    within = np.array([W[i, nets == nets[i]].sum() for i in range(len(W))])
    frame = pd.DataFrame({"k": within, "m": nets})
    expected = (frame["k"] - frame.groupby("m")["k"].transform("mean")) / frame.groupby("m")["k"].transform("std")
    assert np.allclose(compute_metric("module_z.abs.networks", W, modules=nets), expected)


@pytest.mark.parametrize("graph", [_signed_graph, _positive_graph])
def test_modularity_of_a_partition_matches_bct_louvain(graph):
    A, _ = graph()
    kind = "negative_asym" if (A < 0).any() else "modularity"
    for seed in range(3):
        ci, q = bct.community_louvain(A, B=kind, seed=seed)
        assert modularity_q(A, ci) == pytest.approx(q)


def test_louvain_keeps_the_best_of_many_runs_and_is_reproducible():
    A, nets = _signed_graph()
    first = compute_metric("modularity.wei.louvain", A, seed=1)[0]
    assert compute_metric("modularity.wei.louvain", A, seed=1)[0] == first
    best_single = max(bct.community_louvain(A, B="negative_asym", seed=s)[1] for s in range(5))
    assert first >= best_single - 1e-9
    assert first >= compute_metric("modularity.wei.networks", A, modules=nets)[0]


def test_unassigned_nodes_form_one_module_and_get_no_module_values():
    A, nets = _signed_graph()
    modules = nets.astype(float)
    modules[:4] = np.nan
    for name in ("participation.pos.networks", "module_z.abs.networks"):
        values = compute_metric(name, A, modules=modules)
        assert np.isnan(values[:4]).all() and np.isfinite(values[4:]).all()
    together = np.where(np.isnan(modules), 99, modules)
    assert compute_metric("modularity.wei.networks", A, modules=modules)[0] == pytest.approx(modularity_q(A, together))


def test_signed_fallback_decides_how_unsigned_metrics_see_negative_weights():
    A, _ = _signed_graph()
    assert (A < 0).any()
    assert compute_metric("eff_global.wei", A)[0] == pytest.approx(bct.efficiency_wei(np.abs(A)))
    positive = compute_metric("eff_global.wei", A, signed_fallback="positive")[0]
    assert positive == pytest.approx(bct.efficiency_wei(np.clip(A, 0, None)))
    assert np.allclose(compute_metric("btwn_cent.inv", A, signed_fallback="positive"),
                       bct.betweenness_wei(bct.invert(np.clip(A, 0, None))))


def test_defaults_include_the_new_metrics():
    names = parse_metrics(None)
    for name in ("eff_nodal.wei", "participation.pos", "module_z.abs", "eig_cent.wei", "close_cent.wei",
                 "eff_global.wei", "char_path.wei", "clust_mean.costantini", "modularity.wei", "small_world.wei"):
        assert name in names


def test_partition_metrics_are_named_after_each_partition():
    names = check_options(["participation", "modularity", "strength"], "tmfg", None, None, "auc")
    assert names == ["participation.pos.networks", "participation.pos.louvain", "modularity.wei.networks",
                     "modularity.wei.louvain", "strength.abs"]
    only = check_options(["participation"], "tmfg", None, None, "auc", partitions=("louvain",))
    assert only == ["participation.pos.louvain"]


def test_network_level_leaves_out_partition_metrics():
    names = check_options(["participation", "eff_global"], "tmfg", None, None, "auc", level="network")
    assert names == ["eff_global.wei"]
    with pytest.raises(ValueError, match="node level"):
        check_options(["participation"], "tmfg", None, None, "auc", level="network")


def test_small_world_needs_random_networks():
    with pytest.raises(ValueError, match="n_random"):
        check_options(["small_world"], "tmfg", None, None, "auc")
    assert "small_world.wei" not in check_options(None, "tmfg", None, None, "auc")
    assert "small_world.wei" in check_options(None, "tmfg", None, None, "auc", n_random=2)


def test_signed_fallback_is_refused_when_no_negative_weights_remain():
    with pytest.raises(ValueError, match="signed_fallback"):
        check_options(None, "density", {"density": 0.2}, "positive", "auc", signed_fallback="positive")
    with pytest.raises(ValueError, match="signed_fallback"):
        check_options(None, "tmfg", None, None, "auc", signed_fallback="other")


def test_small_world_is_the_ratio_of_normalized_clustering_and_path_length():
    c, nets = _corr()
    labels = [f"r{i}" for i in range(len(c))]
    res = process_subject("s", c, labels, ["small_world.wei", "clust_mean.onnela", "char_path.wei"], "density",
                          {"density": 0.3}, sign="positive", n_random=3, random_seed=0, modules=nets)
    assert res["small_world.wei"] == pytest.approx(res["clust_mean.onnela.norm"] / res["char_path.wei.norm"])
    assert "small_world.wei.norm" not in res


def test_process_subject_returns_one_number_per_global_metric():
    c, nets = _corr()
    labels = [f"r{i}" for i in range(len(c))]
    res = process_subject("s", c, labels, ["eff_global", "participation", "strength"], "density",
                          {"density": [0.2, 0.3]}, return_curves=True, modules=nets)
    assert isinstance(res["eff_global.wei"], float)
    assert set(res["participation.pos.networks"]) == set(labels)
    curves = res["curves"]
    assert set(curves.loc[curves["metric"] == "eff_global.wei", "node"]) == {"global"}


def _toy(n_networks=3, per=6):
    c, nets = _corr(n_networks * per, n_networks)
    labels = [f"roi{i}" for i in range(len(c))]
    atlas = pd.DataFrame({"label": labels, "network_label": [f"net{k}" for k in nets],
                          "hemisphere": ["L" if i % 2 else "R" for i in range(len(c))]})
    mats = {f"s{k}": pd.DataFrame(c * (1 - 0.05 * k) + np.eye(len(c)), index=labels, columns=labels) for k in range(3)}
    return mats, atlas


def test_runner_puts_global_metrics_in_their_own_table(tmp_path):
    mats, atlas = _toy()
    result = compute_graph_metrics(mats, atlas, metrics=["eff_global", "participation", "strength"],
                                   graph_method="density", graph_params={"density": 0.3}, network_graph_method="full",
                                   n_jobs=1, output_dir=str(tmp_path), verbose=False)
    assert list(result.global_df.columns) == [("node", "eff_global.wei"), ("network_hemi", "eff_global.wei")]
    assert list(result.global_df.index) == ["s0", "s1", "s2"]
    assert "eff_global.wei" not in result.node_df.columns.get_level_values(0)
    assert "participation.pos.networks" in result.node_df.columns.get_level_values(0)
    assert "participation.pos.networks" not in result.net_hemi_df.columns.get_level_values(0)
    saved = pd.read_csv(tmp_path / "global" / "node.tsv", sep="\t")
    assert list(saved.columns) == ["ID", "eff_global.wei"]


def test_runner_uses_the_network_column_as_the_given_partition():
    mats, atlas = _toy()
    atlas.loc[:1, "network_label"] = "None"
    result = compute_graph_metrics(mats, atlas, level="node", metrics=["participation"], partitions=("networks",),
                                   graph_method="density", graph_params={"density": 0.3}, n_jobs=1, verbose=False)
    values = result.node_df["participation.pos.networks"]
    assert values[["roi0", "roi1"]].isna().all().all() and values.drop(columns=["roi0", "roi1"]).notna().all().all()


def test_runner_without_networks_uses_louvain_only():
    mats, atlas = _toy()
    result = compute_graph_metrics(mats, atlas.drop(columns="network_label"), level="node", metrics=["participation"],
                                   graph_method="density", graph_params={"density": 0.3}, n_jobs=1, verbose=False)
    assert list(dict.fromkeys(result.node_df.columns.get_level_values(0))) == ["participation.pos.louvain"]
