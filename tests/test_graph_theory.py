import re

import numpy as np
import pandas as pd
import pytest

from brainnet3d.graph_theory import compute_graph_metrics
from brainnet3d.graph_theory.aggregation import build_net2rois, build_net_hemi2rois, compute_net_corr
from brainnet3d.graph_theory.metrics import (
    METRIC_VARIANTS,
    check_options,
    compute_metric,
    parse_metrics,
    process_subject,
)
from brainnet3d.graph_theory.nbs import run_nbs

METRICS = ["clust_coeff", "btwn_cent", "strength", "ge_local"]
DEFAULT_NAMES = ["clust_coeff.costantini", "btwn_cent.inv", "strength.abs", "ge_local.wang"]


def _random_corr(n_rois: int, n_networks: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    net_of_roi = np.arange(n_rois) % n_networks
    factors = rng.standard_normal((200, n_networks))
    ts = factors[:, net_of_roi] + rng.standard_normal((200, n_rois))
    return np.corrcoef(ts.T)


def _toy_inputs(n_networks: int = 4, rois_per_net: int = 3):
    n = n_networks * rois_per_net
    labels = [f"roi{i}" for i in range(n)]
    atlas = pd.DataFrame({
        "label": labels,
        "network_label": [f"net{i % n_networks}" for i in range(n)],
    })
    corr = pd.DataFrame(_random_corr(n, n_networks), index=labels, columns=labels)
    return {"s1": corr}, atlas


def test_process_subject_tmfg_returns_metrics():
    n = 20
    labels = [f"roi{i}" for i in range(n)]
    res = process_subject("s1", _random_corr(n, 4), labels, METRICS, "tmfg")
    assert list(res) == ["subj"] + DEFAULT_NAMES
    for m in DEFAULT_NAMES:
        values = np.array([res[m][lbl] for lbl in labels], dtype=float)
        assert values.shape == (n,)
        assert not np.isnan(values).any()


@pytest.mark.parametrize("apply_fisher_z", [True, False])
def test_network_strength_matches_averaged_matrix(apply_fisher_z):
    matrices, atlas = _toy_inputs()
    result = compute_graph_metrics(
        matrices, atlas, level="network", hemi_split=False,
        metrics=["strength"], apply_fisher_z=apply_fisher_z,
    )

    net_mat = compute_net_corr(matrices, build_net2rois(atlas), apply_fisher_z)["s1"]
    edges = net_mat.values.astype(float)
    if apply_fisher_z:
        edges = np.tanh(edges)
    np.fill_diagonal(edges, 0)
    # a 4-node TMFG keeps every edge, so strength is the plain row sum
    expected = pd.Series(np.abs(edges).sum(axis=1), index=net_mat.index)

    got = result.network_df.loc["s1", "strength.abs"].loc[list(net_mat.index)]
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


def test_single_roi_network_has_no_within_mean_and_no_warning():
    import warnings

    matrices, _ = _toy_inputs()
    net2rois = {"solo": ["roi0"], "pair": ["roi1", "roi2"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        net_mat = compute_net_corr(matrices, net2rois)["s1"]
    assert np.isnan(net_mat.loc["solo", "solo"])
    assert np.isfinite(net_mat.loc["pair", "pair"]) and np.isfinite(net_mat.loc["solo", "pair"])


def test_hemi_split_reads_hemisphere_column():
    atlas = pd.DataFrame({
        "label": ["a", "b", "c", "d"],
        "network_label": ["net0", "net0", "net1", "net1"],
        "hemisphere": ["LH", "right", "L", "midline"],
    })
    assert build_net_hemi2rois(atlas) == {"B_net1": ["d"], "L_net0": ["a"], "L_net1": ["c"], "R_net0": ["b"]}


def test_hemi_split_falls_back_to_label_prefix():
    atlas = pd.DataFrame({"label": ["L_a", "R_b", "x"], "network_label": ["n", "n", "n"]})
    assert build_net_hemi2rois(atlas) == {"B_n": ["x"], "L_n": ["L_a"], "R_n": ["R_b"]}


def test_hemi_split_warns_when_no_hemisphere_found():
    atlas = pd.DataFrame({"label": ["LH_a", "RH_b"], "network_label": ["n", "n"]})
    with pytest.warns(UserWarning, match="hemisphere"):
        build_net_hemi2rois(atlas)


def test_compute_graph_metrics_uses_custom_hemi_col():
    matrices, atlas = _toy_inputs()
    atlas["side"] = ["L", "R"] * (len(atlas) // 2)
    result = compute_graph_metrics(
        matrices, atlas, level="network", metrics=["strength"], hemi_col="side", verbose=False,
    )
    assert {net[0] for _, net in result.net_hemi_df.columns} == {"L", "R"}


def test_metric_tables_have_metric_level_and_float_dtype():
    matrices, atlas = _toy_inputs()
    atlas["hemisphere"] = ["L", "R"] * (len(atlas) // 2)
    result = compute_graph_metrics(
        matrices, atlas, level="both", hemi_split="both", metrics=["strength", "ge_local"], n_jobs=1, verbose=False,
    )
    for df in (result.network_df, result.node_df, result.net_hemi_df):
        assert list(df.columns.get_level_values(0).unique()) == ["strength.abs", "ge_local.wang"]
        assert (df.dtypes == float).all()
    assert list(result.node_df["strength.abs"].columns) == atlas["label"].tolist()


def test_saved_csv_has_one_file_per_level_and_metric(tmp_path):
    matrices, atlas = _toy_inputs()
    compute_graph_metrics(
        matrices, atlas, level="network", hemi_split=False, metrics=["strength", "strength.bin"],
        output_dir=str(tmp_path), verbose=False,
    )
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.csv"))
    assert files == ["correlation/network.csv", "network/strength.abs.csv", "network/strength.bin.csv"]
    header = pd.read_csv(tmp_path / "network" / "strength.abs.csv").columns.tolist()
    assert header == ["ID"] + sorted(atlas["network_label"].unique())


def test_attach_metrics_reads_metric_level():
    import brainnet3d as bnv

    matrices, atlas = _toy_inputs()
    result = compute_graph_metrics(matrices, atlas, level="node", metrics=["strength"], n_jobs=1, verbose=False)
    nodes = atlas.assign(x=0.0, y=0.0, z=0.0)
    plotter = bnv.BrainNetPlotter(bnv.load(matrices["s1"], nodes, subject_id="s1"), subject_id="s1")
    plotter.attach_metrics(result)
    values = result.node_df.loc["s1", "strength.abs"].to_dict()
    assert plotter._extra_cols == {"strength.abs": values, "strength": values}


def _with_values(df: pd.DataFrame, fill) -> pd.DataFrame:
    arr = df.to_numpy(copy=True)
    fill(arr)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def test_node_level_rejects_nan_and_inf_off_diagonal():
    matrices, atlas = _toy_inputs()
    good = matrices["s1"]
    bad = {
        "s_nan": _with_values(good, lambda a: a.__setitem__((0, 1), np.nan)),
        "s_inf": _with_values(good, lambda a: a.__setitem__((2, 3), np.inf)),
        "s_ok": good,
    }
    with pytest.raises(ValueError, match=r"s_nan.*NaN[\s\S]*s_inf.*Inf") as info:
        compute_graph_metrics(bad, atlas, level="node", hemi_split=False, n_jobs=1)
    assert "s_ok" not in str(info.value)


def test_node_level_accepts_inf_diagonal():
    matrices, atlas = _toy_inputs()
    z = {"s1": _with_values(matrices["s1"], lambda a: np.fill_diagonal(a, np.inf))}
    result = compute_graph_metrics(z, atlas, level="node", hemi_split=False, metrics=["strength"], n_jobs=1)
    assert not result.node_df.isna().any().any()


def test_compute_graph_metrics_verbose_false_is_silent(tmp_path, capsys):
    matrices, atlas = _toy_inputs()
    compute_graph_metrics(
        matrices, atlas, level="node", metrics=["strength"], n_jobs=1, output_dir=str(tmp_path), verbose=False,
    )
    assert capsys.readouterr().out == ""
    assert (tmp_path / "node" / "strength.abs.csv").exists()


def _atlas_with_unassigned():
    matrices, atlas = _toy_inputs()
    atlas.loc[[0, 5], "network_label"] = "None"
    atlas["hemisphere"] = ["L"] * 6 + ["R"] * 6
    return matrices, atlas


def test_network_level_leaves_out_unassigned_rois_by_default():
    matrices, atlas = _atlas_with_unassigned()
    result = compute_graph_metrics(
        matrices, atlas, level="both", hemi_split="both", metrics=["strength"], n_jobs=1, verbose=False,
    )
    assert list(result.network_df["strength.abs"].columns) == ["net0", "net1", "net2", "net3"]
    assert not any("None" in net for net in result.net_hemi_df["strength.abs"].columns)
    assert not any("None" in pair for pair in result.net_corr_df.columns)
    assert list(result.node_df["strength.abs"].columns) == atlas["label"].tolist()


@pytest.mark.parametrize("exclude, networks", [
    ((), {"net0", "net1", "net2", "net3", "None"}),
    (None, {"net0", "net1", "net2", "net3", "None"}),
    ("net3", {"net0", "net1", "net2", "None"}),
    (["None", "net3"], {"net0", "net1", "net2"}),
])
def test_exclude_networks_chooses_the_labels_left_out(exclude, networks):
    matrices, atlas = _atlas_with_unassigned()
    result = compute_graph_metrics(
        matrices, atlas, level="network", hemi_split=False, metrics=["strength"], graph_method="full",
        exclude_networks=exclude, verbose=False,
    )
    assert set(result.network_df["strength.abs"].columns) == networks
    assert not result.failed


def test_verbose_reports_graph_method_sign_and_excluded_rois(capsys):
    matrices, atlas = _atlas_with_unassigned()
    compute_graph_metrics(
        matrices, atlas, level="both", hemi_split=True, metrics=["strength"], graph_method="density",
        graph_params={"density": 0.5}, network_graph_method="tmfg", n_jobs=1,
    )
    out = capsys.readouterr().out
    assert re.search(r"Node-level.*graph_method=density, sign=positive", out)
    assert re.search(r"hemisphere.*graph_method=tmfg, sign=abs", out)
    assert re.search(r"leaving out 2 ROIs labelled 'None'", out)


def test_params_record_the_options_each_level_used():
    matrices, atlas = _atlas_with_unassigned()
    result = compute_graph_metrics(
        matrices, atlas, level="both", hemi_split=True, metrics=["strength"], graph_method="density",
        graph_params={"density": np.array([0.4, 0.5])}, network_graph_method="tmfg", n_random=1, n_jobs=1,
        verbose=False,
    )
    params = result.params
    assert params["levels"]["node"] == {
        "graph_method": "density", "graph_params": {"density": [0.4, 0.5]}, "sign": "positive",
        "metrics": ["strength.abs", "strength.abs.norm"], "n_nodes": 12,
    }
    assert params["levels"]["network_hemi"]["graph_method"] == "tmfg"
    assert params["levels"]["network_hemi"]["sign"] == "abs"
    assert params["levels"]["network_hemi"]["n_nodes"] == 8
    assert "network" not in params["levels"]
    assert params["excluded_rois"] == {"None": 2}
    assert params["subjects"] == ["s1"]
    assert params["options"]["summary"] == "auc"
    assert params["options"]["exclude_networks"] == ["None"]
    assert "brainnet3d" in params["packages"] and "bctpy" in params["packages"]


def test_params_seed_reproduces_an_unseeded_run():
    matrices, atlas = _toy_inputs()
    kwargs = dict(level="node", metrics=["clust_coeff.onnela"], graph_method="density",
                  graph_params={"density": 0.5}, n_random=2, n_jobs=1, verbose=False)
    first = compute_graph_metrics(matrices, atlas, **kwargs)
    seed = first.params["options"]["random_seed"]
    assert isinstance(seed, int)
    again = compute_graph_metrics(matrices, atlas, random_seed=seed, **kwargs)
    pd.testing.assert_frame_equal(first.node_df, again.node_df)
    assert again.params["options"]["random_seed"] == seed


def test_params_are_saved_as_json_and_name_a_callable_method(tmp_path):
    import json

    matrices, atlas = _toy_inputs()
    result = compute_graph_metrics(
        matrices, atlas, level="node", metrics=["strength"], graph_method=_keep_k_strongest,
        graph_params={"k": 3}, n_jobs=1, output_dir=str(tmp_path), verbose=False,
    )
    saved = json.loads((tmp_path / "parameters.json").read_text(encoding="utf-8"))
    assert saved == result.params
    assert saved["levels"]["node"]["graph_method"] == "_keep_k_strongest"
    assert saved["options"]["random_seed"] is None


def _failing_method(corrmat):
    raise RuntimeError("boom")


def test_process_subject_raises_on_failure():
    with pytest.raises(RuntimeError, match="boom"):
        process_subject("s1", _random_corr(6, 2), [f"roi{i}" for i in range(6)], METRICS, _failing_method)


def test_failed_subject_is_reported_and_left_nan():
    matrices, atlas = _toy_inputs()
    with pytest.warns(UserWarning, match=r"network / s1: RuntimeError: boom"):
        result = compute_graph_metrics(
            matrices, atlas, level="network", hemi_split=False, graph_method=_failing_method,
        )
    assert result.failed == {"network": {"s1": "RuntimeError: boom"}}
    assert result.network_df.loc["s1"].isna().all()


def test_unknown_graph_method_raises_before_computing():
    matrices, atlas = _toy_inputs()
    with pytest.raises(ValueError, match="graph_method"):
        compute_graph_metrics(matrices, atlas, level="network", graph_method="nope")


def test_parse_metrics_expands_defaults_and_all():
    assert parse_metrics(None) == DEFAULT_NAMES
    assert parse_metrics(["strength", "strength.abs", "strength.neg"]) == ["strength.abs", "strength.neg"]
    assert len(parse_metrics("all")) == sum(len(v) for v in METRIC_VARIANTS.values())
    with pytest.raises(ValueError, match="strength.nope"):
        parse_metrics(["strength.nope"])


def test_metric_variants_on_small_signed_graph():
    pytest.importorskip("bct")
    A = np.array([
        [0.0, 0.5, -0.4, 0.0],
        [0.5, 0.0, 0.3, 0.2],
        [-0.4, 0.3, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0],
    ])
    np.testing.assert_allclose(compute_metric("strength.abs", A), [0.9, 1.0, 0.7, 0.2])
    np.testing.assert_allclose(compute_metric("strength.pos", A), [0.5, 1.0, 0.3, 0.2])
    np.testing.assert_allclose(compute_metric("strength.neg", A), [0.4, 0.0, 0.4, 0.0])
    np.testing.assert_allclose(compute_metric("strength.bin", A), [2, 3, 2, 1])
    np.testing.assert_allclose(compute_metric("clust_coeff.bin", A), [1, 1 / 3, 1, 0])
    np.testing.assert_allclose(
        compute_metric("btwn_cent.inv_norm", A), compute_metric("btwn_cent.inv", A) / 6,
    )


def test_costantini_matches_bct_with_negative_weights_and_low_degree_nodes():
    bct = pytest.importorskip("bct")
    A = np.triu(_random_corr(15, 3, seed=1), 1)
    A[np.abs(A) < 0.1] = 0
    A[:3] = 0
    A[:, :3] = 0
    # node 0 isolated, node 1 pendant, node 2 with two neighbours of very different weight
    for (i, j), w in {(1, 3): 0.6, (2, 3): 0.9, (2, 4): 1e-8, (3, 4): -0.5}.items():
        A[i, j] = w
    A = A + A.T
    assert (A < 0).any()

    got = compute_metric("clust_coeff.costantini", A)
    np.testing.assert_allclose(got, bct.clustering_coef_wu_sign(A.copy(), coef_type="costantini"), rtol=0, atol=1e-12)
    assert got[0] == got[1] == 0
    assert got[2] == pytest.approx(-0.5, abs=1e-12)


def test_rubinov_local_efficiency_by_hand():
    a, b, c, d, e = 0.8, 0.6, 0.5, 0.4, 0.7
    W = np.zeros((4, 4))
    # node 0 links to 1, 2, 3; among its neighbours only 1-2 and 2-3 are linked
    for (i, j), w in {(0, 1): a, (0, 2): b, (0, 3): e, (1, 2): c, (2, 3): d}.items():
        W[i, j] = W[j, i] = w
    to_0 = {1: a, 2: b, 3: e}
    inv_dist = {(1, 2): c, (2, 3): d, (1, 3): 1 / (1 / c + 1 / d)}
    expected = 2 * sum(np.cbrt(to_0[j] * to_0[h] * v) for (j, h), v in inv_dist.items()) / (3 * 2)
    assert compute_metric("ge_local.rubinov", W)[0] == pytest.approx(expected)


def test_density_range_matches_proportional_threshold_loop():
    bct = pytest.importorskip("bct")
    n = 30
    corr = _random_corr(n, 3)
    corr[corr < 0] = 0
    np.fill_diagonal(corr, 0)
    labels = [f"roi{i}" for i in range(n)]
    densities = [0.2, 0.25, 0.3]
    curve = [bct.clustering_coef_wu(bct.threshold_proportional(corr, d)) for d in densities]
    expected = sum((curve[i] + curve[i + 1]) / 2 * (densities[i + 1] - densities[i]) for i in range(2))

    for summary, scale in (("auc", 1.0), ("mean", 1 / 0.1)):
        res = process_subject(
            "s1", corr, labels, ["clust_coeff.onnela"], "density", {"density": densities[::-1]}, summary=summary,
        )
        got = [res["clust_coeff.onnela"][lbl] for lbl in labels]
        np.testing.assert_allclose(got, expected * scale)


def test_disparity_alpha_range_integrates_single_runs():
    n = 30
    corr = _random_corr(n, 3)
    labels = [f"roi{i}" for i in range(n)]
    alphas = [0.05, 0.1, 0.2]
    curve = [
        np.array(list(process_subject("s1", corr, labels, ["strength"], "disparity", {"alpha": a})["strength.abs"].values()))
        for a in alphas
    ]
    assert not np.allclose(curve[0], curve[2])
    expected = sum((curve[i] + curve[i + 1]) / 2 * (alphas[i + 1] - alphas[i]) for i in range(2))
    res = process_subject("s1", corr, labels, ["strength"], "disparity", {"alpha": alphas})
    np.testing.assert_allclose([res["strength.abs"][lbl] for lbl in labels], expected)


def test_return_curves_keeps_values_at_each_density(tmp_path):
    matrices, atlas = _toy_inputs()
    result = compute_graph_metrics(
        matrices, atlas, level="node", metrics=["strength"], graph_method="density",
        graph_params={"density": [0.3, 0.5]}, return_curves=True, n_jobs=1,
        output_dir=str(tmp_path), verbose=False,
    )
    curves = result.curves
    assert list(curves.columns) == ["level", "ID", "metric", "node", "threshold", "value"]
    assert len(curves) == 2 * len(atlas)
    at = {t: df.set_index("node")["value"] for t, df in curves.groupby("threshold")}
    auc = (at[0.3] + at[0.5]) / 2 * 0.2
    np.testing.assert_allclose(result.node_df.loc["s1", "strength.abs"][auc.index].values, auc.values)
    assert (tmp_path / "node" / "curves.csv").exists()


def test_random_normalization_divides_by_the_mean_over_random_networks():
    from brainnet3d.graph_theory.randomize import randomize_signed
    from brainnet3d.graph_theory.sparsify import build_adjacency

    n = 20
    corr = _random_corr(n, 4)
    np.fill_diagonal(corr, 0)
    labels = [f"roi{i}" for i in range(n)]
    names = ["clust_coeff.costantini", "strength.abs", "strength.bin"]
    res = process_subject(
        "s1", corr, labels, names, "density", {"density": 0.3}, sign="abs", n_random=3, random_seed=5,
    )
    assert "strength.bin.norm" not in res

    A = build_adjacency(corr, "density", {"density": 0.3})
    rng = np.random.default_rng(5)
    nulls = [randomize_signed(A, 10, rng) for _ in range(3)]
    for m in names[:2]:
        expected = compute_metric(m, A) / np.mean([compute_metric(m, R) for R in nulls], axis=0)
        np.testing.assert_allclose([res[f"{m}.norm"][lbl] for lbl in labels], expected)


def test_random_normalization_is_integrated_like_the_raw_metric():
    n = 20
    labels = [f"roi{i}" for i in range(n)]
    res = process_subject(
        "s1", _random_corr(n, 4), labels, ["strength.abs"], "density", {"density": [0.3, 0.5]},
        sign="abs", return_curves=True, n_random=2, random_seed=1,
    )
    norm = res["curves"][res["curves"]["metric"] == "strength.abs.norm"]
    at = {t: df.set_index("node")["value"] for t, df in norm.groupby("threshold")}
    expected = (at[0.3] + at[0.5]) / 2 * 0.2
    np.testing.assert_allclose([res["strength.abs.norm"][lbl] for lbl in expected.index], expected.values)


def test_random_normalization_columns_files_and_seeds(tmp_path):
    matrices, atlas = _toy_inputs()
    matrices["s2"] = matrices["s1"]
    kwargs = dict(
        level="node", metrics=["strength.abs", "strength.bin"], graph_method="density",
        graph_params={"density": 0.4}, sign="abs", n_random=2, random_seed=0, n_jobs=1, verbose=False,
    )
    first = compute_graph_metrics(matrices, atlas, output_dir=str(tmp_path), **kwargs)
    again = compute_graph_metrics(matrices, atlas, **kwargs)

    df = first.node_df
    assert list(df.columns.get_level_values(0).unique()) == ["strength.abs", "strength.bin", "strength.abs.norm"]
    pd.testing.assert_frame_equal(df, again.node_df)
    assert not np.allclose(df.loc["s1", "strength.abs.norm"], df.loc["s2", "strength.abs.norm"])
    assert (tmp_path / "node" / "strength.abs.norm.csv").exists()


@pytest.mark.parametrize("hemi_split, present", [
    (True, {"net_hemi_df"}),
    (False, {"network_df"}),
    ("both", {"network_df", "net_hemi_df"}),
])
def test_hemi_split_chooses_network_graphs(hemi_split, present):
    matrices, atlas = _toy_inputs()
    atlas["hemisphere"] = ["L", "R"] * (len(atlas) // 2)
    result = compute_graph_metrics(
        matrices, atlas, level="network", hemi_split=hemi_split, metrics=["strength"], verbose=False,
    )
    assert {name for name in ("network_df", "net_hemi_df") if getattr(result, name) is not None} == present


def test_network_graph_method_overrides_node_method():
    matrices, atlas = _toy_inputs(n_networks=6, rois_per_net=2)
    result = compute_graph_metrics(
        matrices, atlas, level="both", hemi_split=False, metrics=["strength.bin"],
        graph_method="density", graph_params={"density": 0.2}, network_graph_method="full",
        n_jobs=1, verbose=False,
    )
    assert result.node_df.loc["s1"].sum() == 2 * round(0.2 * 66)

    net_mat = np.tanh(compute_net_corr(matrices, build_net2rois(atlas))["s1"].values.astype(float))
    np.fill_diagonal(net_mat, 0)
    assert result.network_df.loc["s1"].sum() == (net_mat > 0).sum()


@pytest.mark.parametrize("kwargs, match", [
    (dict(graph_method="mst", metrics=["clust_coeff"]), "spanning tree"),
    (dict(graph_method="density", graph_params={"density": 0.2}, metrics=["strength.neg"]), "sign"),
    (dict(graph_method="density"), "graph_params"),
    (dict(graph_method="knn", graph_params={"k": [2, 3]}), "single value"),
    (dict(graph_method="density", graph_params={"density": [0.2]}), "two distinct"),
    (dict(graph_method="density", graph_params={"density": 1.5}), "density"),
    (dict(summary="median"), "summary"),
    (dict(n_random=-1), "n_random"),
    (dict(n_random=2, random_swaps=0), "random_swaps"),
    (dict(hemi_split="left"), "hemi_split"),
])
def test_invalid_options_raise_before_computing(kwargs, match):
    matrices, atlas = _toy_inputs()
    with pytest.raises(ValueError, match=match):
        compute_graph_metrics(matrices, atlas, level="both", n_jobs=1, verbose=False, **kwargs)


@pytest.mark.parametrize("level, hemi_split, density, match", [
    ("node", False, 0.1, r"node \(12 nodes\).*; got 0\.1$"),
    ("network", False, [0.2, 0.6], r"network \(4 nodes\).*; got 0\.2\n.*network_graph_method"),
    ("network", True, 0.2, r"network_hemi \(8 nodes\).*; got 0\.2\n.*network_graph_method"),
])
def test_mst_density_below_spanning_tree_raises_before_computing(level, hemi_split, density, match):
    matrices, atlas = _toy_inputs()
    atlas["hemisphere"] = np.repeat(["L", "R"], len(atlas) // 2)  # every network in both hemispheres
    with pytest.raises(ValueError, match=match):
        compute_graph_metrics(
            matrices, atlas, level=level, hemi_split=hemi_split, graph_method="mst_density",
            graph_params={"density": density}, n_jobs=1, verbose=False,
        )


@pytest.mark.parametrize("level, k, match", [
    ("node", 12, r"node \(12 nodes\): k=12 .* 11 possible neighbours$"),
    ("both", 5, r"network \(4 nodes\): k=5 .* 3 possible neighbours\n.*network_graph_method"),
])
def test_knn_beyond_the_graph_size_raises_before_computing(level, k, match):
    matrices, atlas = _toy_inputs()
    with pytest.raises(ValueError, match=match):
        compute_graph_metrics(
            matrices, atlas, level=level, hemi_split=False, graph_method="knn", graph_params={"k": k},
            n_jobs=1, verbose=False,
        )


def test_network_graph_params_lift_mst_density_for_small_graphs():
    matrices, atlas = _toy_inputs()
    result = compute_graph_metrics(
        matrices, atlas, level="both", hemi_split=False, metrics=["strength.bin"], graph_method="mst_density",
        graph_params={"density": 0.2}, network_graph_params={"density": 0.6}, sign="abs", n_jobs=1, verbose=False,
    )
    assert result.failed == {}


def test_all_metrics_skip_those_constant_on_a_spanning_tree():
    names = check_options("all", "mst", None, None, "auc")
    assert not [m for m in names if m.startswith(("clust_coeff.", "ge_local.", "strength.neg"))]
    assert "btwn_cent.log" in names


def _keep_k_strongest(corrmat, k):
    upper = np.abs(np.triu(corrmat, 1))
    cutoff = np.sort(upper, axis=None)[-k]
    kept = np.where(upper >= cutoff, corrmat, 0.0)
    return kept + kept.T


def test_callable_graph_method_may_return_array_and_take_params():
    n = 8
    labels = [f"roi{i}" for i in range(n)]
    res = process_subject("s1", _random_corr(n, 2), labels, ["strength.bin"], _keep_k_strongest, {"k": 3})
    assert sum(res["strength.bin"].values()) == 6


def test_sign_positive_removes_negative_weights():
    n = 10
    corr = _random_corr(n, 2)
    labels = [f"roi{i}" for i in range(n)]
    res = process_subject("s1", corr, labels, ["strength.abs"], "full", sign="positive")
    np.fill_diagonal(corr, 0)
    np.testing.assert_allclose([res["strength.abs"][lbl] for lbl in labels], np.clip(corr, 0, None).sum(axis=1))


def test_sign_negative_keeps_negative_magnitudes():
    n = 10
    corr = _random_corr(n, 2)
    labels = [f"roi{i}" for i in range(n)]
    res = process_subject("s1", corr, labels, ["strength.abs", "strength.pos"], "full", sign="negative")
    np.fill_diagonal(corr, 0)
    expected = -np.clip(corr, None, 0).sum(axis=1)
    np.testing.assert_allclose([res["strength.abs"][lbl] for lbl in labels], expected)
    np.testing.assert_allclose([res["strength.pos"][lbl] for lbl in labels], expected)
    with pytest.raises(ValueError, match="strength.neg"):
        process_subject("s1", corr, labels, ["strength.neg"], "full", sign="negative")


def test_normalize_weights_divides_by_the_largest_weight():
    n = 10
    corr = _random_corr(n, 2)
    labels = [f"roi{i}" for i in range(n)]
    raw = process_subject("s1", corr, labels, ["strength.abs"], "full", sign="abs")
    scaled = process_subject("s1", corr, labels, ["strength.abs"], "full", sign="abs", normalize_weights=True)
    off_diag = corr[~np.eye(n, dtype=bool)]
    np.testing.assert_allclose(
        [scaled["strength.abs"][lbl] for lbl in labels],
        [raw["strength.abs"][lbl] / np.abs(off_diag).max() for lbl in labels],
    )


def test_too_few_positive_edges_warns_once_with_subjects():
    matrices, atlas = _toy_inputs()
    corr = matrices["s1"].to_numpy(copy=True)
    corr -= corr[np.triu_indices(len(corr), 1)].mean()  # many negative weights
    matrices = {sid: pd.DataFrame(corr, index=atlas["label"], columns=atlas["label"]) for sid in ("s1", "s2")}
    n_pos = int((np.triu(corr, 1) > 0).sum())
    with pytest.warns(UserWarning, match=rf"node / s1: {n_pos} edges available, asked for 59") as record:
        compute_graph_metrics(
            matrices, atlas, level="node", metrics=["strength.bin"], graph_method="density",
            graph_params={"density": [0.2, 0.9]}, n_jobs=1, verbose=False,
        )
    assert len([w for w in record if "Fewer edges" in str(w.message)]) == 1
    assert "s2" in str(record[0].message)


def _null_groups(n_nodes: int = 12, n_subjects: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = [f"roi{i}" for i in range(n_nodes)]

    def group(start):
        mats = {}
        for s in range(start, start + n_subjects):
            a = rng.standard_normal((n_nodes, n_nodes))
            mats[f"sub-{s:02d}"] = pd.DataFrame((a + a.T) / 2, index=labels, columns=labels)
        return mats

    return group(0), group(n_subjects)


def test_run_nbs_null_data_gives_valid_pvalues():
    pytest.importorskip("bct")
    g1, g2 = _null_groups()
    res = run_nbs(g1, g2, thresh=1.5, k=100, seed=0, verbose=False)
    assert res.pval.size > 0
    assert ((res.pval > 0) & (res.pval <= 1)).all()
    assert np.unique(res.null).size > 1


def test_run_nbs_is_reproducible_with_seed():
    pytest.importorskip("bct")
    g1, g2 = _null_groups()
    a = run_nbs(g1, g2, thresh=1.5, k=50, seed=1, verbose=False)
    b = run_nbs(g1, g2, thresh=1.5, k=50, seed=1, verbose=False)
    np.testing.assert_array_equal(a.pval, b.pval)
    np.testing.assert_array_equal(a.null, b.null)


def test_run_nbs_params_record_options_and_groups():
    pytest.importorskip("bct")
    g1, g2 = _null_groups(n_subjects=4)
    res = run_nbs(g1, g2, thresh=1.5, k=20, tail="right", seed=1, verbose=False)
    assert res.params["options"] == {"thresh": 1.5, "k": 20, "tail": "right", "paired": False, "seed": 1}
    assert res.params["groups"] == {"g1": list(g1), "g2": list(g2)}
    assert res.params["n_nodes"] == 12
    assert "bctpy" in res.params["packages"]


def test_run_nbs_params_seed_reproduces_an_unseeded_run():
    pytest.importorskip("bct")
    g1, g2 = _null_groups()
    first = run_nbs(g1, g2, thresh=1.5, k=30, verbose=False)
    seed = first.params["options"]["seed"]
    assert isinstance(seed, int)
    again = run_nbs(g1, g2, thresh=1.5, k=30, seed=seed, verbose=False)
    np.testing.assert_array_equal(first.null, again.null)
    np.testing.assert_array_equal(first.pval, again.pval)


def _effect_groups(n_subjects: int = 10, seed: int = 0):
    g1, g2 = _null_groups(n_nodes=16, n_subjects=n_subjects, seed=seed)
    for m in g2.values():
        m.iloc[:6, :6] += 1.2
    return g1, g2


@pytest.mark.parametrize("paired, tail", [(False, "both"), (False, "left"), (False, "right"), (True, "both")])
def test_run_nbs_matches_bct_serial(paired, tail):
    bct = pytest.importorskip("bct")
    import contextlib
    import io

    g1, g2 = _effect_groups()
    x = np.stack([m.values for m in g1.values()], axis=2)
    y = np.stack([m.values for m in g2.values()], axis=2)
    with contextlib.redirect_stdout(io.StringIO()):
        pval, adj, null = bct.nbs_bct(x, y, 1.0, k=60, tail=tail, paired=paired, seed=3)
    res = run_nbs(g1, g2, thresh=1.0, k=60, tail=tail, paired=paired, seed=3, verbose=False)
    np.testing.assert_array_equal(res.pval, pval)
    np.testing.assert_array_equal(res.adj, adj)
    np.testing.assert_array_equal(res.null, null)


def test_run_nbs_is_silent_without_verbose(capsys):
    pytest.importorskip("bct")
    g1, g2 = _effect_groups()
    run_nbs(g1, g2, thresh=1.0, k=20, seed=0, verbose=False)
    assert capsys.readouterr().out == ""


def test_run_nbs_does_not_use_edgewise_bct_loop(monkeypatch):
    bct = pytest.importorskip("bct")

    def _fail(*args, **kwargs):
        raise AssertionError("bct.nbs_bct was called")

    monkeypatch.setattr(bct, "nbs_bct", _fail)
    monkeypatch.setattr(bct.nbs, "nbs_bct", _fail)
    g1, g2 = _effect_groups()
    res = run_nbs(g1, g2, thresh=1.0, k=20, seed=0, verbose=False)
    assert res.null.shape == (20,)


def test_run_nbs_rejects_threshold_with_no_edges():
    bct = pytest.importorskip("bct")
    g1, g2 = _effect_groups()
    with pytest.raises(bct.BCTParamError, match="Unsuitable threshold"):
        run_nbs(g1, g2, thresh=1e6, k=5, seed=0, verbose=False)
