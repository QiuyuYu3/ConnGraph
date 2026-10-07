import numpy as np
import pandas as pd
import pytest

from brainnet3d.graph_theory import compute_graph_metrics
from brainnet3d.graph_theory.aggregation import build_net2rois, build_net_hemi2rois, compute_net_corr
from brainnet3d.graph_theory.metrics import process_subject
from brainnet3d.graph_theory.nbs import run_nbs

METRICS = ["clust_coeff", "btwn_cent", "strength", "ge_local"]


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
    assert res is not None
    for m in METRICS:
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

    got = result.network_df.loc["s1", [f"strength_{n}" for n in net_mat.index]].astype(float)
    np.testing.assert_allclose(got.values, expected.values, rtol=1e-10)


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
    assert {c.split("_", 1)[1][0] for c in result.net_hemi_df.columns} == {"L", "R"}


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
    assert (tmp_path / "node_graph_theory.csv").exists()


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
