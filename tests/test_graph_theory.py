import numpy as np
import pandas as pd
import pytest

from brainnet3d.graph_theory import compute_graph_metrics
from brainnet3d.graph_theory.aggregation import build_net2rois, compute_net_corr
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
