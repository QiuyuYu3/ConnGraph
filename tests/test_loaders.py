import numpy as np
import pandas as pd
import pytest

import brainnet3d as bnv
from brainnet3d.graph_theory import load_xcpd_flat

LABELS = [f"r{i}" for i in range(6)]
XCPD_STEM = "_ses-01_task-rest_space-fsLR_seg-Gordon_stat-pearsoncorrelation_relmat.tsv"


def _matrix(nan_labels=(), seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (len(LABELS), len(LABELS)))
    df = pd.DataFrame((a + a.T) / 2, index=LABELS, columns=LABELS)
    for lbl in nan_labels:
        df.loc[lbl, :] = np.nan
        df.loc[:, lbl] = np.nan
    return df


def _nodes():
    return pd.DataFrame({"label": LABELS, "x": 0.0, "y": 0.0, "z": 0.0})


def test_load_drops_bad_nodes_with_warning():
    with pytest.warns(UserWarning, match=r"\['r2'\]"):
        ds = bnv.load(_matrix(["r2"]), _nodes(), bad_node_threshold=0.5)
    assert "r2" not in ds.matrices["single"].columns
    assert "r2" not in ds.nodes_df["label"].tolist()


@pytest.mark.parametrize("mode, dropped", [("union", {"r1", "r4"}), ("intersection", {"r1"})])
def test_load_group_drop_modes(mode, dropped):
    mats = {"sub-a": _matrix(["r1", "r4"]), "sub-b": _matrix(["r1"], seed=1)}
    with pytest.warns(UserWarning, match="Dropping"):
        ds = bnv.load_group(mats, _nodes(), bad_node_threshold=0.5, drop_mode=mode)
    for mat in ds.matrices.values():
        assert set(LABELS) - set(mat.columns) == dropped


def test_load_xcpd_flat_drops_bad_nodes_with_warning(tmp_path):
    _matrix(["r3"]).to_csv(tmp_path / f"sub-01{XCPD_STEM}", sep="\t")
    _matrix(seed=1).to_csv(tmp_path / f"sub-02{XCPD_STEM}", sep="\t")
    atlas_path = tmp_path / "atlas.tsv"
    pd.DataFrame({"label": LABELS, "network_label": ["a", "b"] * 3}).to_csv(atlas_path, sep="\t", index=False)

    with pytest.warns(UserWarning, match=r"\['r3'\]"):
        matrices, _ = load_xcpd_flat(str(tmp_path), str(atlas_path), atlas="Gordon", bad_node_threshold=0.5)
    assert sorted(matrices) == ["01", "02"]
    assert all("r3" not in m.columns for m in matrices.values())
