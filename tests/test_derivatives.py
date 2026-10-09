import numpy as np
import pandas as pd
import pytest

import brainnet3d as bnv

CHANNELS = [f"S{i}_D{i}" for i in range(1, 7)]


def _fc(seed, bad=()):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (len(CHANNELS), len(CHANNELS)))
    m = (a + a.T) / 2
    np.fill_diagonal(m, 1.0)
    m[[CHANNELS.index(b) for b in bad], :] = np.nan
    m[:, [CHANNELS.index(b) for b in bad]] = np.nan
    return m


def _fnirs_tree(root, session=None, bad_in_02=()):
    """Two participants as fnirs-pipe writes them, with an ROI-level file that must be ignored."""
    for k, sub in enumerate(("01", "02")):
        folder = root / f"sub-{sub}" / (session or "") / "nirs"
        folder.mkdir(parents=True)
        stem = f"sub-{sub}" + (f"_{session}" if session else "") + "_task-rest"
        for j, chromo in enumerate(("hbo", "hbr")):
            names = [f"{c} {chromo}" for c in CHANNELS]
            m = _fc(10 * k + j, bad_in_02 if sub == "02" else ())
            pd.DataFrame(m, index=names, columns=names).to_csv(
                folder / f"{stem}_chromo-{chromo}_stat-pearson_relmat.tsv", sep="\t", index_label="channel", na_rep="n/a")
            pd.DataFrame([[1.0]], index=["roi"], columns=["roi"]).to_csv(
                folder / f"{stem}_chromo-{chromo}_seg-map_agg-roi_stat-pearson_relmat.tsv", sep="\t")
    return root


def test_load_fnirs_pipe_reads_one_chromophore_with_bare_channel_names(tmp_path):
    root = _fnirs_tree(tmp_path, bad_in_02=["S3_D3"])
    with pytest.warns(UserWarning, match="S3_D3"):
        matrices, nodes = bnv.load_fnirs_pipe(str(root), "hbr", verbose=False)
    assert sorted(matrices) == ["01", "02"]
    keep = [c for c in CHANNELS if c != "S3_D3"]
    assert matrices["01"].columns.tolist() == keep and nodes["label"].tolist() == keep
    expected = pd.DataFrame(_fc(1), index=CHANNELS, columns=CHANNELS).loc[keep, keep]
    np.testing.assert_allclose(matrices["01"].to_numpy(), expected.to_numpy())
    record = nodes.attrs["brainnet3d_input"]
    assert (record["source"], record["chromophore"], record["task"], record["dropped"]) == (
        "fnirs-pipe", "hbr", "rest", ["S3_D3"])


def test_load_fnirs_pipe_finds_files_in_a_session_folder(tmp_path):
    root = _fnirs_tree(tmp_path, session="ses-02")
    matrices, _ = bnv.load_fnirs_pipe(str(root), "hbo", session="ses-02", verbose=False)
    np.testing.assert_allclose(matrices["02"].to_numpy(), _fc(10))


def test_load_fnirs_pipe_without_a_session_finds_any_session(tmp_path):
    root = _fnirs_tree(tmp_path, session="ses-02")
    matrices, _ = bnv.load_fnirs_pipe(str(root), "hbo", verbose=False)
    assert sorted(matrices) == ["01", "02"]


XCPD_TAIL = "task-rest_space-fsLR_seg-Gordon_stat-pearsoncorrelation_relmat.tsv"


@pytest.mark.parametrize("session", ["ses-01", None])
def test_load_xcpd_without_a_session_finds_files_with_or_without_one(tmp_path, session):
    for k, sub in enumerate(("01", "02")):
        folder = tmp_path / f"sub-{sub}" / (session or "") / "func"
        folder.mkdir(parents=True)
        name = f"sub-{sub}_" + (f"{session}_" if session else "") + XCPD_TAIL
        pd.DataFrame(_fc(k), index=CHANNELS, columns=CHANNELS).to_csv(folder / name, sep="\t")
    atlas = tmp_path / "atlases" / "atlas-Gordon"
    atlas.mkdir(parents=True)
    pd.DataFrame({"label": CHANNELS, "network_label": ["a", "b"] * 3}).to_csv(
        atlas / "atlas-Gordon_dseg.tsv", sep="\t", index=False)
    matrices, _ = bnv.load_xcpd(str(tmp_path), "Gordon", session=None, verbose=False)
    assert sorted(matrices) == ["01", "02"]
    np.testing.assert_allclose(matrices["02"].to_numpy(), _fc(1))


def test_load_fnirs_pipe_rejects_an_unknown_chromophore(tmp_path):
    with pytest.raises(ValueError, match="hbo"):
        bnv.load_fnirs_pipe(str(_fnirs_tree(tmp_path)), "hbt")


def test_xcpd_loaders_keep_their_import_paths():
    from brainnet3d.graph_theory import load_xcpd as a
    from brainnet3d.graph_theory.xcpd import load_xcpd as b

    assert a is b is bnv.load_xcpd


def test_graph_methods_describe_fnirs_input(tmp_path):
    from brainnet3d.graph_theory import compute_graph_metrics
    from brainnet3d.report.methods import graph_methods

    matrices, nodes = bnv.load_fnirs_pipe(str(_fnirs_tree(tmp_path)), "hbo", verbose=False)
    result = compute_graph_metrics(matrices, nodes, level="node", metrics=["strength"], graph_method="full",
                                   n_jobs=1, verbose=False)
    plain = graph_methods(result.params)["plain"]
    assert "Functional connectivity matrices derived from fnirs-pipe (HbO; Pearson's r) were used" in plain
    assert "each of the 6 channels." in plain and "parcels" not in plain
    result.save_report(tmp_path / "graph.html", static_brain=False)
    assert "fnirs-pipe (chromophore hbo, task rest)" in (tmp_path / "graph.html").read_text(encoding="utf-8")
