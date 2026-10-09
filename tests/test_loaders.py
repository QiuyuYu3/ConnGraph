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


def _xcpd_flat_dir(tmp_path, bad_sub01=()):
    _matrix(bad_sub01).to_csv(tmp_path / f"sub-01{XCPD_STEM}", sep="\t")
    _matrix(seed=1).to_csv(tmp_path / f"sub-02{XCPD_STEM}", sep="\t")
    atlas_path = tmp_path / "atlas.tsv"
    pd.DataFrame({"label": LABELS, "network_label": ["a", "b"] * 3}).to_csv(atlas_path, sep="\t", index=False)
    return str(tmp_path), str(atlas_path)


def test_load_xcpd_flat_drops_bad_nodes_with_warning(tmp_path):
    flat_dir, atlas_path = _xcpd_flat_dir(tmp_path, bad_sub01=["r3"])
    with pytest.warns(UserWarning, match=r"\['r3'\]"):
        matrices, _ = load_xcpd_flat(flat_dir, atlas_path, atlas="Gordon", bad_node_threshold=0.5)
    assert sorted(matrices) == ["01", "02"]
    assert all("r3" not in m.columns for m in matrices.values())


def test_load_xcpd_flat_warns_once_for_skipped_subjects(tmp_path):
    flat_dir, atlas_path = _xcpd_flat_dir(tmp_path)
    with pytest.warns(UserWarning, match=r"Skipped 2 subject\(s\)[\s\S]*sub-03[\s\S]*sub-04") as record:
        matrices, _ = load_xcpd_flat(flat_dir, atlas_path, atlas="Gordon", subject_ids=["01", "03", "04"])
    assert sorted(matrices) == ["01"]
    assert len(record) == 1


def test_load_xcpd_flat_verbose_false_is_silent(tmp_path, capsys):
    flat_dir, atlas_path = _xcpd_flat_dir(tmp_path)
    load_xcpd_flat(flat_dir, atlas_path, atlas="Gordon", verbose=False)
    assert capsys.readouterr().out == ""


def test_missing_nodes_warns():
    with pytest.warns(UserWarning, match="not found in nodes file"):
        ds = bnv.load(_matrix(), _nodes().iloc[1:])
    assert ds.nodes_df["label"].tolist() == LABELS[1:]


def test_no_edges_above_threshold_warns():
    ds = bnv.load(_matrix(), _nodes())
    with pytest.warns(UserWarning, match="No edges"):
        img = bnv.BrainNetPlotter(ds).plot(edge_threshold=2.0, node_color="steelblue")
    assert img is not None


def test_node_table_keeps_none_as_text(tmp_path):
    labels = ["a", "b", "c"]
    pd.DataFrame(np.eye(3), index=labels, columns=labels).to_csv(tmp_path / "matrix.csv")
    (tmp_path / "nodes.csv").write_text(
        "label,x,y,z,network,score\na,0,0,0,None,1.5\nb,1,0,0,Default,NA\nc,2,0,0,,2.5\n", encoding="utf-8"
    )
    nodes = bnv.load(str(tmp_path / "matrix.csv"), str(tmp_path / "nodes.csv")).nodes_df
    assert nodes["network"].tolist()[:2] == ["None", "Default"] and pd.isna(nodes["network"].iloc[2])
    assert pd.api.types.is_float_dtype(nodes["score"]) and pd.isna(nodes["score"].iloc[1])


def test_load_xcpd_flat_records_what_it_read_and_dropped(tmp_path):
    flat_dir, atlas_path = _xcpd_flat_dir(tmp_path, bad_sub01=["r3"])
    with pytest.warns(UserWarning):
        _, atlas = load_xcpd_flat(flat_dir, atlas_path, atlas="Gordon", bad_node_threshold=0.5, verbose=False)
    record = atlas.attrs["brainnet3d_input"]
    assert record["source"] == "XCP-D"
    assert (record["atlas"], record["space"], record["task"], record["session"]) == ("Gordon", "fsLR", "rest", "ses-01")
    assert record["n_loaded"] == 2
    assert (record["bad_node_threshold"], record["drop_mode"], record["dropped"]) == (0.5, "union", ["r3"])


@pytest.mark.parametrize("mode", ["union", "intersection"])
def test_load_group_records_the_drop_rule(mode):
    mats = {"sub-a": _matrix(["r1", "r4"]), "sub-b": _matrix(["r1"], seed=1)}
    with pytest.warns(UserWarning):
        ds = bnv.load_group(mats, _nodes(), bad_node_threshold=0.5, drop_mode=mode)
    record = ds.nodes_df.attrs["brainnet3d_input"]
    assert record["source"] == "matrix files"
    assert (record["n_loaded"], record["bad_node_threshold"], record["drop_mode"]) == (2, 0.5, mode)
    assert record["dropped"] == (["r1", "r4"] if mode == "union" else ["r1"])


@pytest.mark.parametrize("name, sep", [("m.csv", ","), ("m.tsv", "\t"), ("m.txt", " "), ("m.txt", "\t"), ("m.1D", " ")])
def test_load_reads_headerless_text_in_node_table_order(tmp_path, name, sep):
    mat = _matrix(["r2"])
    np.savetxt(tmp_path / name, mat.to_numpy(), delimiter=sep)
    ds = bnv.load(str(tmp_path / name), _nodes())
    pd.testing.assert_frame_equal(ds.matrices["single"], mat, check_names=False)


def test_load_still_reads_labelled_text_with_numeric_labels(tmp_path):
    labels = [str(i) for i in range(1, 7)]
    mat = pd.DataFrame(_matrix().to_numpy(), index=labels, columns=labels)
    mat.to_csv(tmp_path / "m.csv")
    nodes = _nodes().assign(label=labels)
    pd.testing.assert_frame_equal(bnv.load(str(tmp_path / "m.csv"), nodes).matrices["single"], mat, check_names=False)


def test_load_reads_npy_and_arrays(tmp_path):
    mat = _matrix()
    np.save(tmp_path / "m.npy", mat.to_numpy())
    for src in (str(tmp_path / "m.npy"), mat.to_numpy()):
        pd.testing.assert_frame_equal(bnv.load(src, _nodes()).matrices["single"], mat, check_names=False)


def test_load_group_accepts_arrays_in_dict():
    mats = {"sub-a": _matrix().to_numpy(), "sub-b": _matrix(seed=1)}
    ds = bnv.load_group(mats, _nodes())
    assert all(m.columns.tolist() == LABELS for m in ds.matrices.values())
    np.testing.assert_array_equal(ds.matrices["sub-a"].to_numpy(), _matrix().to_numpy())


def test_headerless_size_mismatch_is_an_error():
    with pytest.raises(bnv.exceptions.DataValidationError, match="node table"):
        bnv.load(_matrix().to_numpy(), _nodes().iloc[:5])


def test_load_reads_mat_files(tmp_path):
    from scipy.io import savemat

    mat = _matrix()
    savemat(tmp_path / "one.mat", {"Z": mat.to_numpy()})
    savemat(tmp_path / "two.mat", {"r": mat.to_numpy(), "p": np.ones((6, 6))})
    pd.testing.assert_frame_equal(bnv.load(str(tmp_path / "one.mat"), _nodes()).matrices["single"], mat, check_names=False)
    with pytest.raises(bnv.exceptions.DataValidationError, match=r"mat_key[\s\S]*'p'[\s\S]*'r'"):
        bnv.load(str(tmp_path / "two.mat"), _nodes())
    ds = bnv.load(str(tmp_path / "two.mat"), _nodes(), mat_key="r")
    pd.testing.assert_frame_equal(ds.matrices["single"], mat, check_names=False)


def _cifti_parcels(labels):
    import nibabel as nib

    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(len(labels), bool), name="CortexLeft")
    return nib.cifti2.ParcelsAxis.from_brain_models([(lbl, bm[i:i + 1]) for i, lbl in enumerate(labels)])


def test_load_reads_cifti_pconn_with_its_parcel_names(tmp_path):
    import nibabel as nib

    order = LABELS[::-1]
    mat = _matrix().loc[order, order]
    parcels = _cifti_parcels(order)
    img = nib.Cifti2Image(mat.to_numpy(), nib.cifti2.Cifti2Header.from_axes((parcels, parcels)))
    nib.save(img, tmp_path / "sub-01.pconn.nii")
    got = bnv.load(str(tmp_path / "sub-01.pconn.nii"), _nodes()).matrices["single"]
    pd.testing.assert_frame_equal(got, mat, check_names=False)


def test_fisher_z_input_is_converted_back_to_r():
    mat = _matrix()
    z = np.arctanh(mat.to_numpy())
    ds = bnv.load_group({"sub-a": z}, _nodes(), values="z")
    np.testing.assert_allclose(ds.matrices["sub-a"].to_numpy(), mat.to_numpy())
    assert ds.nodes_df.attrs["brainnet3d_input"]["values"] == "z"
    with pytest.raises(ValueError, match="values"):
        bnv.load(mat, _nodes(), values="t")


TS_STEM = "_ses-01_task-rest_space-fsLR_seg-Gordon_stat-mean_timeseries.tsv"


def _series(seed):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((60, len(LABELS))) @ rng.standard_normal((len(LABELS), len(LABELS)))
    return pd.DataFrame(data, columns=LABELS)


def _xcpd_with_series(folder, missing_in_02=()):
    folder.mkdir(parents=True, exist_ok=True)
    series = {"01": _series(0), "02": _series(1)}
    for col in missing_in_02:
        series["02"][col] = np.nan
    for sub, ts in series.items():
        ts.to_csv(folder / f"sub-{sub}{TS_STEM}", sep="\t", index=False, na_rep="n/a")
        r = pd.DataFrame(np.corrcoef(ts.to_numpy().T), index=LABELS, columns=LABELS)
        r.to_csv(folder / f"sub-{sub}{XCPD_STEM}", sep="\t")
    return series


def _xcpd_atlas(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": LABELS, "network_label": ["a", "b"] * 3}).to_csv(path, sep="\t", index=False)
    return str(path)


def test_load_xcpd_flat_from_time_series_matches_relmat(tmp_path):
    series = _xcpd_with_series(tmp_path)
    atlas_path = _xcpd_atlas(tmp_path / "atlas" / "atlas.tsv")
    relmat, _ = load_xcpd_flat(str(tmp_path), atlas_path, atlas="Gordon", verbose=False)
    pearson, atlas = load_xcpd_flat(str(tmp_path), atlas_path, atlas="Gordon", verbose=False, connectivity="correlation")
    for sub in ("01", "02"):
        np.testing.assert_allclose(pearson[sub].to_numpy(), relmat[sub].to_numpy(), atol=1e-12)
    assert atlas.attrs["brainnet3d_input"]["connectivity"] == "correlation"
    partial, _ = load_xcpd_flat(str(tmp_path), atlas_path, atlas="Gordon", verbose=False,
                                connectivity="partial correlation")
    prec = np.linalg.inv(np.cov(series["01"].to_numpy().T))
    d = np.sqrt(np.diag(prec))
    off = ~np.eye(len(LABELS), dtype=bool)
    np.testing.assert_allclose(partial["01"].to_numpy()[off], (-prec / np.outer(d, d))[off], atol=1e-10)


def test_load_xcpd_tree_from_time_series_drops_missing_parcels(tmp_path):
    from brainnet3d.graph_theory import load_xcpd

    for sub in ("01", "02"):
        (tmp_path / f"sub-{sub}").mkdir()
    flat = tmp_path / "flat"
    _xcpd_with_series(flat, missing_in_02=["r5"])
    for f in flat.iterdir():
        sub = f.name.split("_")[0]
        (tmp_path / sub / "ses-01" / "func").mkdir(parents=True, exist_ok=True)
        f.replace(tmp_path / sub / "ses-01" / "func" / f.name)
    _xcpd_atlas(tmp_path / "atlases" / "atlas-Gordon" / "atlas-Gordon_dseg.tsv")
    with pytest.warns(UserWarning, match=r"\['r5'\]"):
        mats, atlas = load_xcpd(str(tmp_path), "Gordon", verbose=False, connectivity="correlation")
    assert all(m.columns.tolist() == LABELS[:5] for m in mats.values())
    record = atlas.attrs["brainnet3d_input"]
    assert (record["dropped"], record["connectivity"], record["shrinkage"]) == (["r5"], "correlation", False)


def test_load_group_subject_id_drops_the_extension(tmp_path):
    for sid in ("sub01", "sub02"):
        np.save(tmp_path / f"{sid}.npy", _matrix().to_numpy())
    ds = bnv.load_group(str(tmp_path), _nodes(), pattern="*.npy")
    assert sorted(ds.matrices) == ["sub01", "sub02"]


def test_load_group_takes_a_stack_of_matrices():
    stack = np.stack([_matrix(seed=s).to_numpy() for s in range(3)])
    ds = bnv.load_group(stack, _nodes(), subject_ids=["a", "b", "c"])
    assert list(ds.matrices) == ["a", "b", "c"]
    np.testing.assert_array_equal(ds.matrices["c"].to_numpy(), stack[2])
    assert list(bnv.load_group(stack, _nodes()).matrices) == ["sub-01", "sub-02", "sub-03"]
    with pytest.raises(ValueError, match="subject_ids"):
        bnv.load_group(stack, _nodes(), subject_ids=["a"])


def test_load_group_names_subjects_after_their_folders(tmp_path):
    for sid in ("s115", "s116"):
        (tmp_path / sid).mkdir()
        np.save(tmp_path / sid / "corr_000.npy", _matrix().to_numpy())
    ds = bnv.load_group(str(tmp_path), _nodes(), pattern="*/corr_000.npy")
    assert sorted(ds.matrices) == ["s115", "s116"]


def test_load_group_refuses_two_files_for_one_subject(tmp_path):
    for name in ("sub-01_run-1.npy", "sub-01_run-2.npy"):
        np.save(tmp_path / name, _matrix().to_numpy())
    with pytest.raises(bnv.exceptions.DataValidationError, match="sub-01"):
        bnv.load_group(str(tmp_path), _nodes(), pattern="*.npy")


AFNI_NAMES = [f"name_{lbl}" for lbl in LABELS]


def _write_netcc(path, blocks, names=AFNI_NAMES):
    # laid out as 3dNetCorr writes it
    n = len(names)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {n}  # Number of network ROIs\n# {len(blocks)}  # Number of netcc matrices\n# WITH_ROI_LABELS\n")
        f.write("".join(f" {s:>10s} \t" for s in names[:-1]) + f"  {names[-1]:>10s}\n")
        values = [10 * (i + 1) for i in range(n)]
        f.write("".join(f" {v:10d} \t" for v in values[:-1]) + f"  {values[-1]:10d}\n")
        for name, mat in blocks.items():
            f.write(f"# {name}\n")
            for row in mat:
                f.write("\t".join(f"{x:12.4f}" for x in row) + "\n")


def _afni_nodes():
    return pd.DataFrame({"label": AFNI_NAMES, "x": 0.0, "y": 0.0, "z": 0.0})


def test_load_reads_afni_netcc_blocks(tmp_path):
    r = np.round(_matrix().to_numpy(), 4)
    np.fill_diagonal(r, 1.0)
    z = np.round(np.arctanh(np.clip(r, -0.9999, 0.9999)), 4)
    partial = np.round(r / 2, 4)
    _write_netcc(tmp_path / "corr_000.netcc", {"CC": r, "FZ": z, "PC": partial})
    cc = bnv.load(str(tmp_path / "corr_000.netcc"), _afni_nodes()).matrices["single"]
    assert cc.columns.tolist() == AFNI_NAMES
    np.testing.assert_array_equal(cc.to_numpy(), r)
    fz = bnv.load(str(tmp_path / "corr_000.netcc"), _afni_nodes(), values="z").matrices["single"]
    np.testing.assert_allclose(fz.to_numpy(), np.tanh(z))
    pc = bnv.load(str(tmp_path / "corr_000.netcc"), _afni_nodes(), mat_key="PC").matrices["single"]
    np.testing.assert_array_equal(pc.to_numpy(), partial)
    with pytest.raises(bnv.exceptions.DataValidationError, match=r"'PCB'[\s\S]*CC"):
        bnv.load(str(tmp_path / "corr_000.netcc"), _afni_nodes(), mat_key="PCB")
