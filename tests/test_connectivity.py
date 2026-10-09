import warnings

import numpy as np
import pandas as pd
import pytest

import brainnet3d as bnv
from brainnet3d.exceptions import DataValidationError

LABELS = [f"r{i}" for i in range(6)]


def _series(seed=0, n_time=80):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_time, len(LABELS))) @ rng.standard_normal((len(LABELS), len(LABELS)))
    return pd.DataFrame(data, columns=LABELS)


def _nodes():
    return pd.DataFrame({"label": LABELS, "x": 0.0, "y": 0.0, "z": 0.0})


def _partial(data):
    prec = np.linalg.inv(np.cov(np.asarray(data).T))
    d = np.sqrt(np.diag(prec))
    out = -prec / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def test_pearson_matches_numpy():
    ts = _series()
    mat = bnv.compute_connectivity({"sub-a": ts})["sub-a"]
    assert mat.index.tolist() == mat.columns.tolist() == LABELS
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(ts.to_numpy().T), atol=1e-12)


def test_partial_correlation_matches_inverse_covariance():
    ts = _series()
    mat = bnv.compute_connectivity({"sub-a": ts}, kind="partial correlation")["sub-a"]
    np.testing.assert_allclose(mat.to_numpy(), _partial(ts), atol=1e-10)


def test_shrinkage_uses_nilearn_default_estimator():
    from nilearn.connectome import ConnectivityMeasure

    ts = _series()
    mat = bnv.compute_connectivity({"sub-a": ts}, shrinkage=True)["sub-a"]
    ref = ConnectivityMeasure(kind="correlation").fit_transform([ts.to_numpy()])[0]
    np.testing.assert_allclose(mat.to_numpy(), ref, atol=1e-12)
    assert np.abs(mat.to_numpy() - np.corrcoef(ts.to_numpy().T)).max() > 1e-3


def test_missing_and_constant_regions_give_missing_rows():
    ts = _series()
    ts.loc[5, "r1"] = np.nan
    ts["r4"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mat = bnv.compute_connectivity({"sub-a": ts})["sub-a"]
    for lbl in ("r1", "r4"):
        assert mat.loc[lbl].isna().all() and mat[lbl].isna().all()
    keep = ["r0", "r2", "r3", "r5"]
    np.testing.assert_allclose(mat.loc[keep, keep].to_numpy(), np.corrcoef(ts[keep].to_numpy().T), atol=1e-12)


def test_time_points_missing_everywhere_are_skipped():
    ts = _series()
    full = ts.copy()
    ts.iloc[[3, 10, 11]] = np.nan
    mat = bnv.compute_connectivity({"sub-a": ts})["sub-a"]
    kept = full.drop(index=[3, 10, 11]).to_numpy()
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(kept.T), atol=1e-12)


def test_partial_correlation_needs_more_time_points_than_regions():
    with pytest.raises(DataValidationError, match="shrinkage"):
        bnv.compute_connectivity({"sub-a": _series(n_time=6)}, kind="partial correlation")
    bnv.compute_connectivity({"sub-a": _series(n_time=6)}, kind="partial correlation", shrinkage=True)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError, match="kind"):
        bnv.compute_connectivity({"sub-a": _series()}, kind="tangent")


def test_load_timeseries_from_labelled_tsv_folder(tmp_path):
    for i, sid in enumerate(("sub-01", "sub-02")):
        _series(seed=i).to_csv(tmp_path / f"{sid}_timeseries.tsv", sep="\t", index=False)
    ds = bnv.load_timeseries(str(tmp_path), _nodes(), pattern="*_timeseries.tsv")
    assert sorted(ds.matrices) == ["sub-01", "sub-02"]
    np.testing.assert_allclose(ds.matrices["sub-02"].to_numpy(), np.corrcoef(_series(seed=1).to_numpy().T), atol=1e-12)
    record = ds.nodes_df.attrs["brainnet3d_input"]
    assert (record["source"], record["connectivity"], record["shrinkage"]) == ("time series", "correlation", False)
    assert record["n_loaded"] == 2


@pytest.mark.parametrize("name", ["s.csv", "s.tsv", "s.txt", "s.1D", "s.npy", "s.mat", "array"])
def test_load_timeseries_reads_unlabelled_data_in_node_table_order(tmp_path, name):
    from scipy.io import savemat

    ts = _series()
    path = tmp_path / name
    if name == "array":
        src = ts.to_numpy()
    elif name.endswith(".npy"):
        np.save(path, ts.to_numpy())
    elif name.endswith(".mat"):
        savemat(path, {"ROISignals": ts.to_numpy()})
    else:
        sep = {".csv": ",", ".tsv": "\t"}.get(path.suffix, " ")
        np.savetxt(path, ts.to_numpy(), delimiter=sep, header="comment" if name.endswith(".1D") else "")
    src = ts.to_numpy() if name == "array" else str(path)
    mat = bnv.load_timeseries({"sub-a": src}, _nodes()).matrices["sub-a"]
    assert mat.columns.tolist() == LABELS
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(ts.to_numpy().T), atol=1e-12)


def test_load_timeseries_detects_a_header_of_numeric_labels(tmp_path):
    labels = [str(i) for i in range(1, 7)]
    ts = _series().set_axis(labels, axis=1)
    ts.to_csv(tmp_path / "s.csv", index=False)
    mat = bnv.load_timeseries({"sub-a": str(tmp_path / "s.csv")}, _nodes().assign(label=labels)).matrices["sub-a"]
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(ts.to_numpy().T), atol=1e-12)


def test_load_timeseries_reads_cifti_ptseries_with_its_parcel_names(tmp_path):
    import nibabel as nib

    order = LABELS[::-1]
    ts = _series()[order]
    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(len(order), bool), name="CortexLeft")
    parcels = nib.cifti2.ParcelsAxis.from_brain_models([(lbl, bm[i:i + 1]) for i, lbl in enumerate(order)])
    series = nib.cifti2.SeriesAxis(start=0, step=0.8, size=len(ts))
    nib.save(nib.Cifti2Image(ts.to_numpy(), nib.cifti2.Cifti2Header.from_axes((series, parcels))),
             tmp_path / "sub-01.ptseries.nii")
    mat = bnv.load_timeseries({"sub-01": str(tmp_path / "sub-01.ptseries.nii")}, _nodes()).matrices["sub-01"]
    assert mat.columns.tolist() == order
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(ts.to_numpy().T), atol=1e-12)


def test_load_timeseries_drops_missing_regions_before_partial_correlation():
    a, b = _series(seed=0), _series(seed=1)
    a["r2"] = np.nan
    with pytest.warns(UserWarning, match=r"\['r2'\]"):
        ds = bnv.load_timeseries({"sub-a": a, "sub-b": b}, _nodes(), kind="partial correlation")
    keep = [lbl for lbl in LABELS if lbl != "r2"]
    assert ds.matrices["sub-b"].columns.tolist() == keep
    np.testing.assert_allclose(ds.matrices["sub-b"].to_numpy(), _partial(b[keep]), atol=1e-10)
    assert ds.nodes_df.attrs["brainnet3d_input"]["dropped"] == ["r2"]
    assert ds.nodes_df["label"].tolist() == keep


def test_load_timeseries_tolerates_a_region_only_some_files_have():
    a, b = _series(seed=0), _series(seed=1)
    a["extra"] = np.nan
    with pytest.warns(UserWarning, match=r"\['extra'\]"):
        ds = bnv.load_timeseries({"sub-a": a, "sub-b": b}, _nodes())
    assert all(m.columns.tolist() == LABELS for m in ds.matrices.values())


def test_load_timeseries_column_count_must_match_node_table():
    with pytest.raises(DataValidationError, match="node table"):
        bnv.load_timeseries({"sub-a": _series().to_numpy()}, _nodes().iloc[:5])


def test_load_timeseries_takes_a_list_or_stack_of_arrays():
    arrays = [_series(seed=s).to_numpy() for s in range(2)]
    for given in (arrays, np.stack(arrays)):
        ds = bnv.load_timeseries(given, _nodes(), subject_ids=["a", "b"])
        assert list(ds.matrices) == ["a", "b"]
        np.testing.assert_allclose(ds.matrices["b"].to_numpy(), np.corrcoef(arrays[1].T), atol=1e-12)
    assert list(bnv.load_timeseries(arrays, _nodes()).matrices) == ["sub-01", "sub-02"]


@pytest.mark.parametrize("with_labels", [False, True])
def test_load_timeseries_reads_afni_netts_one_region_per_row(tmp_path, with_labels):
    ts = _series()
    with open(tmp_path / "corr_000.netts", "w", encoding="utf-8") as f:
        for i, col in enumerate(LABELS):
            values = "\t".join(f"{x:.3e}" for x in ts[col])
            f.write((f"{10 * (i + 1)}\t" if with_labels else "") + values + "\n")
    rounded = np.array([[float(f"{x:.3e}") for x in ts[col]] for col in LABELS]).T
    mat = bnv.load_timeseries({"s115": str(tmp_path / "corr_000.netts")}, _nodes()).matrices["s115"]
    np.testing.assert_allclose(mat.to_numpy(), np.corrcoef(rounded.T), atol=1e-12)


def test_load_timeseries_names_subjects_after_their_folders(tmp_path):
    for i, sid in enumerate(("s115", "s116")):
        (tmp_path / sid).mkdir()
        _series(seed=i).to_csv(tmp_path / sid / "ts.csv", index=False)
    ds = bnv.load_timeseries(str(tmp_path), _nodes(), pattern="*/ts.csv")
    assert sorted(ds.matrices) == ["s115", "s116"]
