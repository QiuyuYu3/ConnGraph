import json
import warnings

import numpy as np
import pandas as pd
import pytest

import conngraph as bnv
from conngraph.cli import main as cli
from conngraph.derivatives import MNE_MEASURES
from conngraph.exceptions import DataValidationError

mne = pytest.importorskip("mne")
mne_connectivity = pytest.importorskip("mne_connectivity")

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
BANDS = dict(fmin=(8, 13), fmax=(13, 30), faverage=True)


def _epochs(seed, n_epochs):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_epochs, len(CHANNELS), 250))
    data[:, 1] += 0.5 * data[:, 0]
    return mne.EpochsArray(data, mne.create_info(CHANNELS, 125.0, "eeg"), verbose=False)


def _save(conn, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # h5netcdf notes that these files are not strict netCDF-4
        conn.save(path)
    return str(path)


def _files(folder, method="wpli", n=3, **options):
    folder.mkdir(exist_ok=True)
    return {f"sub-{s:02d}": _save(mne_connectivity.spectral_connectivity_epochs(
        _epochs(s, 10 + s), method=method, verbose=False, **(options or BANDS)), folder / f"sub-{s:02d}_connectivity.nc")
        for s in range(1, n + 1)}


def test_each_band_is_read_as_a_symmetric_matrix(tmp_path):
    files = _files(tmp_path)
    assert bnv.mne_connectivity_bands(files["sub-01"]) == [(8.0, 13.0), (13.0, 30.0)]
    matrices, nodes = bnv.load_mne_connectivity(files, (13.0, 30.0), verbose=False)
    m = matrices["sub-01"].to_numpy()
    assert list(matrices["sub-01"].columns) == CHANNELS and nodes["label"].tolist() == CHANNELS
    np.testing.assert_allclose(m, m.T)
    stored = mne_connectivity.read_connectivity(files["sub-01"]).get_data("dense")[..., 1]
    np.testing.assert_allclose(np.tril(m, -1), np.tril(stored, -1))
    record = nodes.attrs["conngraph_input"]
    assert (record["source"], record["measure"], record["band"], record["epochs"]) == (
        "MNE-Connectivity", "wpli", [13.0, 30.0], [11, 13])


def test_imaginary_coherence_is_taken_as_its_absolute_value(tmp_path):
    files = _files(tmp_path, method="imcoh", n=1)
    stored = mne_connectivity.read_connectivity(files["sub-01"]).get_data("dense")[..., 0]
    assert (np.tril(stored, -1) < 0).any()
    matrices, _ = bnv.load_mne_connectivity(files, (8.0, 13.0), verbose=False)
    m = matrices["sub-01"].to_numpy()
    np.testing.assert_allclose(m, m.T)
    np.testing.assert_allclose(np.tril(m, -1), np.abs(np.tril(stored, -1)))


def test_node_table_rows_follow_the_files(tmp_path):
    files = _files(tmp_path, n=1)
    table = pd.DataFrame({"label": CHANNELS[::-1] + ["EOG"], "network": ["a", "b"] * 4 + ["c"]})
    _, nodes = bnv.load_mne_connectivity(files, (8.0, 13.0), nodes=table, verbose=False)
    assert nodes["label"].tolist() == CHANNELS
    assert nodes.set_index("label")["network"].to_dict() == table.set_index("label")["network"].drop("EOG").to_dict()
    with pytest.raises(DataValidationError, match="no row for Fz"):
        bnv.load_mne_connectivity(files, (8.0, 13.0), nodes=table[table["label"] != "Fz"], verbose=False)


@pytest.mark.parametrize("method, options, message", [
    ("dpli", None, "supported undirected methods"),
    ("wpli", dict(fmin=8, fmax=30), "faverage=True"),
])
def test_unsupported_files_are_refused(tmp_path, method, options, message):
    files = _files(tmp_path, method=method, n=1, **(options or {}))
    with pytest.raises(DataValidationError, match=message):
        bnv.load_mne_connectivity(files, (8.0, 13.0), verbose=False)


def test_envelope_correlation_must_be_averaged_over_epochs(tmp_path):
    conn = mne_connectivity.envelope_correlation(_epochs(1, 6).filter(8, 13, verbose=False))
    epoched = {"sub-01": _save(conn, tmp_path / "sub-01_connectivity.nc")}
    with pytest.raises(DataValidationError, match="one matrix per epoch"):
        bnv.load_mne_connectivity(epoched, verbose=False)
    combined = {"sub-01": _save(conn.combine(), tmp_path / "sub-02_connectivity.nc")}
    matrices, nodes = bnv.load_mne_connectivity(combined, verbose=False)
    assert matrices["sub-01"].shape == (len(CHANNELS), len(CHANNELS))
    assert MNE_MEASURES[nodes.attrs["conngraph_input"]["measure"]][2]  # a correlation keeps Fisher z


def test_cli_runs_each_band_without_fisher_z(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=6)
    pd.DataFrame({"label": CHANNELS, "network": ["front", "front", "back", "back"] * 2}).to_csv(
        folder / "nodes.tsv", sep="\t", index=False)
    out = tmp_path / "out"
    fast = ["--input-type", "mne", "--graph-method", "density", "--graph-param", "density=0.5", "--metrics", "strength",
            "--n-jobs", "1", "--no-report", "--quiet"]
    assert cli.main([str(folder), str(out), "participant", *fast]) == 0
    assert cli.main([str(folder), str(out), "group", "--input-type", "mne", "--no-report", "--quiet"]) == 0
    for band in ("band-8to13Hz", "band-13to30Hz"):
        params = json.loads((out / "group" / band / "parameters.json").read_text(encoding="utf-8"))
        assert params["options"]["apply_fisher_z"] is False
        assert (params["input"]["measure"], params["input"]["source"]) == ("wpli", "MNE-Connectivity")
        assert "node" in params["levels"]
