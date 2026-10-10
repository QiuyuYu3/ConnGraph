import json
import warnings

import numpy as np
import pandas as pd
import pytest

import conngraph as bnv
from conngraph.cli import _shared
from conngraph.cli import main as cli
from conngraph.derivatives import MNE_MEASURES
from conngraph.exceptions import DataValidationError

mne = pytest.importorskip("mne")
mne_connectivity = pytest.importorskip("mne_connectivity")

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]
BANDS = dict(fmin=(8, 13), fmax=(13, 30), faverage=True)
OPTODES = ["S1_D1", "S1_D2", "S2_D1", "S2_D2"]
NIRS = dict(channels=[f"{o} {c}" for c in ("hbo", "hbr") for o in OPTODES], types=["hbo"] * 4 + ["hbr"] * 4)


def _epochs(seed, n_epochs, channels=CHANNELS, types="eeg"):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_epochs, len(channels), 250))
    data[:, 1] += 0.5 * data[:, 0]
    return mne.EpochsArray(data, mne.create_info(channels, 125.0, types), verbose=False)


def _save(conn, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # h5netcdf notes that these files are not strict netCDF-4
        conn.save(path)
    return str(path)


def _files(folder, method="wpli", n=3, tag="", channels=CHANNELS, types="eeg", **options):
    folder.mkdir(exist_ok=True)
    return {f"sub-{s:02d}": _save(mne_connectivity.spectral_connectivity_epochs(
        _epochs(s, 10 + s, channels, types), method=method, verbose=False, **(options or BANDS)),
        folder / f"sub-{s:02d}{tag}_connectivity.nc") for s in range(1, n + 1)}


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


def test_a_file_with_one_band_is_read(tmp_path):
    files = _files(tmp_path, n=1, fmin=(8,), fmax=(13,), faverage=True)
    assert bnv.mne_connectivity_bands(files["sub-01"]) == [(8.0, 13.0)]
    matrices, _ = bnv.load_mne_connectivity(files, (8.0, 13.0), verbose=False)
    assert matrices["sub-01"].shape == (len(CHANNELS), len(CHANNELS))


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


def test_standard_electrode_names_get_template_coordinates(tmp_path):
    files = _files(tmp_path, n=1)
    _, nodes = bnv.load_mne_connectivity(files, (8.0, 13.0), verbose=False)
    assert nodes["label"].tolist() == CHANNELS and not nodes[["x", "y", "z"]].isna().any().any()
    np.testing.assert_allclose(nodes.set_index("label").loc["Cz", ["x", "y", "z"]].to_numpy(float), [0.4, -9.2, 100.2],
                               atol=1.0)
    assert nodes.attrs["conngraph_input"]["electrodes"] == "colin27_1005"
    _, bare = bnv.load_mne_connectivity(files, (8.0, 13.0), montage=None, verbose=False)
    assert "x" not in bare and "electrodes" not in bare.attrs["conngraph_input"]


def test_given_coordinates_are_kept_and_not_mixed_with_a_template(tmp_path):
    files = _files(tmp_path, n=1)
    table = pd.DataFrame({"label": CHANNELS, "x": 1.0, "y": 2.0, "z": 3.0})
    _, nodes = bnv.load_mne_connectivity(files, (8.0, 13.0), nodes=table, verbose=False)
    assert (nodes["x"] == 1.0).all() and "electrodes" not in nodes.attrs["conngraph_input"]
    with pytest.raises(DataValidationError, match="already has x, y, z"):
        bnv.load_mne_connectivity(files, (8.0, 13.0), nodes=table, montage="colin27_1020", verbose=False)


def test_template_coordinates_match_names_in_any_case_and_skip_unknown_names():
    coords = bnv.montage_coordinates(["CZ", "EXG1", "oz"])
    assert coords["label"].tolist() == ["CZ", "oz"] and list(coords.columns) == ["label", "x", "y", "z"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # old template names are mapped without MNE's renaming warning
        old = bnv.montage_coordinates(["Cz"], "standard_1020")
    np.testing.assert_allclose(old[["x", "y", "z"]], bnv.montage_coordinates(["Cz"], "colin27_1020")[["x", "y", "z"]])
    with pytest.raises(DataValidationError, match="--coords"):
        bnv.montage_coordinates(["Cz"], "biosemi64")


@pytest.mark.parametrize("options, message", [
    (["--input-type", "xcpd", "--modality", "fmri"], "xcpd input is fmri"),
    (["--input-type", "nirspipe", "--modality", "eeg"], "nirspipe input is fnirs"),
    (["--input-type", "matrix"], "--modality is required for matrix input"),
    (["--input-type", "mne-connectivity", "--modality", "fmri"], "mne-connectivity input is eeg, meg or fnirs"),
    (["--input-type", "matrix", "--modality", "fmri", "--montage", "colin27_1020"], "--montage needs --modality eeg"),
    (["--input-type", "mne-connectivity", "--modality", "eeg", "--montage", "colin27_1020", "--coords", "c.tsv"],
     "either with --montage or with --coords"),
    (["--input-type", "matrix", "--modality", "eeg"], "choose --fisher-z or --no-fisher-z"),
    (["--input-type", "matrix", "--modality", "meg"], "choose --fisher-z or --no-fisher-z"),
    (["--input-type", "matrix", "--modality", "fmri", "--fisher-z", "--no-fisher-z"], "not both"),
    (["--input-type", "mne-connectivity", "--modality", "eeg", "--no-fisher-z"], "set by the measure"),
])
def test_cli_modality_rules(options, message, capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["in", "out", "group", *options])
    assert message in capsys.readouterr().err


def test_cli_modality_is_inferred_or_given_and_eeg_matrices_may_choose_fisher_z():
    assert cli.parse_args(["in", "out", "group", "--input-type", "xcpd"])[0].modality == "fmri"
    assert cli.parse_args(["in", "out", "group", "--input-type", "nirspipe"])[0].modality == "fnirs"
    args, _ = cli.parse_args(["in", "out", "group", "--input-type", "matrix", "--modality", "eeg", "--fisher-z"])
    assert args.no_fisher_z is False
    args, _ = cli.parse_args(["in", "out", "group", "--input-type", "matrix", "--modality", "eeg", "--values", "z"])
    assert args.no_fisher_z is False


def _matrix_folder(folder):
    folder.mkdir()
    rng = np.random.default_rng(0)
    for s in range(1, 7):
        m = rng.uniform(0.1, 0.9, (len(CHANNELS), len(CHANNELS)))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0)
        pd.DataFrame(m, index=CHANNELS, columns=CHANNELS).to_csv(folder / f"sub-{s:02d}_matrix.tsv", sep="	")
    pd.DataFrame({"label": CHANNELS, "network": ["front", "front", "back", "back"] * 2}).to_csv(
        folder / "nodes.tsv", sep="	", index=False)
    return folder


@pytest.mark.parametrize("modality, placed", [("eeg", True), ("fmri", False)])
def test_only_eeg_matrix_input_gets_template_electrode_positions(tmp_path, modality, placed):
    folder = _matrix_folder(tmp_path / "in")
    args, parser = cli.parse_args([str(folder), str(tmp_path / "out"), "group", "--input-type", "matrix",
                                   "--modality", modality, "--no-fisher-z", "--quiet"])
    _, nodes = _shared.load_input(args, parser)
    record = nodes.attrs["conngraph_input"]
    assert record["modality"] == modality
    assert ({"x", "y", "z"} <= set(nodes.columns)) is placed and ("electrodes" in record) is placed


def test_mne_connectivity_input_places_electrodes_only_for_eeg(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=1)
    for modality, placed in (("eeg", True), ("fnirs", False)):
        args, parser = cli.parse_args([str(folder), str(tmp_path / "out"), "group", "--input-type", "mne-connectivity",
                                       "--modality", modality, "--quiet"])
        _, nodes = _shared.load_input(args, parser, "meas-wpli_band-8to13Hz")
        assert ("x" in nodes) is placed and nodes.attrs["conngraph_input"]["modality"] == modality


def test_eeg_reports_speak_of_electrodes(tmp_path):
    folder = _matrix_folder(tmp_path / "in")
    out = tmp_path / "out"
    common = ["--input-type", "matrix", "--modality", "eeg", "--no-fisher-z", "--quiet"]
    assert cli.main([str(folder), str(out), "participant", *common, "--graph-method", "density", "--graph-param",
                     "density=0.5", "--metrics", "strength", "--level", "node", "--n-jobs", "1"]) == 0
    assert cli.main([str(folder), str(out), "group", *common]) == 0
    html = (out / "group" / "graph_report.html").read_text(encoding="utf-8")
    assert "Mean over participants of each electrode" in html and "highest electrodes" in html.lower()
    assert "electrode" in (out / "sub-01.html").read_text(encoding="utf-8")
    assert "matrix files (EEG, electrode positions from MNE's colin27_1005 template" in html


def test_cli_runs_each_band_without_fisher_z(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=6)
    pd.DataFrame({"label": CHANNELS, "network": ["front", "front", "back", "back"] * 2}).to_csv(
        folder / "nodes.tsv", sep="	", index=False)
    out = tmp_path / "out"
    mne_input = ["--input-type", "mne-connectivity", "--modality", "eeg"]
    fast = [*mne_input, "--graph-method", "density", "--graph-param", "density=0.5", "--metrics", "strength",
            "--n-jobs", "1", "--no-report", "--quiet"]
    assert cli.main([str(folder), str(out), "participant", *fast]) == 0
    assert cli.main([str(folder), str(out), "group", *mne_input, "--no-report", "--quiet"]) == 0
    for band in ("band-8to13Hz", "band-13to30Hz"):
        params = json.loads((out / "group" / "meas-wpli" / band / "parameters.json").read_text(encoding="utf-8"))
        assert params["options"]["apply_fisher_z"] is False
        assert (params["input"]["measure"], params["input"]["source"]) == ("wpli", "MNE-Connectivity")
        assert "node" in params["levels"]
        assert (params["input"]["electrodes"], params["input"]["modality"]) == ("colin27_1005", "eeg")
        saved = pd.read_csv(out / "group" / "meas-wpli" / band / "nodes.tsv", sep="	")
        assert not saved[["x", "y", "z"]].isna().any().any()


def test_cli_runs_each_measure_and_band_in_its_own_folder(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=6, tag="_desc-wpli")
    _files(folder, method="coh", n=6, tag="_desc-coh", fmin=(8,), fmax=(13,), faverage=True)
    out = tmp_path / "out"
    mne_input = ["--input-type", "mne-connectivity", "--modality", "eeg", "--quiet"]
    assert cli.main([str(folder), str(out), "participant", *mne_input, "--graph-method", "density", "--graph-param",
                     "density=0.5", "--metrics", "strength", "--level", "node", "--n-jobs", "1", "--no-report"]) == 0
    assert cli.main([str(folder), str(out), "group", *mne_input, "--no-report"]) == 0
    for measure, bands in (("wpli", ("band-8to13Hz", "band-13to30Hz")), ("coh", ("band-8to13Hz",))):
        for band in bands:
            params = json.loads((out / "group" / f"meas-{measure}" / band / "parameters.json").read_text(encoding="utf-8"))
            assert params["input"]["measure"] == measure and len(params["subjects"]) == 6
    assert not (out / "group" / "meas-coh" / "band-13to30Hz").exists()
    assert (out / "sub-01" / "sub-01_meas-coh_band-8to13Hz_metrics.json").exists()


def test_two_files_of_one_measure_for_a_participant_are_refused(tmp_path):
    folder = tmp_path / "in"
    for run in (1, 2):
        _files(folder, n=1, tag=f"_run-{run}", fmin=(8,), fmax=(13,), faverage=True)
    with pytest.raises(SystemExit, match="two files for sub-01"):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "mne-connectivity", "--modality",
                  "eeg", "--graph-method", "density", "--graph-param", "density=0.5", "--n-jobs", "1", "--no-report",
                  "--quiet"])


def test_hbo_and_hbr_channels_are_read_one_chromophore_at_a_time(tmp_path):
    files = _files(tmp_path, n=1, **NIRS)
    with pytest.raises(DataValidationError, match="HbO and HbR"):
        bnv.load_mne_connectivity(files, (8.0, 13.0), verbose=False)
    matrices, nodes = bnv.load_mne_connectivity(files, (8.0, 13.0), chromophore="hbr", verbose=False)
    assert nodes["label"].tolist() == OPTODES and list(matrices["sub-01"].columns) == OPTODES
    full = mne_connectivity.read_connectivity(files["sub-01"]).get_data("dense")[..., 0]
    np.testing.assert_allclose(np.tril(matrices["sub-01"].to_numpy(), -1), np.tril(full[4:, 4:], -1))
    assert nodes.attrs["conngraph_input"]["chromophore"] == "hbr"


def test_cli_analyses_each_chromophore_of_fnirs_files_in_its_own_folder(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=6, fmin=(8,), fmax=(13,), faverage=True, **NIRS)
    out = tmp_path / "out"
    nirs = ["--input-type", "mne-connectivity", "--modality", "fnirs", "--quiet"]
    assert cli.main([str(folder), str(out), "participant", *nirs, "--graph-method", "density", "--graph-param",
                     "density=0.5", "--metrics", "strength", "--n-jobs", "1", "--no-report"]) == 0
    assert cli.main([str(folder), str(out), "group", *nirs, "--no-report"]) == 0
    for chromo in ("hbo", "hbr"):
        params = json.loads((out / "group" / "meas-wpli" / f"chromo-{chromo}" / "band-8to13Hz" / "parameters.json")
                            .read_text(encoding="utf-8"))
        assert params["input"]["chromophore"] == chromo and params["input"]["modality"] == "fnirs"
    assert (out / "sub-01" / "sub-01_meas-wpli_chromo-hbo_band-8to13Hz_metrics.json").exists()
    only = tmp_path / "only"
    assert cli.main([str(folder), str(only), "participant", *nirs, "--chromophore", "hbr", "--graph-method",
                     "density", "--graph-param", "density=0.5", "--metrics", "strength", "--n-jobs", "1",
                     "--no-report"]) == 0
    assert [p.name for p in (only / "sub-01").glob("*_metrics.json")] == ["sub-01_meas-wpli_chromo-hbr_band-8to13Hz_metrics.json"]


def test_cli_refuses_hbo_and_hbr_channels_without_fnirs_modality(tmp_path):
    folder = tmp_path / "in"
    _files(folder, n=1, fmin=(8,), fmax=(13,), faverage=True, **NIRS)
    with pytest.raises(SystemExit, match="--modality fnirs"):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "mne-connectivity", "--modality",
                  "eeg", "--graph-method", "density", "--graph-param", "density=0.5", "--n-jobs", "1", "--no-report",
                  "--quiet"])
