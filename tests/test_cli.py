import json

import numpy as np
import pandas as pd
import pytest

from brainnet3d.cli import graph as graph_cli
from brainnet3d.cli import nbs as nbs_cli

pytest.importorskip("bct")

STEM = "_ses-01_task-rest_space-fsLR_seg-Toy_stat-pearsoncorrelation_relmat.tsv"


@pytest.fixture(scope="module")
def xcpd(dataset, tmp_path_factory):
    """A minimal XCP-D derivatives tree built from the toy dataset, plus a coordinates table."""
    root = tmp_path_factory.mktemp("xcpd")
    for i, (sid, mat) in enumerate(sorted(dataset.matrices.items()), 1):
        func = root / f"sub-{i:02d}" / "ses-01" / "func"
        func.mkdir(parents=True)
        mat.to_csv(func / f"sub-{i:02d}{STEM}", sep="\t")
    atlas_dir = root / "atlases" / "atlas-Toy"
    atlas_dir.mkdir(parents=True)
    nodes = dataset.nodes_df
    pd.DataFrame({"index": range(1, len(nodes) + 1), "label": nodes["label"], "network_label": nodes["network"],
                  "hemisphere": nodes["hemisphere"]}).to_csv(atlas_dir / "atlas-Toy_dseg.tsv", sep="\t", index=False)
    coords = root / "coords.csv"
    nodes[["label", "x", "y", "z"]].to_csv(coords, index=False)
    groups = root / "participants.tsv"
    pd.DataFrame({"participant_id": [f"sub-{i:02d}" for i in range(1, 7)],
                  "dx": ["A", "A", "A", "B", "B", "B"]}).to_csv(groups, sep="\t", index=False)
    return root, coords, groups


def test_graph_command_writes_tables_parameters_description_and_report(xcpd, tmp_path):
    root, coords, _ = xcpd
    out = tmp_path / "graph"
    graph_cli.main([str(root), str(out), "--input-format", "xcpd", "--atlas", "Toy", "--coords", str(coords), "--network-graph-method", "full",
                    "--metrics", "strength", "clust_coeff", "--n-jobs", "1", "--no-static-brain", "--quiet"])
    assert (out / "node" / "strength.abs.csv").exists()
    assert (out / "network_hemi" / "clust_coeff.costantini.csv").exists()
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["command"].startswith("brainnet3d-graph ")
    assert params["input"]["source"] == "XCP-D" and params["input"]["atlas"] == "Toy"
    description = json.loads((out / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "brainnet3d"
    report = (out / "graph_report.html").read_text(encoding="utf-8")
    assert "Run command" in report and "derived from XCP-D (Toy atlas; fsLR space" in report
    assert "Option 2: interactive" in report


def test_graph_command_reads_a_folder_of_matrix_files(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="	")
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "matrix", "--level", "node", "--graph-method", "density",
                    "--graph-param", "density=0.2,0.3", "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"])
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["levels"]["node"]["graph_params"] == {"density": [0.2, 0.3]}
    assert params["input"]["source"] == "matrix files"
    assert not (out / "graph_report.html").exists()


def test_graph_command_rejects_unknown_graph_param_syntax(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--input-format", "xcpd", "--atlas", "Toy", "--graph-param", "density",
                        "--quiet"])


def test_nbs_command_compares_two_groups_from_a_table(xcpd, tmp_path):
    root, coords, groups = xcpd
    out = tmp_path / "nbs"
    nbs_cli.main([str(root), str(out), "--input-format", "xcpd", "--atlas", "Toy", "--groups", str(groups), "--group-column", "dx",
                  "--contrast", "A", "B", "--thresh", "1.0", "--perms", "20", "--seed", "0", "--coords", str(coords),
                  "--no-static-brain", "--quiet"])
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["groups"] == {"g1": ["01", "02", "03"], "g2": ["04", "05", "06"]}
    assert params["input"]["fisher_z"] is True and params["input"]["contrast"] == ["A", "B"]
    components = pd.read_csv(out / "nbs_components.tsv", sep="\t")
    assert list(components.columns) == ["component", "edges", "nodes", "p"]
    assert len(pd.read_csv(out / "nbs_null.tsv", sep="\t")) == 20
    report = (out / "nbs_report.html").read_text(encoding="utf-8")
    assert "Fisher z-transformed before testing" in report and "Run command" in report


@pytest.mark.parametrize("flag", ["--n-jobs", "--nprocs"])
def test_both_commands_take_a_worker_count(flag):
    common = ["in", "out", "--input-format", "xcpd", "--atlas", "Toy", flag, "3"]
    assert graph_cli.build_parser().parse_args(common).n_jobs == 3
    nbs_args = common + ["--groups", "g.tsv", "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1"]
    assert nbs_cli.build_parser().parse_args(nbs_args).n_jobs == 3


def test_nbs_command_passes_the_worker_count(xcpd, tmp_path, monkeypatch):
    from brainnet3d.graph_theory import nbs as nbs_module

    seen = {}
    original = nbs_module.run_nbs

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(nbs_module, "run_nbs", recording)
    root, _, groups = xcpd
    nbs_cli.main([str(root), str(tmp_path / "nbs"), "--input-format", "xcpd", "--atlas", "Toy", "--groups", str(groups),
                  "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0", "--perms", "20", "--seed", "0",
                  "--no-report", "--quiet", "--nprocs", "2"])
    assert seen["n_jobs"] == 2


def test_nbs_command_reports_groups_missing_from_the_data(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": ["sub-01", "sub-99"], "dx": ["A", "B"]}).to_csv(table, sep="\t", index=False)
    with pytest.raises(SystemExit, match="sub-99"):
        nbs_cli.main([str(root), str(tmp_path / "out"), "--input-format", "xcpd", "--atlas", "Toy", "--groups", str(table),
                      "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0", "--quiet"])


FAST = ["--level", "node", "--graph-method", "density", "--graph-param", "density=0.2",
        "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"]


def _layout(dataset, folder):
    """A matrix or time series input folder holding only the fixed-name node table."""
    folder.mkdir()
    dataset.nodes_df[["label", "network", "hemisphere", "x", "y", "z"]].to_csv(folder / "nodes.tsv", sep="\t",
                                                                               index=False)
    return folder


def test_input_format_is_required(xcpd, tmp_path, capsys):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--atlas", "Toy"] + FAST)
    assert "--input-format" in capsys.readouterr().err


@pytest.mark.parametrize("option", [["--input-format", "xcpd-flat"], ["--atlas-file", "a.tsv"], ["--nodes", "n.tsv"],
                                    ["--pattern", "*.csv"], ["--label-col", "x"], ["--network-col", "x"]])
def test_removed_input_options_are_rejected(xcpd, tmp_path, capsys, option):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--input-format", "xcpd", "--atlas", "Toy"] + FAST + option)
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err or "invalid choice" in err


def test_matrix_folder_is_read_by_fixed_names(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    sids = sorted(dataset.matrices)
    for k, sid in enumerate(sids):
        mat = dataset.matrices[sid]
        if k % 3 == 0:
            mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
        elif k % 3 == 1:
            np.save(folder / f"{sid}_matrix.npy", mat.to_numpy())
        else:
            mat.to_csv(folder / f"{sid}_matrix.csv")
    (folder / "notes_matrix.txt").write_text("not a participant")
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "matrix"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["subjects"] == sids
    assert params["input"]["source"] == "matrix files"


def test_matrix_folder_refuses_two_files_for_one_participant(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    np.save(folder / f"{sid}_matrix.npy", mat.to_numpy())
    with pytest.raises(SystemExit, match=sid):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-format", "matrix"] + FAST)


def test_matrix_folder_needs_nodes_tsv(dataset, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    with pytest.raises(SystemExit, match="nodes.tsv"):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-format", "matrix"] + FAST)


def _toy_series(dataset, seed):
    rng = np.random.default_rng(seed)
    n = len(dataset.nodes_df)
    return pd.DataFrame(rng.standard_normal((100, n)) @ rng.standard_normal((n, n)), columns=dataset.nodes_df["label"])


def test_graph_command_reads_a_folder_of_time_series(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "ts")
    for i in range(3):
        np.savetxt(folder / f"sub-{i:02d}_timeseries.txt", _toy_series(dataset, i).to_numpy())
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "timeseries", "--connectivity", "partial-correlation",
                    "--shrinkage"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["input"]["source"] == "time series"
    assert (params["input"]["connectivity"], params["input"]["shrinkage"]) == ("partial correlation", True)
    assert params["subjects"] == ["sub-00", "sub-01", "sub-02"]


def test_graph_command_computes_xcpd_connectivity_from_time_series(dataset, tmp_path):
    stem = "_ses-01_task-rest_space-fsLR_seg-Toy_stat-mean_timeseries.tsv"
    for i in range(1, 4):
        func = tmp_path / "xcpd" / f"sub-{i:02d}" / "ses-01" / "func"
        func.mkdir(parents=True)
        _toy_series(dataset, i).to_csv(func / f"sub-{i:02d}{stem}", sep="\t", index=False)
    atlas_dir = tmp_path / "xcpd" / "atlases" / "atlas-Toy"
    atlas_dir.mkdir(parents=True)
    nodes = dataset.nodes_df
    pd.DataFrame({"label": nodes["label"], "network_label": nodes["network"],
                  "hemisphere": nodes["hemisphere"]}).to_csv(atlas_dir / "atlas-Toy_dseg.tsv", sep="\t", index=False)
    out = tmp_path / "graph"
    graph_cli.main([str(tmp_path / "xcpd"), str(out), "--input-format", "xcpd", "--atlas", "Toy", "--connectivity",
                    "correlation"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert (params["input"]["source"], params["input"]["connectivity"]) == ("XCP-D", "correlation")


def test_graph_command_reads_unlabelled_fisher_z_matrices(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "z")
    for sid, mat in dataset.matrices.items():
        z = np.arctanh(np.clip(mat.to_numpy(float), -0.999, 0.999))
        np.save(folder / f"{sid}_matrix.npy", z)
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "matrix", "--values", "z"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["input"]["values"] == "z"
    assert params["subjects"] == sorted(dataset.matrices)


def test_connectivity_options_need_time_series(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    with pytest.raises(SystemExit):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-format", "matrix", "--connectivity",
                        "correlation"] + FAST)


def test_fisher_z_is_applied_before_nbs(xcpd):
    from brainnet3d.cli._shared import fisher_z

    m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
    z = fisher_z({"s": m})["s"]
    assert z.iloc[0, 1] == pytest.approx(np.arctanh(0.5)) and z.iloc[0, 0] == 0
