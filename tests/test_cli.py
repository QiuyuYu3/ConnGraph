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
    graph_cli.main([str(root), str(out), "--atlas", "Toy", "--coords", str(coords), "--network-graph-method", "full",
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


def test_graph_command_reads_a_folder_of_matrix_files(mock_dir, tmp_path):
    out = tmp_path / "graph"
    graph_cli.main([str(mock_dir), str(out), "--input-format", "matrix", "--nodes", str(mock_dir / "nodes.csv"),
                    "--network-col", "network", "--level", "node", "--graph-method", "density",
                    "--graph-param", "density=0.2,0.3", "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"])
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["levels"]["node"]["graph_params"] == {"density": [0.2, 0.3]}
    assert params["input"]["source"] == "matrix files"
    assert not (out / "graph_report.html").exists()


def test_graph_command_rejects_unknown_graph_param_syntax(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--atlas", "Toy", "--graph-param", "density", "--quiet"])


def test_nbs_command_compares_two_groups_from_a_table(xcpd, tmp_path):
    root, coords, groups = xcpd
    out = tmp_path / "nbs"
    nbs_cli.main([str(root), str(out), "--atlas", "Toy", "--groups", str(groups), "--group-column", "dx",
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


def test_nbs_command_reports_groups_missing_from_the_data(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": ["sub-01", "sub-99"], "dx": ["A", "B"]}).to_csv(table, sep="\t", index=False)
    with pytest.raises(SystemExit, match="sub-99"):
        nbs_cli.main([str(root), str(tmp_path / "out"), "--atlas", "Toy", "--groups", str(table), "--group-column", "dx",
                      "--contrast", "A", "B", "--thresh", "1.0", "--quiet"])


FAST = ["--network-col", "network", "--level", "node", "--graph-method", "density", "--graph-param", "density=0.2",
        "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"]


def _toy_series(dataset, seed):
    rng = np.random.default_rng(seed)
    n = len(dataset.nodes_df)
    return pd.DataFrame(rng.standard_normal((100, n)) @ rng.standard_normal((n, n)), columns=dataset.nodes_df["label"])


def test_graph_command_reads_a_folder_of_time_series(dataset, mock_dir, tmp_path):
    folder = tmp_path / "ts"
    folder.mkdir()
    for i in range(3):
        np.savetxt(folder / f"sub-{i:02d}_timeseries.txt", _toy_series(dataset, i).to_numpy())
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "timeseries", "--nodes", str(mock_dir / "nodes.csv"),
                    "--pattern", "*_timeseries.txt", "--connectivity", "partial-correlation", "--shrinkage"] + FAST)
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
    pd.DataFrame({"label": nodes["label"], "network": nodes["network"],
                  "hemisphere": nodes["hemisphere"]}).to_csv(atlas_dir / "atlas-Toy_dseg.tsv", sep="\t", index=False)
    out = tmp_path / "graph"
    graph_cli.main([str(tmp_path / "xcpd"), str(out), "--atlas", "Toy", "--connectivity", "correlation"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert (params["input"]["source"], params["input"]["connectivity"]) == ("XCP-D", "correlation")


def test_graph_command_reads_unlabelled_fisher_z_matrices(dataset, mock_dir, tmp_path):
    folder = tmp_path / "z"
    folder.mkdir()
    for sid, mat in dataset.matrices.items():
        z = np.arctanh(np.clip(mat.to_numpy(float), -0.999, 0.999))
        np.save(folder / f"{sid}.npy", z)
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-format", "matrix", "--nodes", str(mock_dir / "nodes.csv"),
                    "--pattern", "*.npy", "--values", "z"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["input"]["values"] == "z"
    assert params["subjects"] == sorted(dataset.matrices)


def test_connectivity_options_need_time_series(mock_dir, tmp_path):
    with pytest.raises(SystemExit):
        graph_cli.main([str(mock_dir), str(tmp_path), "--input-format", "matrix", "--nodes",
                        str(mock_dir / "nodes.csv"), "--connectivity", "correlation"] + FAST)


def test_fisher_z_is_applied_before_nbs(xcpd):
    from brainnet3d.cli._shared import fisher_z

    m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
    z = fisher_z({"s": m})["s"]
    assert z.iloc[0, 1] == pytest.approx(np.arctanh(0.5)) and z.iloc[0, 0] == 0
