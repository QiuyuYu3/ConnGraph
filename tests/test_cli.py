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
    graph_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--coords", str(coords), "--network-graph-method", "full",
                    "--metrics", "strength", "clust_coeff", "--n-jobs", "1", "--no-static-brain", "--quiet"])
    out = out / "ses-01" / "atlas-Toy"
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
    graph_cli.main([str(folder), str(out), "--input-type", "matrix", "--level", "node", "--graph-method", "density",
                    "--graph-param", "density=0.2,0.3", "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"])
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["levels"]["node"]["graph_params"] == {"density": [0.2, 0.3]}
    assert params["input"]["source"] == "matrix files"
    assert not (out / "graph_report.html").exists()


def test_graph_command_rejects_unknown_graph_param_syntax(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--input-type", "xcpd", "--atlases", "Toy", "--graph-param", "density",
                        "--quiet"])


def test_nbs_command_compares_two_groups_from_a_table(xcpd, tmp_path):
    root, coords, groups = xcpd
    out = tmp_path / "nbs"
    nbs_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--groups", str(groups), "--group-column", "dx",
                  "--contrast", "A", "B", "--thresh", "1.0", "--perms", "20", "--seed", "0", "--coords", str(coords),
                  "--no-static-brain", "--quiet"])
    out = out / "ses-01" / "atlas-Toy"
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
    common = ["in", "out", "--input-type", "xcpd", "--atlases", "Toy", flag, "3"]
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
    nbs_cli.main([str(root), str(tmp_path / "nbs"), "--input-type", "xcpd", "--atlases", "Toy", "--groups", str(groups),
                  "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0", "--perms", "20", "--seed", "0",
                  "--no-report", "--quiet", "--nprocs", "2"])
    assert seen["n_jobs"] == 2


def test_nbs_command_reports_groups_missing_from_the_data(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": ["sub-01", "sub-99"], "dx": ["A", "B"]}).to_csv(table, sep="\t", index=False)
    with pytest.raises(SystemExit, match="sub-99"):
        nbs_cli.main([str(root), str(tmp_path / "out"), "--input-type", "xcpd", "--atlases", "Toy", "--groups", str(table),
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
        graph_cli.main([str(root), str(tmp_path / "out"), "--atlases", "Toy"] + FAST)
    assert "--input-type" in capsys.readouterr().err


@pytest.mark.parametrize("option", [["--input-type", "xcpd-flat"], ["--input-format", "xcpd"], ["--atlas-file", "a.tsv"], ["--nodes", "n.tsv"],
                                    ["--pattern", "*.csv"], ["--label-col", "x"], ["--network-col", "x"]])
def test_removed_input_options_are_rejected(xcpd, tmp_path, capsys, option):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        graph_cli.main([str(root), str(tmp_path / "out"), "--input-type", "xcpd", "--atlases", "Toy"] + FAST + option)
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
    graph_cli.main([str(folder), str(out), "--input-type", "matrix"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["subjects"] == sids
    assert params["input"]["source"] == "matrix files"


def test_matrix_folder_refuses_two_files_for_one_participant(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    np.save(folder / f"{sid}_matrix.npy", mat.to_numpy())
    with pytest.raises(SystemExit, match=sid):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-type", "matrix"] + FAST)


def test_matrix_folder_runs_without_coordinates(dataset, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    dataset.nodes_df[["label", "network"]].to_csv(folder / "nodes.tsv", sep="\t", index=False)
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-type", "matrix"] + [a for a in FAST if a != "--no-report"])
    assert "no x, y, z coordinates" in (out / "graph_report.html").read_text(encoding="utf-8")


def test_matrix_folder_needs_nodes_tsv(dataset, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    with pytest.raises(SystemExit, match="nodes.tsv"):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-type", "matrix"] + FAST)


def _toy_series(dataset, seed):
    rng = np.random.default_rng(seed)
    n = len(dataset.nodes_df)
    return pd.DataFrame(rng.standard_normal((100, n)) @ rng.standard_normal((n, n)), columns=dataset.nodes_df["label"])


def test_graph_command_reads_a_folder_of_time_series(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "ts")
    for i in range(3):
        np.savetxt(folder / f"sub-{i:02d}_timeseries.txt", _toy_series(dataset, i).to_numpy())
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-type", "timeseries", "--connectivity", "partial-correlation",
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
    graph_cli.main([str(tmp_path / "xcpd"), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--connectivity",
                    "correlation"] + FAST)
    params = json.loads((out / "ses-01" / "atlas-Toy" / "parameters.json").read_text(encoding="utf-8"))
    assert (params["input"]["source"], params["input"]["connectivity"]) == ("XCP-D", "correlation")


def test_graph_command_reads_unlabelled_fisher_z_matrices(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "z")
    for sid, mat in dataset.matrices.items():
        z = np.arctanh(np.clip(mat.to_numpy(float), -0.999, 0.999))
        np.save(folder / f"{sid}_matrix.npy", z)
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-type", "matrix", "--values", "z"] + FAST)
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["input"]["values"] == "z"
    assert params["subjects"] == sorted(dataset.matrices)


def test_connectivity_options_need_time_series(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    with pytest.raises(SystemExit):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-type", "matrix", "--connectivity",
                        "correlation"] + FAST)


def _fnirs_pipe_tree(dataset, root):
    """fnirs-pipe output for the toy participants: one matrix per chromophore, channels named after the toy nodes."""
    for sid, mat in dataset.matrices.items():
        folder = root / sid / "ses-01" / "nirs"
        folder.mkdir(parents=True)
        for chromo, sign in (("hbo", 1.0), ("hbr", -1.0)):
            names = [f"{c} {chromo}" for c in mat.columns]
            pd.DataFrame(sign * mat.to_numpy(), index=names, columns=names).to_csv(
                folder / f"{sid}_ses-01_task-rest_chromo-{chromo}_stat-pearson_relmat.tsv", sep="\t",
                index_label="channel")
    return root


def test_fnirs_pipe_input_writes_each_chromophore_on_its_own(dataset, tmp_path):
    root = _fnirs_pipe_tree(dataset, tmp_path / "fnirs")
    out = tmp_path / "graph"
    graph_cli.main([str(root), str(out), "--input-type", "fnirs-pipe", "--graph-method", "full",
                    "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"])
    for chromo in ("hbo", "hbr"):
        params = json.loads((out / "ses-01" / f"chromo-{chromo}" / "parameters.json").read_text(encoding="utf-8"))
        assert (params["input"]["source"], params["input"]["chromophore"]) == ("fnirs-pipe", chromo)
        assert list(params["levels"]) == ["node"]
    only = tmp_path / "hbr_only"
    graph_cli.main([str(root), str(only), "--input-type", "fnirs-pipe", "--chromophore", "hbr", "--session-id", "01",
                    "--graph-method", "full", "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"])
    assert sorted(p.name for p in (only / "ses-01").iterdir()) == ["chromo-hbr"]


def test_nbs_command_runs_each_atlas(xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "nbs"
    nbs_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--session-id", "ses-01",
                  "--groups", str(groups), "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0",
                  "--perms", "20", "--seed", "0", "--no-report", "--quiet"])
    assert (out / "ses-01" / "atlas-Toy" / "nbs_components.tsv").exists()


def test_fisher_z_is_applied_before_nbs(xcpd):
    from brainnet3d.cli._shared import fisher_z

    m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
    z = fisher_z({"s": m})["s"]
    assert z.iloc[0, 1] == pytest.approx(np.arctanh(0.5)) and z.iloc[0, 0] == 0


TAIL = "task-rest_space-fsLR_seg-Toy_stat-pearsoncorrelation_relmat.tsv"


def _xcpd_waves(dataset, root, sessions):
    """The toy participants as XCP-D output in each of the given sessions (None: no session level)."""
    for k, ses in enumerate(sessions):
        for i, (sid, mat) in enumerate(sorted(dataset.matrices.items()), 1):
            func = root / f"sub-{i:02d}" / (ses or "") / "func"
            func.mkdir(parents=True, exist_ok=True)
            (mat * (1 - 0.1 * k)).to_csv(func / (f"sub-{i:02d}_" + (f"{ses}_" if ses else "") + TAIL), sep="\t")
    atlas_dir = root / "atlases" / "atlas-Toy"
    atlas_dir.mkdir(parents=True)
    nodes = dataset.nodes_df
    pd.DataFrame({"label": nodes["label"], "network_label": nodes["network"],
                  "hemisphere": nodes["hemisphere"]}).to_csv(atlas_dir / "atlas-Toy_dseg.tsv", sep="\t", index=False)
    return root


def _session(folder):
    return json.loads((folder / "parameters.json").read_text(encoding="utf-8"))["input"]["session"]


def test_each_session_gets_its_own_result_folder(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02"])
    out = tmp_path / "graph"
    assert graph_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy"] + FAST) == 0
    assert sorted(p.name for p in out.iterdir()) == ["ses-01", "ses-02"]
    assert [_session(out / s / "atlas-Toy") for s in ("ses-01", "ses-02")] == ["ses-01", "ses-02"]


def test_session_id_selects_sessions_with_or_without_prefix(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02", "ses-03"])
    out = tmp_path / "graph"
    graph_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--session-id", "03", "ses-01"]
                   + FAST)
    assert sorted(p.name for p in out.iterdir()) == ["ses-01", "ses-03"]


def test_input_without_sessions_has_no_session_folder(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", [None])
    out = tmp_path / "graph"
    graph_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy"] + FAST)
    assert sorted(p.name for p in out.iterdir()) == ["atlas-Toy"]


def test_matrix_folder_with_sessions(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        for ses in ("ses-01", "ses-02"):
            mat.to_csv(folder / f"{sid}_{ses}_matrix.tsv", sep="\t")
    out = tmp_path / "graph"
    graph_cli.main([str(folder), str(out), "--input-type", "matrix"] + FAST)
    for ses in ("ses-01", "ses-02"):
        params = json.loads((out / ses / "parameters.json").read_text(encoding="utf-8"))
        assert params["subjects"] == sorted(dataset.matrices) and params["input"]["session"] == ses


def test_matrix_folder_mixing_files_with_and_without_session_is_refused(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    (a, ma), (b, mb) = list(dataset.matrices.items())[:2]
    ma.to_csv(folder / f"{a}_ses-01_matrix.tsv", sep="\t")
    mb.to_csv(folder / f"{b}_matrix.tsv", sep="\t")
    with pytest.raises(SystemExit, match="session"):
        graph_cli.main([str(folder), str(tmp_path / "out"), "--input-type", "matrix"] + FAST)


def test_a_failed_session_leaves_an_error_file_and_the_others_still_run(dataset, tmp_path, capsys):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_ses-01_matrix.tsv", sep="\t")
        np.save(folder / f"{sid}_ses-02_matrix.npy", np.ones((2, 3)))
    out = tmp_path / "graph"
    assert graph_cli.main([str(folder), str(out), "--input-type", "matrix"] + FAST) == 1
    assert (out / "ses-01" / "parameters.json").exists()
    assert "not square" in (out / "ses-02" / "error.txt").read_text(encoding="utf-8")
    assert "ses-02" in capsys.readouterr().err


def test_nbs_command_runs_each_session(dataset, xcpd, tmp_path):
    _, _, groups = xcpd
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02"])
    out = tmp_path / "nbs"
    nbs_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--groups", str(groups),
                  "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0", "--perms", "20", "--seed", "0",
                  "--n-jobs", "1", "--no-report", "--quiet"])
    for ses in ("ses-01", "ses-02"):
        assert (out / ses / "atlas-Toy" / "nbs_components.tsv").exists()
        assert _session(out / ses / "atlas-Toy") == ses


def _wave_without_sub06(dataset, root):
    root = _xcpd_waves(dataset, root, ["ses-01", "ses-02"])
    (root / "sub-06" / "ses-02" / "func" / f"sub-06_ses-02_{TAIL}").unlink()
    return root


def test_graph_report_lists_skipped_participants(dataset, xcpd, tmp_path):
    _, coords, _ = xcpd
    root = _wave_without_sub06(dataset, tmp_path / "xcpd")
    out = tmp_path / "graph"
    with pytest.warns(UserWarning, match="sub-06"):
        graph_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--session-id", "02",
                        "--coords", str(coords), "--no-static-brain"] + [a for a in FAST if a != "--no-report"])
    report = (out / "ses-02" / "atlas-Toy" / "graph_report.html").read_text(encoding="utf-8")
    assert "sub-06: no file found" in report


def test_nbs_leaves_out_participants_missing_in_a_session(dataset, xcpd, tmp_path):
    _, coords, groups = xcpd
    root = _wave_without_sub06(dataset, tmp_path / "xcpd")
    out = tmp_path / "nbs"
    with pytest.warns(UserWarning, match="sub-06"):
        nbs_cli.main([str(root), str(out), "--input-type", "xcpd", "--atlases", "Toy", "--session-id", "02",
                      "--groups", str(groups), "--group-column", "dx", "--contrast", "A", "B", "--thresh", "1.0",
                      "--perms", "20", "--seed", "0", "--n-jobs", "1", "--coords", str(coords), "--no-static-brain",
                      "--quiet"])
    out = out / "ses-02" / "atlas-Toy"
    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert params["groups"]["g2"] == ["04", "05"] and params["input"]["missing"] == ["sub-06"]
    report = (out / "nbs_report.html").read_text(encoding="utf-8")
    assert "session ses-02" in report and "sub-06" in report
