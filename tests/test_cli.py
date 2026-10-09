import json

import numpy as np
import pandas as pd
import pytest

from conngraph.cli import main as cli

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


FAST = ["--level", "node", "--graph-method", "density", "--graph-param", "density=0.2",
        "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"]
XCPD = ["--input-type", "xcpd", "--atlases", "Toy"]


def _params(folder):
    return json.loads((folder / "parameters.json").read_text(encoding="utf-8"))


def _both(input_dir, out, participant_args, group_args=("--no-report",), common=()):
    """Run the participant level, then the group level, and return the group level's exit code."""
    assert cli.main([str(input_dir), str(out), "participant", *common, *participant_args]) == 0
    return cli.main([str(input_dir), str(out), "group", *common, "--quiet", *group_args])


def test_participant_level_writes_one_set_of_files_per_participant(dataset, xcpd, tmp_path):
    root, _, _ = xcpd
    out = tmp_path / "out"
    cli.main([str(root), str(out), "participant", *XCPD, "--graph-method", "tmfg", "--network-graph-method", "full",
              "--metrics", "strength", "clust_coeff", "--n-jobs", "1", "--no-report", "--quiet"])
    folder = out / "sub-01" / "ses-01"
    stem = "sub-01_ses-01_atlas-Toy"
    node = pd.read_csv(folder / f"{stem}_level-node_metrics.tsv", sep="\t", index_col=0)
    assert list(node.columns) == ["strength.abs", "clust_coeff.costantini"] and len(node) == len(dataset.nodes_df)
    assert (folder / f"{stem}_level-networkhemi_metrics.tsv").exists()
    conn = pd.read_csv(folder / f"{stem}_level-networkhemi_connectivity.tsv", sep="\t", index_col=0)
    assert conn.shape[0] == conn.shape[1] and np.allclose(conn, conn.T)
    sidecar = json.loads((folder / f"{stem}_metrics.json").read_text(encoding="utf-8"))
    assert sidecar["subjects"] == ["01"] and sidecar["command"].startswith("conngraph ")
    assert sorted(p.name for p in out.glob("sub-*")) == [f"sub-{i:02d}" for i in range(1, 7)]
    description = json.loads((out / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["DatasetType"] == "derivative" and description["GeneratedBy"][0]["Name"] == "ConnGraph"
    assert not (out / "group").exists() and not list(out.rglob("*.html"))


def test_reports_only_rebuilds_the_reports_from_the_written_files(xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "out"
    assert cli.main([str(root), str(out), "participant", *XCPD, "--graph-method", "density", "--graph-param",
                     "density=0.2", "--network-graph-method", "full", "--metrics", "strength", "--n-jobs", "1",
                     "--no-report", "--quiet"]) == 0
    # plots carry their numbers encoded, so the change is looked for in a table of the report
    first = out / "sub-01" / "ses-01" / "sub-01_ses-01_atlas-Toy_level-networkhemi_metrics.tsv"
    table = pd.read_csv(first, sep="\t", index_col=0)
    table.iloc[0, 0] = 123.456
    table.to_csv(first, sep="\t")
    assert cli.main([str(root), str(out), "participant", *XCPD, "--reports-only", "--participant-label", "01",
                     "--quiet"]) == 0
    assert [p.name for p in out.glob("*.html")] == ["sub-01.html"]
    page = (out / "sub-01.html").read_text(encoding="utf-8")
    assert 'data-v="123.456"' in page
    assert 'href="sub-01/ses-01/sub-01_ses-01_atlas-Toy_level-networkhemi_metrics.tsv"' in page

    assert cli.main([str(root), str(out), "group", *XCPD, "--groups", str(groups), "--group-column", "dx",
                     "--contrast", "A", "B", "--correction", "fdr", "--n-perms", "10", "--nbs-thresh", "1.0",
                     "--n-jobs", "1", "--no-report", "--quiet"]) == 0
    group = out / "group" / "ses-01" / "atlas-Toy"
    saved = group / "node" / "strength.abs.tsv"
    table = pd.read_csv(saved, sep="\t", index_col=0)
    table.iloc[0, 0] = 654.321
    table.to_csv(saved, sep="\t")
    assert cli.main([str(root), str(out), "group", *XCPD, "--reports-only", "--quiet"]) == 0
    reports = [group / "graph_report.html", group / "compare" / "compare_report.html", group / "nbs" / "nbs_report.html"]
    assert all(p.exists() for p in reports)
    mean = table.iloc[:, 0].mean()
    assert f'data-v="{mean:.10g}"' in reports[0].read_text(encoding="utf-8")
    links = ('href="node/strength.abs.tsv"', 'href="metrics_node.tsv"', 'href="nbs_components.tsv"')
    assert all(link in p.read_text(encoding="utf-8") for link, p in zip(links, reports))
    description = json.loads((out / "dataset_description.json").read_text(encoding="utf-8"))
    assert "--reports-only" not in description["GeneratedBy"][0]["Description"]


def test_reports_only_refuses_analysis_options_and_missing_results(xcpd, tmp_path, capsys):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "participant", *XCPD, "--reports-only", "--metrics", "strength"])
    assert "--metrics has no effect with --reports-only" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no group results"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, "--reports-only", "--quiet"])


def test_group_level_collects_the_participants_into_tables_and_a_report(xcpd, surfaces, tmp_path):
    root, coords, _ = xcpd
    out = tmp_path / "out"
    _both(root, out, [*XCPD, "--graph-method", "tmfg", "--network-graph-method", "full", "--metrics", "strength",
                      "clust_coeff",
                      "--n-jobs", "1", "--quiet"], ["--coords", str(coords), "--surfaces", *surfaces], common=XCPD)
    group = out / "group" / "ses-01" / "atlas-Toy"
    table = pd.read_csv(group / "node" / "strength.abs.tsv", sep="\t", dtype={"ID": str})
    assert list(table["ID"]) == ["01", "02", "03", "04", "05", "06"]
    assert (group / "network_hemi" / "clust_coeff.costantini.tsv").exists()
    params = _params(group)
    assert params["subjects"] == ["01", "02", "03", "04", "05", "06"]
    assert params["input"]["source"] == "XCP-D" and params["input"]["atlas"] == "Toy"
    report = (group / "graph_report.html").read_text(encoding="utf-8")
    assert "Run command" in report and "derived from XCP-D (Toy atlas; fsLR space" in report
    assert "Group mean on the brain" in report and "scatter3d" not in report


def test_group_tables_match_the_python_api(dataset, tmp_path):
    from conngraph.graph_theory import compute_graph_metrics

    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    out = tmp_path / "out"
    _both(folder, out, ["--input-type", "matrix", *FAST], common=["--input-type", "matrix"])
    table = pd.read_csv(out / "group" / "node" / "strength.abs.tsv", sep="\t", index_col="ID")
    api = compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="node", metrics="strength",
                                graph_method="density", graph_params={"density": 0.2}, network_col="network",
                                n_jobs=1, verbose=False)
    assert np.allclose(table.loc[sorted(dataset.matrices)], api.node_df["strength.abs"].loc[sorted(dataset.matrices)])


def test_participants_run_one_at_a_time_match_a_single_run(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    opts = ["--input-type", "matrix", "--level", "node", "--graph-method", "density", "--graph-param", "density=0.3",
            "--metrics", "clust_coeff", "--n-random", "3", "--random-seed", "7", "--n-jobs", "1", "--no-report",
            "--quiet"]
    together, apart = tmp_path / "together", tmp_path / "apart"
    cli.main([str(folder), str(together), "participant", *opts])
    for sid in sorted(dataset.matrices):
        cli.main([str(folder), str(apart), "participant", *opts, "--participant-label", sid])
    for sid in sorted(dataset.matrices):
        name = f"{sid}/{sid}_level-node_metrics.tsv"
        a = pd.read_csv(together / name, sep="\t", index_col=0)
        b = pd.read_csv(apart / name, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(a, b)


def test_nodes_are_dropped_by_looking_at_every_participant(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    ids = sorted(dataset.matrices)
    for sid in ids:
        mat = dataset.matrices[sid].copy()
        if sid == ids[-1]:
            bad = mat.columns[0]
            mat.loc[bad, :] = np.nan
            mat.loc[:, bad] = np.nan
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="Dropping"):
        cli.main([str(folder), str(out), "participant", "--input-type", "matrix", *FAST, "--participant-label", ids[0]])
    node = pd.read_csv(out / ids[0] / f"{ids[0]}_level-node_metrics.tsv", sep="\t", index_col=0)
    assert bad not in node.index and len(node) == len(dataset.nodes_df) - 1
    assert sorted(p.name for p in out.glob("sub-*")) == [ids[0]]


def test_group_level_refuses_participants_run_with_different_options(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    ids = sorted(dataset.matrices)
    out = tmp_path / "out"
    cli.main([str(folder), str(out), "participant", "--input-type", "matrix", *FAST, "--participant-label", ids[0]])
    other = [a if a != "density=0.2" else "density=0.3" for a in FAST]
    cli.main([str(folder), str(out), "participant", "--input-type", "matrix", *other, "--participant-label", ids[1]])
    with pytest.raises(SystemExit, match="different options"):
        cli.main([str(folder), str(out), "group", "--input-type", "matrix", "--no-report", "--quiet"])


def test_group_level_needs_participant_results_or_a_test(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit, match="participant level"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, "--no-report", "--quiet"])


@pytest.mark.parametrize("level, option", [
    ("group", ["--graph-method", "full"]),
    ("group", ["--metrics", "strength"]),
    ("group", ["--n-random", "5"]),
    ("participant", ["--group-column", "dx"]),
    ("participant", ["--contrast", "A", "B"]),
    ("participant", ["--nbs-thresh", "3"]),
])
def test_options_of_the_other_level_are_refused(xcpd, tmp_path, capsys, level, option):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), level, *XCPD, "--quiet", *option])
    assert f"{option[0]} belongs to the" in capsys.readouterr().err


def test_analysis_level_is_required(xcpd, tmp_path, capsys):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), *XCPD])
    assert "analysis_level" in capsys.readouterr().err


def test_group_matrix_folder_records_graph_options_and_skips_the_report(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    out = tmp_path / "out"
    _both(folder, out, ["--input-type", "matrix", "--level", "node", "--graph-method", "density", "--graph-param",
                        "density=0.2,0.3", "--metrics", "strength", "--n-jobs", "1", "--no-report", "--quiet"],
          common=["--input-type", "matrix"])
    params = _params(out / "group")
    assert params["levels"]["node"]["graph_params"] == {"density": [0.2, 0.3]}
    assert params["input"]["source"] == "matrix files"
    assert not (out / "group" / "graph_report.html").exists()


def test_participant_level_rejects_unknown_graph_param_syntax(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "participant", *XCPD, "--graph-param", "density", "--quiet"])


def test_participant_level_needs_a_graph_method(xcpd, tmp_path, capsys):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "participant", *XCPD, "--quiet"])
    assert "--graph-method is required" in capsys.readouterr().err


NBS = ["--nbs-thresh", "1.0", "--n-perms", "20", "--random-seed", "0"]


def _nbs_args(groups):
    return ["--groups", str(groups), "--group-column", "dx", "--contrast", "A", "B", *NBS]


def test_group_level_runs_nbs_from_a_groups_table(xcpd, surfaces, tmp_path):
    root, coords, groups = xcpd
    out = tmp_path / "out"
    cli.main([str(root), str(out), "group", *XCPD, *_nbs_args(groups), "--coords", str(coords), "--surfaces", *surfaces,
              "--quiet"])
    nbs = out / "group" / "ses-01" / "atlas-Toy" / "nbs"
    params = _params(nbs)
    assert params["groups"] == {"g1": ["01", "02", "03"], "g2": ["04", "05", "06"]}
    assert params["input"]["fisher_z"] is True and params["input"]["contrast"] == ["A", "B"]
    components = pd.read_csv(nbs / "nbs_components.tsv", sep="\t")
    assert list(components.columns) == ["component", "edges", "nodes", "p"]
    assert len(pd.read_csv(nbs / "nbs_null.tsv", sep="\t")) == 20
    report = (nbs / "nbs_report.html").read_text(encoding="utf-8")
    assert "Fisher z-transformed before testing" in report and "Run command" in report


def test_nbs_needs_the_groups_and_contrast(xcpd, tmp_path):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *NBS, "--quiet"])


@pytest.mark.parametrize("flag", ["--n-jobs", "--nprocs"])
@pytest.mark.parametrize("level", ["participant", "group"])
def test_both_levels_take_a_worker_count(flag, level):
    args = cli.build_parser().parse_args(["in", "out", level, *XCPD, flag, "3"])
    assert args.n_jobs == 3


def test_nbs_gets_the_worker_count(xcpd, tmp_path, monkeypatch):
    from conngraph.graph_theory import nbs as nbs_module

    seen = {}
    original = nbs_module.run_nbs

    def recording(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(nbs_module, "run_nbs", recording)
    root, _, groups = xcpd
    cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *_nbs_args(groups), "--no-report", "--quiet",
              "--nprocs", "2"])
    assert seen["n_jobs"] == 2


def test_nbs_reports_groups_missing_from_the_data(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": ["sub-01", "sub-99"], "dx": ["A", "B"]}).to_csv(table, sep="\t", index=False)
    with pytest.raises(SystemExit, match="sub-99"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, "--groups", str(table), "--group-column", "dx",
                  "--contrast", "A", "B", *NBS, "--quiet"])


def _layout(dataset, folder):
    """A matrix or time series input folder holding only the fixed-name node table."""
    folder.mkdir()
    dataset.nodes_df[["label", "network", "hemisphere", "x", "y", "z"]].to_csv(folder / "nodes.tsv", sep="\t",
                                                                               index=False)
    return folder


def test_input_type_is_required(xcpd, tmp_path, capsys):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "participant", "--atlases", "Toy"] + FAST)
    assert "--input-type" in capsys.readouterr().err


@pytest.mark.parametrize("option", [["--input-type", "xcpd-flat"], ["--input-format", "xcpd"], ["--atlas-file", "a.tsv"],
                                    ["--nodes", "n.tsv"], ["--pattern", "*.csv"], ["--label-col", "x"],
                                    ["--network-col", "x"], ["--thresh", "1"], ["--perms", "10"], ["--seed", "1"],
                                    ["--nbs-perms", "10"], ["--no-static-brain"]])
def test_removed_options_are_rejected(xcpd, tmp_path, capsys, option):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "participant", *XCPD] + FAST + option)
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
    out = tmp_path / "out"
    _both(folder, out, ["--input-type", "matrix", *FAST], common=["--input-type", "matrix"])
    params = _params(out / "group")
    assert params["subjects"] == sids
    assert params["input"]["source"] == "matrix files"


def test_matrix_folder_refuses_two_files_for_one_participant(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    np.save(folder / f"{sid}_matrix.npy", mat.to_numpy())
    with pytest.raises(SystemExit, match=sid):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "matrix"] + FAST)


def test_matrix_folder_runs_without_coordinates(dataset, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    dataset.nodes_df[["label", "network"]].to_csv(folder / "nodes.tsv", sep="\t", index=False)
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    out = tmp_path / "out"
    _both(folder, out, ["--input-type", "matrix", *FAST], group_args=(), common=["--input-type", "matrix"])
    assert "no x, y, z coordinates" in (out / "group" / "graph_report.html").read_text(encoding="utf-8")


def test_matrix_folder_needs_nodes_tsv(dataset, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    sid, mat = next(iter(dataset.matrices.items()))
    mat.to_csv(folder / f"{sid}_matrix.tsv", sep="\t")
    with pytest.raises(SystemExit, match="nodes.tsv"):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "matrix"] + FAST)


def _toy_series(dataset, seed):
    rng = np.random.default_rng(seed)
    n = len(dataset.nodes_df)
    return pd.DataFrame(rng.standard_normal((100, n)) @ rng.standard_normal((n, n)), columns=dataset.nodes_df["label"])


def test_time_series_folder(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "ts")
    for i in range(3):
        np.savetxt(folder / f"sub-{i:02d}_timeseries.txt", _toy_series(dataset, i).to_numpy())
    out = tmp_path / "out"
    common = ["--input-type", "timeseries", "--connectivity", "partial-correlation", "--shrinkage"]
    _both(folder, out, FAST, common=common)
    params = _params(out / "group")
    assert params["input"]["source"] == "time series"
    assert (params["input"]["connectivity"], params["input"]["shrinkage"]) == ("partial correlation", True)
    assert params["subjects"] == ["sub-00", "sub-01", "sub-02"]


def test_xcpd_connectivity_from_time_series(dataset, tmp_path):
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
    out = tmp_path / "out"
    _both(tmp_path / "xcpd", out, FAST, common=[*XCPD, "--connectivity", "correlation"])
    params = _params(out / "group" / "ses-01" / "atlas-Toy")
    assert (params["input"]["source"], params["input"]["connectivity"]) == ("XCP-D", "correlation")


def test_unlabelled_fisher_z_matrices(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "z")
    for sid, mat in dataset.matrices.items():
        z = np.arctanh(np.clip(mat.to_numpy(float), -0.999, 0.999))
        np.save(folder / f"{sid}_matrix.npy", z)
    out = tmp_path / "out"
    _both(folder, out, FAST, common=["--input-type", "matrix", "--values", "z"])
    params = _params(out / "group")
    assert params["input"]["values"] == "z"
    assert params["subjects"] == sorted(dataset.matrices)


def test_connectivity_options_need_time_series(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    with pytest.raises(SystemExit):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "matrix", "--connectivity",
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


def test_fnirs_pipe_input_handles_each_chromophore_on_its_own(dataset, tmp_path):
    root = _fnirs_pipe_tree(dataset, tmp_path / "fnirs")
    out = tmp_path / "out"
    fast = ["--graph-method", "full", "--metrics", "strength", "--n-jobs", "1", "--quiet"]
    _both(root, out, fast, common=["--input-type", "fnirs-pipe"])
    sid = sorted(dataset.matrices)[0]
    for chromo in ("hbo", "hbr"):
        assert (out / sid / "ses-01" / f"{sid}_ses-01_chromo-{chromo}_level-node_metrics.tsv").exists()
        params = _params(out / "group" / "ses-01" / f"chromo-{chromo}")
        assert (params["input"]["source"], params["input"]["chromophore"]) == ("fnirs-pipe", chromo)
        assert list(params["levels"]) == ["node"]
    only = tmp_path / "hbr_only"
    cli.main([str(root), str(only), "participant", "--input-type", "fnirs-pipe", "--chromophore", "hbr",
              "--session-id", "01", *fast])
    assert sorted(p.name.split("_")[2] for p in (only / sid / "ses-01").glob("*.tsv")) == ["chromo-hbr"]


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
    return _params(folder)["input"]["session"]


def test_each_session_gets_its_own_files_and_group_folder(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02"])
    out = tmp_path / "out"
    assert _both(root, out, FAST, common=XCPD) == 0
    assert sorted(p.name for p in (out / "sub-01").iterdir()) == ["ses-01", "ses-02"]
    assert sorted(p.name for p in (out / "group").iterdir()) == ["ses-01", "ses-02"]
    assert [_session(out / "group" / s / "atlas-Toy") for s in ("ses-01", "ses-02")] == ["ses-01", "ses-02"]


def test_session_id_selects_sessions_with_or_without_prefix(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02", "ses-03"])
    out = tmp_path / "out"
    cli.main([str(root), str(out), "participant", *XCPD, "--session-id", "03", "ses-01"] + FAST)
    assert sorted(p.name for p in (out / "sub-01").iterdir()) == ["ses-01", "ses-03"]


def test_input_without_sessions_has_no_session_folder(dataset, tmp_path):
    root = _xcpd_waves(dataset, tmp_path / "xcpd", [None])
    out = tmp_path / "out"
    _both(root, out, FAST, common=XCPD)
    assert (out / "sub-01" / "sub-01_atlas-Toy_level-node_metrics.tsv").exists()
    assert sorted(p.name for p in (out / "group").iterdir()) == ["atlas-Toy"]


def test_matrix_folder_with_sessions(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        for ses in ("ses-01", "ses-02"):
            mat.to_csv(folder / f"{sid}_{ses}_matrix.tsv", sep="\t")
    out = tmp_path / "out"
    _both(folder, out, FAST, common=["--input-type", "matrix"])
    for ses in ("ses-01", "ses-02"):
        params = _params(out / "group" / ses)
        assert params["subjects"] == sorted(dataset.matrices) and params["input"]["session"] == ses


def test_matrix_folder_mixing_files_with_and_without_session_is_refused(dataset, tmp_path):
    folder = _layout(dataset, tmp_path / "in")
    (a, ma), (b, mb) = list(dataset.matrices.items())[:2]
    ma.to_csv(folder / f"{a}_ses-01_matrix.tsv", sep="\t")
    mb.to_csv(folder / f"{b}_matrix.tsv", sep="\t")
    with pytest.raises(SystemExit, match="session"):
        cli.main([str(folder), str(tmp_path / "out"), "participant", "--input-type", "matrix"] + FAST)


def test_a_failed_session_leaves_an_error_file_and_the_others_still_run(dataset, tmp_path, capsys):
    folder = _layout(dataset, tmp_path / "in")
    for sid, mat in dataset.matrices.items():
        mat.to_csv(folder / f"{sid}_ses-01_matrix.tsv", sep="\t")
        np.save(folder / f"{sid}_ses-02_matrix.npy", np.ones((2, 3)))
    out = tmp_path / "out"
    assert cli.main([str(folder), str(out), "participant", "--input-type", "matrix"] + FAST) == 1
    sid = sorted(dataset.matrices)[0]
    assert (out / sid / "ses-01" / f"{sid}_ses-01_metrics.json").exists()
    assert "not square" in (out / "logs" / "ses-02" / "error.txt").read_text(encoding="utf-8")
    assert "ses-02" in capsys.readouterr().err


def test_nbs_runs_each_session(dataset, xcpd, tmp_path):
    _, _, groups = xcpd
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02"])
    out = tmp_path / "out"
    cli.main([str(root), str(out), "group", *XCPD, *_nbs_args(groups), "--n-jobs", "1", "--no-report", "--quiet"])
    for ses in ("ses-01", "ses-02"):
        nbs = out / "group" / ses / "atlas-Toy" / "nbs"
        assert (nbs / "nbs_components.tsv").exists() and _session(nbs) == ses


def _wave_without_sub06(dataset, root):
    root = _xcpd_waves(dataset, root, ["ses-01", "ses-02"])
    (root / "sub-06" / "ses-02" / "func" / f"sub-06_ses-02_{TAIL}").unlink()
    return root


def test_group_report_lists_skipped_participants(dataset, xcpd, surfaces, tmp_path):
    _, coords, _ = xcpd
    root = _wave_without_sub06(dataset, tmp_path / "xcpd")
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="sub-06"):
        _both(root, out, FAST, ["--coords", str(coords), "--surfaces", *surfaces], common=[*XCPD, "--session-id", "02"])
    report = (out / "group" / "ses-02" / "atlas-Toy" / "graph_report.html").read_text(encoding="utf-8")
    assert "sub-06: no file found" in report


def test_nbs_leaves_out_participants_missing_in_a_session(dataset, xcpd, surfaces, tmp_path):
    _, coords, groups = xcpd
    root = _wave_without_sub06(dataset, tmp_path / "xcpd")
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="sub-06"):
        cli.main([str(root), str(out), "group", *XCPD, "--session-id", "02", *_nbs_args(groups), "--n-jobs", "1",
                  "--coords", str(coords), "--surfaces", *surfaces, "--quiet"])
    nbs = out / "group" / "ses-02" / "atlas-Toy" / "nbs"
    params = _params(nbs)
    assert params["groups"]["g2"] == ["04", "05"] and params["input"]["missing"] == ["sub-06"]
    report = (nbs / "nbs_report.html").read_text(encoding="utf-8")
    assert "session ses-02" in report and "sub-06" in report


def test_whole_graph_metrics_are_written_per_participant_and_collected(xcpd, tmp_path):
    root, _, _ = xcpd
    out = tmp_path / "out"
    _both(root, out, ["--level", "node", "--graph-method", "density", "--graph-param", "density=0.3", "--metrics",
                      "eff_global", "participation", "--partition", "networks", "--signed-fallback", "positive",
                      "--sign", "abs", "--n-jobs", "1", "--quiet"], common=XCPD)
    one = pd.read_csv(out / "sub-01" / "ses-01" / "sub-01_ses-01_atlas-Toy_level-node_global.tsv", sep="\t")
    assert list(one.columns) == ["eff_global.wei"] and len(one) == 1
    node = pd.read_csv(out / "sub-01" / "ses-01" / "sub-01_ses-01_atlas-Toy_level-node_metrics.tsv", sep="\t")
    assert list(node.columns) == ["node", "participation.pos.networks"]
    group = out / "group" / "ses-01" / "atlas-Toy"
    table = pd.read_csv(group / "global" / "node.tsv", sep="\t", dtype={"ID": str})
    assert list(table["ID"]) == ["01", "02", "03", "04", "05", "06"]
    assert table.loc[0, "eff_global.wei"] == pytest.approx(one.loc[0, "eff_global.wei"])
    options = _params(group)["options"]
    assert options["signed_fallback"] == "positive" and options["partitions"] == ["networks"]


def test_fisher_z_is_applied_before_nbs():
    from conngraph.cli._shared import fisher_z

    m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])
    z = fisher_z({"s": m})["s"]
    assert z.iloc[0, 1] == pytest.approx(np.arctanh(0.5)) and z.iloc[0, 0] == 0


NETWORK = ["--graph-method", "density", "--graph-param", "density=0.3", "--metrics", "strength",
           "--network-graph-method", "full", "--n-jobs", "1", "--no-report", "--quiet"]


def _compare_args(groups, *extra):
    return ["--groups", str(groups), "--group-column", "dx", "--contrast", "A", "B", "--n-perms", "20",
            "--random-seed", "0", "--correction", "fdr", "--no-report", *extra]


def test_a_groups_table_compares_metrics_and_blocks_by_default(dataset, xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "out"
    assert _both(root, out, NETWORK, _compare_args(groups), common=XCPD) == 0
    compare = out / "group" / "ses-01" / "atlas-Toy" / "compare"
    assert sorted(p.name for p in compare.glob("*.tsv")) == ["blocks_networkhemi.tsv", "metrics_networkhemi.tsv",
                                                             "metrics_node.tsv"]
    node = pd.read_csv(compare / "metrics_node.tsv", sep="\t")
    assert list(node.columns) == ["metric", "node", "t", "p", "p_fdr", "p_fwe", "mean_group1", "mean_group2",
                                  "n_group1", "n_group2", "significant"]
    assert len(node) == len(dataset.nodes_df) and set(node["metric"]) == {"strength.abs"}
    params = _params(compare)
    assert params["groups"] == {"g1": ["01", "02", "03"], "g2": ["04", "05", "06"]}
    assert params["options"]["n_perms"] == 20 and params["options"]["seed"] == 0
    assert params["graph"]["levels"]["node"]["graph_method"] == "density" and params["input"]["contrast"] == ["A", "B"]


def test_edges_are_compared_when_named_and_need_no_participant_results(dataset, xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "out"
    cli.main([str(root), str(out), "group", *XCPD, *_compare_args(groups, "--compare", "edges"), "--quiet"])
    compare = out / "group" / "ses-01" / "atlas-Toy" / "compare"
    n = len(dataset.nodes_df)
    assert [p.name for p in compare.glob("*.tsv")] == ["edges.tsv"]
    assert len(pd.read_csv(compare / "edges.tsv", sep="\t")) == n * (n - 1) // 2


def test_comparing_metrics_needs_participant_results(xcpd, tmp_path):
    root, _, groups = xcpd
    with pytest.raises(SystemExit, match="participant-level results"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *_compare_args(groups, "--compare", "metrics"),
                  "--quiet"])


def test_an_empty_compare_skips_the_comparisons(xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "out"
    assert _both(root, out, NETWORK, _compare_args(groups, "--compare"), common=XCPD) == 0
    assert not (out / "group" / "ses-01" / "atlas-Toy" / "compare").exists()


def test_blocks_are_left_out_by_default_and_refused_when_named_without_the_network_level(xcpd, tmp_path):
    root, _, groups = xcpd
    out = tmp_path / "out"
    assert _both(root, out, FAST, _compare_args(groups), common=XCPD) == 0
    compare = out / "group" / "ses-01" / "atlas-Toy" / "compare"
    assert [p.name for p in compare.glob("*.tsv")] == ["metrics_node.tsv"]
    with pytest.raises(SystemExit, match="network level"):
        cli.main([str(root), str(out), "group", *XCPD, *_compare_args(groups, "--compare", "blocks"), "--quiet"])


def test_covariates_come_from_the_groups_table(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": [f"sub-{i:02d}" for i in range(1, 7)], "dx": ["A", "A", "A", "B", "B", "B"],
                  "age": [20, 25, 31, 22, 40, "n/a"]}).to_csv(table, sep="\t", index=False)
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="06"):
        _both(root, out, FAST, _compare_args(table, "--covariates", "age"), common=XCPD)
    params = _params(out / "group" / "ses-01" / "atlas-Toy" / "compare")
    assert params["covariate_columns"] == ["age"] and params["groups"]["g2"] == ["04", "05"]
    with pytest.raises(SystemExit, match="no column 'sex'"):
        cli.main([str(root), str(out), "group", *XCPD, *_compare_args(table, "--covariates", "sex"), "--quiet"])


@pytest.mark.parametrize("option", [["--compare", "edges"], ["--covariates", "age"], ["--groups", "g.tsv"],
                                    ["--correction", "fdr"], ["--alpha", "0.01"]])
def test_comparison_options_need_a_complete_groups_table(xcpd, tmp_path, capsys, option):
    root, _, _ = xcpd
    with pytest.raises(SystemExit):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *option, "--quiet"])
    err = capsys.readouterr().err
    assert "needs --groups" in err or "needs --group-column" in err


def test_nbs_warns_that_it_ignores_covariates(xcpd, tmp_path):
    root, _, _ = xcpd
    table = tmp_path / "participants.tsv"
    pd.DataFrame({"participant_id": [f"sub-{i:02d}" for i in range(1, 7)], "dx": ["A", "A", "A", "B", "B", "B"],
                  "age": [20, 25, 31, 22, 40, 33]}).to_csv(table, sep="\t", index=False)
    with pytest.warns(UserWarning, match="NBS does not adjust"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *_compare_args(table, "--covariates", "age"),
                  "--nbs-thresh", "1.0", "--n-jobs", "1", "--quiet"])
    assert (tmp_path / "out" / "group" / "ses-01" / "atlas-Toy" / "nbs" / "nbs_components.tsv").exists()


def test_comparisons_need_a_chosen_correction(xcpd, tmp_path):
    root, _, groups = xcpd
    args = [a for a in _compare_args(groups, "--compare", "edges") if a not in ("--correction", "fdr")]
    with pytest.raises(SystemExit, match="--correction"):
        cli.main([str(root), str(tmp_path / "out"), "group", *XCPD, *args, "--quiet"])
    out = tmp_path / "out2"
    cli.main([str(root), str(out), "group", *XCPD, *_compare_args(groups, "--compare", "edges"), "--alpha", "0.2",
              "--quiet"])
    compare = out / "group" / "ses-01" / "atlas-Toy" / "compare"
    edges = pd.read_csv(compare / "edges.tsv", sep="\t")
    assert edges["significant"].tolist() == (edges["p_fdr"] < 0.2).tolist()
    assert _params(compare)["options"]["alpha"] == 0.2


def test_group_comparisons_get_a_report(xcpd, surfaces, tmp_path):
    root, coords, groups = xcpd
    out = tmp_path / "out"
    args = [a for a in _compare_args(groups) if a != "--no-report"]
    _both(root, out, NETWORK, [*args, "--coords", str(coords), "--surfaces", *surfaces], common=XCPD)
    report = out / "group" / "ses-01" / "atlas-Toy" / "compare" / "compare_report.html"
    assert "Group comparison report" in report.read_text(encoding="utf-8")


def test_participant_level_writes_one_report_per_participant(dataset, xcpd, surfaces, tmp_path):
    import re

    _, coords, _ = xcpd
    root = _xcpd_waves(dataset, tmp_path / "xcpd", ["ses-01", "ses-02"])
    out = tmp_path / "out"
    cli.main([str(root), str(out), "participant", *XCPD, "--graph-method", "density", "--graph-param", "density=0.3",
              "--network-graph-method", "full", "--metrics", "strength", "eff_global", "--coords", str(coords),
              "--surfaces", *surfaces, "--interactive-brain", "--n-jobs", "1", "--quiet"])
    assert sorted(p.name for p in out.glob("sub-*.html")) == [f"sub-{i:02d}.html" for i in range(1, 7)]
    text = (out / "sub-01.html").read_text(encoding="utf-8")
    assert re.findall(r'<h2 id="([^"]+)"', text)[:3] == ["Summary", "ses-01_atlas-Toy", "ses-02_atlas-Toy"]
    assert '<h2 id="Quantities">' in text and "Edges kept" in text and ">ses-02 atlas-Toy</th>" in text
    for words in ("Connectivity matrix", "Node metrics", "Network metrics", "Whole graph", "Graph",
                  "Open the rotatable 3-D view", "Methods"):
        assert words in text, words
    leftovers = [p.name for p in out.rglob("*.json") if not p.name.endswith(("_metrics.json", "description.json"))]
    assert not leftovers
    pngs = sorted(p.name for p in (out / "sub-01" / "figures").glob("*.png"))
    assert pngs == [f"sub-01_{ses}_atlas-Toy_brain_strength.abs.png" for ses in ("ses-01", "ses-02")]
    assert all(f'src="sub-01/figures/{name}"' in text for name in pngs) and "scatter3d" not in text
    assert sorted(p.name for p in (out / "sub-01" / "figures").glob("*.html")) == [p[:-4] + ".html" for p in pngs]


def test_participant_reports_built_in_parallel_match_those_built_one_by_one(xcpd, surfaces, tmp_path):
    import re

    root, coords, _ = xcpd
    pages = {}
    for n_jobs in ("1", "3"):
        out = tmp_path / n_jobs
        assert cli.main([str(root), str(out), "participant", *XCPD, "--graph-method", "density", "--graph-param",
                         "density=0.3", "--level", "node", "--metrics", "strength", "--coords", str(coords),
                         "--surfaces", *surfaces, "--n-jobs", n_jobs, "--quiet"]) == 0
        # plotly ids, the time and the command line differ between any two runs
        pages[n_jobs] = {p.name: re.sub(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d"
                                        r"|<pre>conngraph .*?</pre>", "", p.read_text(encoding="utf-8"))
                         for p in out.glob("sub-*.html")}
        assert sorted(p.name for p in out.glob("sub-*/figures/*.png")) == [
            f"sub-{i:02d}_ses-01_atlas-Toy_brain_strength.abs.png" for i in range(1, 7)]
    assert len(pages["1"]) == 6 and pages["1"] == pages["3"]


def test_report_workers_fit_in_the_memory_limit(capsys):
    import argparse

    from conngraph.cli.participant import REPORT_MB, REPORT_MB_PER_REGION, _report_workers

    parser = argparse.ArgumentParser(prog="conngraph")
    need = REPORT_MB + REPORT_MB_PER_REGION * 100
    args = argparse.Namespace(n_jobs=8, mem=int(2.5 * need), quiet=False)
    assert _report_workers(args, parser, 30, 100) == 2
    assert "Building 2 report(s) at a time instead of 8" in capsys.readouterr().out
    assert _report_workers(argparse.Namespace(n_jobs=8, mem=100 * int(need), quiet=False), parser, 5, 100) == 5
    assert _report_workers(argparse.Namespace(n_jobs=8, mem=1, quiet=True), parser, 30, 100) == 1
    assert capsys.readouterr().out == ""


def test_a_failed_participant_report_is_logged_and_the_others_are_written(xcpd, tmp_path, monkeypatch, capsys):
    from conngraph.cli import participant

    def fail_for_sub02(args, prog, sid, *rest):
        if sid.removeprefix("sub-").startswith("02"):
            raise RuntimeError("cannot draw sub-02")
        save(args, prog, sid, *rest)

    save = participant._save_section
    monkeypatch.setattr(participant, "_save_section", fail_for_sub02)
    root, _, _ = xcpd
    out = tmp_path / "out"
    code = cli.main([str(root), str(out), "participant", *XCPD, "--graph-method", "density", "--graph-param",
                     "density=0.3", "--level", "node", "--metrics", "strength", "--n-jobs", "1", "--quiet"])
    assert code == 1
    assert sorted(p.name for p in out.glob("sub-*.html")) == [f"sub-{i:02d}.html" for i in (1, 3, 4, 5, 6)]
    log = next(out.glob("logs/**/report-sub-02.err"))
    assert "RuntimeError: cannot draw sub-02" in log.read_text(encoding="utf-8")
    assert "1 participant report(s) failed: sub-02" in capsys.readouterr().err


def test_participant_reports_can_be_skipped(xcpd, tmp_path):
    root, _, _ = xcpd
    out = tmp_path / "out"
    cli.main([str(root), str(out), "participant", *XCPD, *FAST])
    assert not list(out.glob("*.html"))
