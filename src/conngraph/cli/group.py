"""
Group level: participant results collected into tables and a report, and comparisons between two groups.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

from conngraph.cli import _shared
from conngraph.cli.participant import CONNECTIVITY_FILES, GLOBAL_FILES, LEVEL_FILES
from conngraph.loaders import INPUT_ATTR

SIDECAR = "_metrics.json"
NODES_FILE = "nodes.tsv"


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, session: str | None, variant: str | None,
        command: str) -> None:
    out = _shared.run_folder(args.output_dir, "group", session, variant)
    sidecars = _participant_results(args, session, variant)
    where = " ".join(x for x in (session, variant) if x)
    compare = _comparisons(args, parser, sidecars)
    if compare and args.correction is None:
        raise SystemExit(f"{parser.prog}: choose how to correct the group comparisons with --correction fdr, fwe or "
                         "none (or skip them with an empty --compare)")
    if not sidecars and args.nbs_thresh is None and not compare:
        raise SystemExit(f"{parser.prog}: no participant-level results in {args.output_dir}"
                         + (f" for {where}" if where else "") + "; run the participant level first")
    if args.covariates and args.nbs_thresh is not None:
        warnings.warn("NBS does not adjust for --covariates; only the other comparisons do", stacklevel=2)
    matrices, atlas = _shared.load_input(args, parser, variant, session)
    report_nodes = _shared.report_nodes(args, atlas, variant)
    nodes = atlas if report_nodes is None else report_nodes
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(out / NODES_FILE, sep="\t", index=False)
    result = _graph_metrics(args, parser, sidecars, matrices, atlas, nodes, out, command) if sidecars else None
    if not args.groups:
        return
    matrices = {sid: matrices[sid] for sid in _shared.chosen(matrices, args)}
    groups, rows, missing = _contrast_groups(args, parser, matrices, atlas)
    if compare:
        _compare(args, parser, compare, result, matrices, groups, rows, missing, atlas, out / "compare", command)
    if args.nbs_thresh is not None:
        _nbs(args, parser, matrices, groups, missing, atlas, out / "nbs", command)


def _comparisons(args: argparse.Namespace, parser: argparse.ArgumentParser, sidecars: list) -> list[str]:
    """Those named by --compare, else metrics and blocks whenever there are groups and participant results."""
    if not args.groups:
        return []
    if args.compare is None:
        return ["metrics", "blocks"] if sidecars else []
    if {"metrics", "blocks"} & set(args.compare) and not sidecars:
        raise SystemExit(f"{parser.prog}: comparing metrics or blocks needs participant-level results in "
                         f"{args.output_dir}; run the participant level first")
    return list(dict.fromkeys(args.compare))


def _participant_results(args: argparse.Namespace, session: str | None, variant: str | None) -> list[pathlib.Path]:
    pattern = "sub-*" + (f"_{session}*" if session else "") + (f"_{variant}" if variant else "") + SIDECAR
    paths = sorted(pathlib.Path(args.output_dir).glob(f"sub-*/{session + '/' if session else ''}{pattern}"))
    if not args.participant_label:
        return paths
    labels = {s.removeprefix("sub-") for s in args.participant_label}
    return [p for p in paths if p.name.split("_")[0].removeprefix("sub-") in labels]


def _graph_metrics(args, parser, sidecars, matrices, atlas, nodes, out, command) -> None:
    from conngraph.graph_theory.runner import _mean_matrix, _save

    records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sidecars]
    first_path, first = records[0]
    odd = [path.name for path, params in records[1:] if _comparable(params) != _comparable(first)]
    if odd:
        raise SystemExit(f"{parser.prog}: these participants were run with different options from {first_path.name}: "
                         f"{', '.join(odd)}. Rerun the participant level with one set of options.")
    ids = [params["subjects"][0] for _, params in records]
    absent = [sid for sid in ids if sid not in matrices]
    if absent:
        raise SystemExit(f"{parser.prog}: the input no longer has a matrix for {', '.join(absent)}; rerun the "
                         "participant level")
    result = _read_participants(records, {str(c): c for c in matrices[ids[0]].columns}, parser.prog)
    seeds = {params["options"]["random_seed"] for _, params in records}
    options = dict(first["options"], random_seed=seeds.pop() if len(seeds) == 1 else "per participant")
    result.params = dict(first, options=options, subjects=ids, failed=result.failed,
                         warnings=[w for _, params in records for w in params["warnings"]],
                         input=atlas.attrs.get(INPUT_ATTR, {}), command=f"{first['command']}\n{command}")
    if "node" in first["levels"]:
        result.mean_matrix = _mean_matrix({sid: matrices[sid] for sid in ids}, first["options"]["apply_fisher_z"])

    result.nodes = nodes
    _save(result, str(out), verbose=False)
    if not args.no_report:
        _graph_report(args, out)
    if not args.quiet:
        print(f"[{parser.prog}] Collected {len(ids)} participant(s) in {out}")
    return result


def _read_participants(records: list[tuple[pathlib.Path, dict]], labels: dict, prog: str):
    """Participants' results stacked from their files, a row each; labels maps the files' node names to the input's."""
    from conngraph.graph_theory.runner import GraphMetricsResult, _net_corr_to_wide

    def read(attr: str, sidecar: pathlib.Path, suffix: str, index: str | None = None) -> pd.DataFrame | None:
        frame = _read(sidecar, suffix, index)
        if frame is not None:
            result.files.setdefault(attr, []).append(str(_file(sidecar, suffix).resolve()))
        return frame

    ids = [params["subjects"][0] for _, params in records]
    result = GraphMetricsResult()
    for attr, name, row in LEVEL_FILES:
        frames = [read(attr, path, f"_level-{name}_metrics.tsv", row) for path, _ in records]
        if frames[0] is None:
            continue
        if name == "node" and set(frames[0].index) != set(labels):
            raise SystemExit(f"{prog}: the input's nodes differ from those of the participant results; "
                             "rerun the participant level")
        if any(f is None or not f.index.equals(frames[0].index) or not f.columns.equals(frames[0].columns)
               for f in frames):
            raise SystemExit(f"{prog}: participants differ in their {name} rows or metrics; rerun the "
                             "participant level")
        rows = [labels[r] for r in frames[0].index] if name == "node" else list(frames[0].index)
        columns = pd.MultiIndex.from_product([frames[0].columns, rows], names=["metric", "roi" if row == "node" else row])
        setattr(result, attr, pd.DataFrame(np.vstack([f.to_numpy(dtype=float).T.ravel() for f in frames]),
                                           index=ids, columns=columns))
    for attr, name in CONNECTIVITY_FILES:
        squares = [read(attr, path, f"_level-{name}_connectivity.tsv", "network") for path, _ in records]
        if squares[0] is not None:
            setattr(result, attr, _net_corr_to_wide(dict(zip(ids, squares))))
    tables = {level: [read("global_df", path, f"_level-{name}_global.tsv") for path, _ in records]
              for level, name in GLOBAL_FILES}
    tables = {level: frames for level, frames in tables.items() if frames[0] is not None}
    if tables:
        result.global_df = pd.concat({level: pd.concat(frames, ignore_index=True).set_axis(ids)
                                      for level, frames in tables.items()}, axis=1, names=["level", "metric"])
    curves = [read("curves", path, "_curves.tsv") for path, _ in records]
    if curves[0] is not None:
        result.curves = pd.concat([c.assign(ID=sid) for sid, c in zip(ids, curves)], ignore_index=True)[
            ["level", "ID", "metric", "node", "threshold", "value"]]

    result.failed = {}
    for _, params in records:
        for lvl, subs in params["failed"].items():
            result.failed.setdefault(lvl, {}).update(subs)
    return result


def _comparable(params: dict) -> tuple:
    options = {k: v for k, v in params["options"].items() if k != "random_seed"}
    return json.dumps(options, sort_keys=True), json.dumps(params["levels"], sort_keys=True)


def _file(sidecar: pathlib.Path, suffix: str) -> pathlib.Path:
    return sidecar.with_name(sidecar.name.removesuffix(SIDECAR) + suffix)


def _read(sidecar: pathlib.Path, suffix: str, index: str | None = None) -> pd.DataFrame | None:
    path = _file(sidecar, suffix)
    if not path.exists():
        return None
    # Only empty cells are missing, so a network called "None" keeps its name
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""],
                        dtype={index: str} if index else None)
    return table.set_index(index) if index else table


def _contrast_groups(args, parser, matrices, atlas) -> tuple[dict, pd.DataFrame, list[str]]:
    """{matrix key: group} for the two contrast groups, their rows of the groups table, and those without a matrix."""
    table = pd.read_csv(args.groups, sep="\t" if args.groups.endswith((".tsv", ".txt")) else ",", dtype=str)
    for col in (args.participant_column, args.group_column, *(args.covariates or [])):
        if col not in table:
            raise SystemExit(f"{parser.prog}: {args.groups} has no column {col!r}")
    by_label = {str(k).removeprefix("sub-"): k for k in matrices}
    split = atlas.attrs.get(INPUT_ATTR, {}).get("split_runs", {})
    listed = {str(p).removeprefix("sub-") for p in table.loc[table[args.group_column].isin(args.contrast),
                                                             args.participant_column]}
    several = [f"sub-{i} ({', '.join(split[i])})" for i in sorted(listed & set(split))]
    if several:
        here = "XCP-D --combine-runs, or --combine-runs with --connectivity here" if args.input_type == "xcpd" else \
            "before running ConnGraph"
        raise SystemExit(f"{parser.prog}: comparing groups needs one matrix per participant, but {len(several)} have "
                         f"several runs: {'; '.join(several)}. Combine them ({here}) or pick one with "
                         "--bids-filter-file.")
    groups, missing = {}, []
    for name in args.contrast:
        ids = [str(p).removeprefix("sub-") for p in table.loc[table[args.group_column] == name, args.participant_column]]
        if not ids:
            raise SystemExit(f"{parser.prog}: no participant in group {name!r} of column {args.group_column!r}")
        absent = [f"sub-{i}" for i in ids if i not in by_label]
        ids = [i for i in ids if i in by_label]
        if not ids:
            raise SystemExit(f"{parser.prog}: no participant in group {name!r} has a matrix (missing: {', '.join(absent)})")
        missing += absent
        groups.update({by_label[i]: name for i in ids})
    if missing:
        warnings.warn(f"Leaving out {len(missing)} participant(s) in {args.groups} without a matrix: "
                      f"{', '.join(missing)}", stacklevel=2)
    keys = table[args.participant_column].map(lambda p: by_label.get(str(p).removeprefix("sub-")))
    rows = table[keys.isin(list(groups))].set_axis(keys[keys.isin(list(groups))])
    return groups, rows[~rows.index.duplicated()], missing


def _input_params(args, atlas, missing) -> dict:
    return {**atlas.attrs.get(INPUT_ATTR, {}), "fisher_z": not args.no_fisher_z,
            "groups_file": os.path.abspath(args.groups), "group_column": args.group_column,
            "contrast": list(args.contrast), "missing": missing}


def _compare(args, parser, compare, result, matrices, groups, rows, missing, atlas, out, command) -> None:
    from conngraph.graph_theory.compare import compare_groups

    if args.compare is None and result.net_corr_df is None and result.net_hemi_corr_df is None:
        compare = [c for c in compare if c != "blocks"]
    try:
        comparison = compare_groups(groups, args.contrast, result, matrices if "edges" in compare else None, compare,
                                    covariates=rows[args.covariates] if args.covariates else None,
                                    n_perms=args.n_perms, seed=args.random_seed, apply_fisher_z=not args.no_fisher_z,
                                    verbose=not args.quiet, correction=args.correction, alpha=args.alpha)
    except ValueError as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None
    params = comparison.params
    params["input"] = _input_params(args, atlas, missing)
    if result is not None:
        params["graph"] = {"options": result.params["options"], "levels": result.params["levels"]}
    params["command"] = command
    _write_comparison(comparison, out)
    if not args.no_report:
        _compare_report(args, out)
    if not args.quiet:
        print(f"[{parser.prog}] Group comparison: {out}")


def _nbs(args, parser, matrices, groups, missing, atlas, out, command) -> None:
    from conngraph.graph_theory.nbs import run_nbs

    if not args.no_fisher_z:
        matrices = _shared.fisher_z(matrices)
    g1, g2 = ({str(k).removeprefix("sub-"): matrices[k] for k, g in groups.items() if g == name}
              for name in args.contrast)
    try:
        result = run_nbs(g1, g2, thresh=args.nbs_thresh, k=args.n_perms, tail=args.nbs_tail, seed=args.random_seed,
                         verbose=not args.quiet, n_jobs=args.n_jobs)
    except Exception as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None

    result.params["input"] = _input_params(args, atlas, missing)
    result.params["command"] = command
    _write_nbs_tables(result, out)
    _shared.write_json(out / "parameters.json", result.params)
    if not args.no_report:
        _nbs_report(args, out)
        if not args.quiet:
            print(f"[{parser.prog}] Report: {out / 'nbs_report.html'}")


def _write_nbs_tables(result, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    labels = list(result.labels)
    n_comp = len(result.pval)
    pd.DataFrame({
        "component": np.arange(1, n_comp + 1),
        "edges": [int((result.adj == c + 1).sum() // 2) for c in range(n_comp)],
        "nodes": [int(np.any(result.adj == c + 1, axis=0).sum()) for c in range(n_comp)],
        "p": result.pval,
    }).to_csv(out / "nbs_components.tsv", sep="\t", index=False)
    rows = [{"roi_a": labels[i], "roi_b": labels[j], "component": int(result.adj[i, j]),
             "p": result.pval[int(result.adj[i, j]) - 1], "mean_group1": result.mean_g1[i, j],
             "mean_group2": result.mean_g2[i, j], "difference": result.mean_g1[i, j] - result.mean_g2[i, j]}
            for i, j in np.argwhere(np.triu(result.adj, 1) > 0)]
    pd.DataFrame(rows, columns=["roi_a", "roi_b", "component", "p", "mean_group1", "mean_group2", "difference"]).to_csv(
        out / "nbs_edges.tsv", sep="\t", index=False)
    pd.DataFrame({"permutation": np.arange(1, len(result.null) + 1), "largest_component": result.null}).to_csv(
        out / "nbs_null.tsv", sep="\t", index=False)
    for k, mean in ((1, result.mean_g1), (2, result.mean_g2)):
        pd.DataFrame(mean, index=pd.Index(labels, name="label"), columns=labels).to_csv(
            out / f"nbs_mean_group{k}.tsv", sep="\t")


def _read_nbs(out: pathlib.Path):
    """The NBS result _write_nbs_tables and parameters.json hold, read back from the files."""
    from conngraph.graph_theory.nbs import NBSResult
    from conngraph.graph_theory.runner import _read_table

    files = {"means": [out / f"nbs_mean_group{k}.tsv" for k in (1, 2)], "components": [out / "nbs_components.tsv"],
             "edges": [out / "nbs_edges.tsv"], "null": [out / "nbs_null.tsv"]}
    means = [_read_table(path, "label") for path in files["means"]]
    labels = list(means[0].index)
    components = _read_table(files["components"][0])
    edges = _read_table(files["edges"][0], text=("roi_a", "roi_b"))
    pos = {lab: k for k, lab in enumerate(labels)}
    adj = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b, c in zip(edges["roi_a"], edges["roi_b"], edges["component"]):
        adj[pos[a], pos[b]] = adj[pos[b], pos[a]] = c
    return NBSResult(pval=components["p"].to_numpy(float), adj=adj,
                     null=_read_table(files["null"][0])["largest_component"].to_numpy(),
                     labels=labels, params=json.loads((out / "parameters.json").read_text(encoding="utf-8")),
                     mean_g1=means[0].to_numpy(float), mean_g2=means[1].to_numpy(float),
                     files={k: [str(p.resolve()) for p in v] for k, v in files.items()})


def _write_comparison(comparison, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, table in comparison.tables.items():
        table.to_csv(out / f"{name}.tsv", sep="\t", index=False)
    _shared.write_json(out / "parameters.json", dict(comparison.params, tables=list(comparison.tables)))


def _read_comparison(out: pathlib.Path):
    """The comparison _write_comparison wrote, read back from the files."""
    from conngraph.graph_theory.compare import GroupComparisonResult
    from conngraph.graph_theory.runner import _read_table

    params = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    text = ("metric", "node", "roi_a", "roi_b", "network_a", "network_b")
    names = params.pop("tables")
    tables = {name: _read_table(out / f"{name}.tsv", text=text) for name in names}
    return GroupComparisonResult(tables=tables, params=params,
                                 files={name: [str((out / f"{name}.tsv").resolve())] for name in names})


def rebuild(args: argparse.Namespace, parser: argparse.ArgumentParser, session: str | None, variant: str | None,
            command: str) -> None:
    """The reports of the group results already written, without recomputing them."""
    out = _shared.run_folder(args.output_dir, "group", session, variant)
    built = []
    for folder, report, build in ((out, "graph_report.html", _graph_report),
                                  (out / "compare", "compare_report.html", _compare_report),
                                  (out / "nbs", "nbs_report.html", _nbs_report)):
        if (folder / "parameters.json").exists():
            build(args, folder)
            built.append(report)
    if not built:
        raise SystemExit(f"{parser.prog}: no group results in {out}; run the group level first")
    if not args.quiet:
        print(f"[{parser.prog}] Rebuilt {', '.join(built)} in {out}")


# Each report shows what was written, read back from the files

def _graph_report(args: argparse.Namespace, out: pathlib.Path) -> None:
    from conngraph.graph_theory.runner import _load

    _load(str(out)).save_report(out / "graph_report.html", nodes=_saved_nodes(args, out),
                                surfaces=_shared.surfaces(args), interactive_brain=args.interactive_brain)


def _compare_report(args: argparse.Namespace, out: pathlib.Path) -> None:
    label_col, network_col = _shared.node_columns(args)
    _read_comparison(out).save_report(out / "compare_report.html", nodes=_saved_nodes(args, out.parent),
                                      label_col=label_col, network_col=network_col)


def _nbs_report(args: argparse.Namespace, out: pathlib.Path) -> None:
    label_col, network_col = _shared.node_columns(args)
    _read_nbs(out).save_report(out / "nbs_report.html", nodes=_saved_nodes(args, out.parent),
                               surfaces=_shared.surfaces(args), label_col=label_col, network_col=network_col,
                               interactive_brain=args.interactive_brain)


def _saved_nodes(args: argparse.Namespace, folder: pathlib.Path) -> pd.DataFrame:
    """The node table saved with the run, with --coords added when it has no coordinates."""
    from conngraph.graph_theory.runner import _read_table

    saved = _read_table(folder / NODES_FILE, text=(_shared.node_columns(args)[0],))
    merged = _shared.report_nodes(args, saved)
    return saved if merged is None else merged
