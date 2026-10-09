"""
Participant level: graph-theory metrics written as one set of files per participant, session and atlas.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from conngraph.cli import _shared

# Result attribute, file label and row name of each level
LEVEL_FILES = (("node_df", "node", "node"), ("network_df", "network", "network"),
               ("net_hemi_df", "networkhemi", "network"))
CONNECTIVITY_FILES = (("net_corr_df", "network"), ("net_hemi_corr_df", "networkhemi"))
GLOBAL_FILES = (("node", "node"), ("network", "network"), ("network_hemi", "networkhemi"))


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, session: str | None, variant: str | None,
        command: str) -> None:
    from conngraph.graph_theory import compute_graph_metrics

    matrices, atlas = _shared.load_input(args, parser, variant, session)
    ids = _shared.chosen(matrices, args)
    if not ids:
        raise SystemExit(f"{parser.prog}: none of --participant-label {' '.join(args.participant_label)} has a matrix")
    label_col, network_col = _shared.node_columns(args)
    level = args.level or ("node" if args.input_type == "fnirs-pipe" else "both")
    metrics = "all" if args.metrics == ["all"] else args.metrics
    try:
        result = compute_graph_metrics(
            {sid: matrices[sid] for sid in ids}, atlas, level=level,
            hemi_split={"true": True, "false": False, "both": "both"}[args.hemi_split],
            metrics=metrics, label_col=label_col, network_col=network_col, hemi_col=args.hemi_col,
            apply_fisher_z=not args.no_fisher_z, graph_method=args.graph_method,
            graph_params=dict(args.graph_param) if args.graph_param else None, sign=args.sign,
            network_graph_method=args.network_graph_method,
            network_graph_params=dict(args.network_graph_param) if args.network_graph_param else None,
            summary=args.summary, return_curves=args.return_curves, normalize_weights=args.normalize_weights,
            n_random=args.n_random, random_swaps=args.random_swaps, random_seed=args.random_seed,
            exclude_networks=args.exclude_networks, n_jobs=args.n_jobs, verbose=not args.quiet,
            signed_fallback=args.signed_fallback, partitions=tuple(args.partition) if args.partition else None,
        )
    except ValueError as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None
    result.params["command"] = command
    meshes = None
    nodes = _shared.report_nodes(args, atlas, variant)
    nodes = atlas if nodes is None else nodes
    for sid in ids:
        params = _write_participant(result, sid, pathlib.Path(args.output_dir), session, variant)
        if args.report_dir:
            if meshes is None:
                meshes = _meshes(args, nodes)
            _save_section(args, result, sid, matrices[sid], nodes, params, session, variant, meshes)
    if not args.quiet:
        print(f"[{parser.prog}] Wrote {len(ids)} participant(s) to {args.output_dir}")


def file_stem(sid: str, session: str | None, variant: str | None) -> tuple[str, str]:
    """Folder (relative to the output) and file name prefix of one participant's results."""
    label, run = _shared.participant_parts(sid)
    folder = "/".join(x for x in (f"sub-{label}", session) if x)
    return folder, "_".join(x for x in (f"sub-{label}", session, run, variant) if x)


def _write_participant(result, sid: str, out: pathlib.Path, session: str | None, variant: str | None) -> dict:
    folder, stem = file_stem(sid, session, variant)
    folder = out / folder
    folder.mkdir(parents=True, exist_ok=True)
    for attr, name, row in LEVEL_FILES:
        df = getattr(result, attr)
        if df is not None:
            row_values = df.loc[sid]
            metrics = list(dict.fromkeys(row_values.index.get_level_values(0)))
            table = pd.DataFrame({m: row_values[m] for m in metrics})
            table.index.name = row
            table.to_csv(folder / f"{stem}_level-{name}_metrics.tsv", sep="\t")
    for attr, name in CONNECTIVITY_FILES:
        wide = getattr(result, attr)
        if wide is not None:
            _square(wide.loc[sid]).to_csv(folder / f"{stem}_level-{name}_connectivity.tsv", sep="\t")
    if result.global_df is not None:
        for level, name in GLOBAL_FILES:
            if level in result.global_df.columns.get_level_values(0):
                result.global_df[level].loc[[sid]].to_csv(folder / f"{stem}_level-{name}_global.tsv", sep="\t",
                                                          index=False)
    if result.curves is not None:
        curves = result.curves[result.curves["ID"] == sid].drop(columns="ID")
        curves.to_csv(folder / f"{stem}_curves.tsv", sep="\t", index=False)
    params = dict(result.params, subjects=[sid],
                  failed={lvl: {sid: subs[sid]} for lvl, subs in result.params["failed"].items() if sid in subs},
                  warnings=[w for w in result.params["warnings"] if f" / {sid}: " in w])
    _shared.write_json(folder / f"{stem}_metrics.json", params)
    return params


def _meshes(args: argparse.Namespace, nodes: pd.DataFrame) -> list | None:
    """Surfaces for the interactive brain views, or None when they are off."""
    from conngraph.report import figures
    from conngraph.report.pages import _surfaces

    if not args.interactive_brain:
        return None
    if not {"x", "y", "z"} <= set(nodes.columns):
        return []
    return figures.surface_meshes(_surfaces(_shared.surfaces(args)))


def _save_section(args, result, sid, matrix, nodes, params, session, variant, meshes) -> None:
    """This participant's report section for one session and atlas, kept until every session has run."""
    from conngraph.report.participant import participant_section

    from conngraph.report.pages import _FigureFolder

    label, run = _shared.participant_parts(sid)
    parts = [x for x in (session, run, variant) if x]
    out = pathlib.Path(args.output_dir)
    figs = _FigureFolder(out / f"sub-{label}.html", out / f"sub-{label}" / "figures", file_stem(sid, session, variant)[1])
    section = participant_section(result, sid, matrix, nodes, params, "_".join(parts) or "results",
                                  " ".join(parts) or "Results", meshes, figs, _shared.surfaces(args))
    folder = pathlib.Path(args.report_dir) / f"sub-{label}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{section['id']}.json").write_text(json.dumps(section), encoding="utf-8")


def write_reports(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """OUTPUT/sub-<label>.html for every participant run, from the sections their sessions and atlases left."""
    from conngraph.report.participant import save_participant_report

    folders = sorted(pathlib.Path(args.report_dir).glob("sub-*"))
    for folder in folders:
        sections = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
        save_participant_report(folder.name.removeprefix("sub-"), sections,
                                pathlib.Path(args.output_dir) / f"{folder.name}.html")
    if folders and not args.quiet:
        print(f"[{parser.prog}] Wrote {len(folders)} participant report(s) to {args.output_dir}")


def _square(row: pd.Series) -> pd.DataFrame:
    """The network-by-network matrix from one row of upper-triangle "a__b" columns."""
    pairs = [key.split("__") for key in row.index]
    names = list(dict.fromkeys(a for a, _ in pairs))
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for (a, b), value in zip(pairs, row.to_numpy(dtype=float)):
        mat.loc[a, b] = mat.loc[b, a] = value
    mat.index.name = "network"
    return mat
