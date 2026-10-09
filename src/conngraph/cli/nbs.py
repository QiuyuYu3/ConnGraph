"""
brainnet3d-nbs: network-based statistic between two groups, written as tables, a parameter record and a report.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

from brainnet3d.cli import _shared
from brainnet3d.loaders import INPUT_ATTR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainnet3d-nbs",
        description="Compare two groups of connectivity matrices with the network-based statistic and write "
                    "nbs_components.tsv, nbs_edges.tsv, nbs_null.tsv, parameters.json, dataset_description.json and "
                    "nbs_report.html.",
    )
    _shared.add_input_arguments(parser)
    g = parser.add_argument_group("network-based statistic")
    g.add_argument("--groups", required=True, help="table with one row per participant, e.g. participants.tsv")
    g.add_argument("--participant-column", default="participant_id", help="participant column (default: participant_id)")
    g.add_argument("--group-column", required=True, help="column holding each participant's group")
    g.add_argument("--contrast", nargs=2, required=True, metavar=("GROUP1", "GROUP2"), help="the two groups to compare")
    g.add_argument("--thresh", type=float, required=True, help="t-statistic threshold for keeping an edge")
    g.add_argument("--perms", type=int, default=5000, help="permutations (default: 5000)")
    g.add_argument("--tail", choices=["both", "left", "right"], default="both", help="tail of the test (default: both)")
    g.add_argument("--seed", type=int, help="random seed; a drawn seed is recorded when omitted")
    g.add_argument("--no-fisher-z", action="store_true", help="test raw r instead of Fisher z-transformed values")
    g.add_argument("--n-jobs", "--nprocs", type=int, default=-1,
                   help="parallel workers for the permutations (default: all but one); results do not depend on it")
    return parser


def main(argv: list[str] | None = None) -> int:
    from brainnet3d.graph_theory.nbs import run_nbs

    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    command = _shared.command_line(parser.prog, argv)
    return _shared.run_all(args, parser, lambda session, variant, out: _run(
        args, parser, variant, out, command, run_nbs, session))


def _run(args, parser, variant, out, command, run_nbs, session=None) -> None:
    matrices, atlas = _shared.load_input(args, parser, variant, session)
    label_col, network_col = _shared.node_columns(args)

    table = pd.read_csv(args.groups, sep="\t" if args.groups.endswith((".tsv", ".txt")) else ",", dtype=str)
    for col in (args.participant_column, args.group_column):
        if col not in table:
            raise SystemExit(f"{parser.prog}: {args.groups} has no column {col!r}")
    by_label = {str(k).removeprefix("sub-"): k for k in matrices}
    split = atlas.attrs.get(INPUT_ATTR, {}).get("split_runs", {})
    listed = {str(p).removeprefix("sub-") for p in table.loc[table[args.group_column].isin(args.contrast),
                                                             args.participant_column]}
    several = [f"sub-{i} ({', '.join(split[i])})" for i in sorted(listed & set(split))]
    if several:
        here = "XCP-D --combine-runs, or --combine-runs with --connectivity here" if args.input_type == "xcpd" else \
            "before running brainnet3d"
        raise SystemExit(f"{parser.prog}: NBS needs one matrix per participant, but {len(several)} have several runs: "
                         f"{'; '.join(several)}. Combine them ({here}) or pick one with --bids-filter-file.")
    groups, missing = [], []
    for name in args.contrast:
        ids = [str(p).removeprefix("sub-") for p in table.loc[table[args.group_column] == name, args.participant_column]]
        if not ids:
            raise SystemExit(f"{parser.prog}: no participant in group {name!r} of column {args.group_column!r}")
        absent = [f"sub-{i}" for i in ids if i not in by_label]
        ids = [i for i in ids if i in by_label]
        if not ids:
            raise SystemExit(f"{parser.prog}: no participant in group {name!r} has a matrix (missing: {', '.join(absent)})")
        missing += absent
        groups.append(ids)
    if missing:
        warnings.warn(f"Leaving out {len(missing)} participant(s) in {args.groups} without a matrix: "
                      f"{', '.join(missing)}", stacklevel=2)

    if not args.no_fisher_z:
        matrices = _shared.fisher_z(matrices)
    g1, g2 = ({i: matrices[by_label[i]] for i in ids} for ids in groups)
    try:
        result = run_nbs(g1, g2, thresh=args.thresh, k=args.perms, tail=args.tail, seed=args.seed, verbose=not args.quiet,
                         n_jobs=args.n_jobs)
    except Exception as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None

    result.params["input"] ={**atlas.attrs.get(INPUT_ATTR, {}), "fisher_z": not args.no_fisher_z,
                              "groups_file": os.path.abspath(args.groups), "group_column": args.group_column,
                              "contrast": list(args.contrast), "missing": missing}
    result.params["command"] = command
    _write_tables(result, out)
    _shared.write_json(out / "parameters.json", result.params)
    _shared.write_description(out, "brainnet3d network-based statistic", args.input_dir, command)
    if not args.no_report:
        nodes = _shared.report_nodes(args, atlas, variant)
        result.save_report(out / "nbs_report.html", nodes=atlas if nodes is None else nodes,
                           surfaces=_shared.surfaces(args), static_brain=not args.no_static_brain,
                           label_col=label_col, network_col=network_col)
        if not args.quiet:
            print(f"[{parser.prog}] Report: {out / 'nbs_report.html'}")


def _write_tables(result, out: pathlib.Path) -> None:
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


if __name__ == "__main__":
    sys.exit(main())
