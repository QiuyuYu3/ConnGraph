"""
brainnet3d-nbs: network-based statistic between two groups, written as tables, a parameter record and a report.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

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
    return parser


def main(argv: list[str] | None = None) -> int:
    from brainnet3d.graph_theory.nbs import run_nbs

    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output_dir)
    matrices, atlas = _shared.load_input(args, parser)

    table = pd.read_csv(args.groups, sep="\t" if args.groups.endswith((".tsv", ".txt")) else ",", dtype=str)
    for col in (args.participant_column, args.group_column):
        if col not in table:
            raise SystemExit(f"{parser.prog}: {args.groups} has no column {col!r}")
    by_label = {str(k).removeprefix("sub-"): k for k in matrices}
    groups, missing = [], []
    for name in args.contrast:
        ids = [str(p).removeprefix("sub-") for p in table.loc[table[args.group_column] == name, args.participant_column]]
        if not ids:
            raise SystemExit(f"{parser.prog}: no participant in group {name!r} of column {args.group_column!r}")
        missing += [f"sub-{i}" for i in ids if i not in by_label]
        groups.append(ids)
    if missing:
        raise SystemExit(f"{parser.prog}: participants in {args.groups} without a matrix: {', '.join(missing)}")

    if not args.no_fisher_z:
        matrices = _shared.fisher_z(matrices)
    g1, g2 = ({i: matrices[by_label[i]] for i in ids} for ids in groups)
    try:
        result = run_nbs(g1, g2, thresh=args.thresh, k=args.perms, tail=args.tail, seed=args.seed, verbose=not args.quiet)
    except Exception as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None

    command = _shared.command_line(parser.prog, argv)
    result.params["input"] = {**atlas.attrs.get(INPUT_ATTR, {}), "fisher_z": not args.no_fisher_z,
                              "groups_file": os.path.abspath(args.groups), "group_column": args.group_column,
                              "contrast": list(args.contrast)}
    result.params["command"] = command
    _write_tables(result, out)
    _shared.write_json(out / "parameters.json", result.params)
    _shared.write_description(out, "brainnet3d network-based statistic", args.input_dir, command)
    if not args.no_report:
        nodes = _shared.report_nodes(args, atlas)
        result.save_report(out / "nbs_report.html", nodes=atlas if nodes is None else nodes,
                           surfaces=_shared.surfaces(args), static_brain=not args.no_static_brain,
                           label_col=args.label_col, network_col=args.network_col)
        if not args.quiet:
            print(f"[{parser.prog}] Report: {out / 'nbs_report.html'}")
    return 0


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
