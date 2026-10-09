"""
brainnet3d-graph: graph-theory metrics of connectivity matrices, written as tables, a parameter record and a report.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from brainnet3d.cli import _shared
from brainnet3d.graph_theory.sparsify import GRAPH_METHODS, SIGNS


def graph_param(text: str) -> tuple[str, float | list[float]]:
    """KEY=VALUE or KEY=V1,V2,... ; a list of values is summarized over its range."""
    name, sep, value = text.partition("=")
    if not sep or not name or not value:
        raise argparse.ArgumentTypeError(f"{text!r}: expected KEY=VALUE, e.g. density=0.1 or density=0.1,0.2,0.3")
    try:
        values = [float(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: values must be numbers") from None
    return name, values if len(values) > 1 else values[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainnet3d-graph",
        description="Compute graph-theory metrics for every participant and write one table per level and metric, "
                    "parameters.json, dataset_description.json and graph_report.html.",
    )
    _shared.add_input_arguments(parser)
    g = parser.add_argument_group("graph metrics")
    g.add_argument("--level", choices=["node", "network", "both"], default="both", help="levels to compute (default: both)")
    g.add_argument("--hemi-split", choices=["true", "false", "both"], default="true",
                   help="network level: one node per network and hemisphere (true, default), per network (false), or both")
    g.add_argument("--metrics", nargs="+", metavar="NAME",
                   help='metric or metric.variant names, or "all" (default: the default variant of each metric)')
    g.add_argument("--hemi-col", default="hemisphere", help="hemisphere column (default: hemisphere)")
    g.add_argument("--no-fisher-z", action="store_true", help="average raw r instead of Fisher z at the network level")
    g.add_argument("--graph-method", choices=GRAPH_METHODS, default="tmfg", help="graph construction (default: tmfg)")
    g.add_argument("--graph-param", type=graph_param, action="append", metavar="KEY=VALUE",
                   help="parameter of the graph method, e.g. density=0.1 or density=0.1,0.2,0.3 for a range")
    g.add_argument("--sign", choices=SIGNS, help="treatment of negative weights (default depends on the method)")
    g.add_argument("--network-graph-method", choices=GRAPH_METHODS, help="graph construction for the network level")
    g.add_argument("--network-graph-param", type=graph_param, action="append", metavar="KEY=VALUE",
                   help="parameter of the network-level graph method")
    g.add_argument("--summary", choices=["auc", "mean"], default="auc", help="summary over a parameter range (default: auc)")
    g.add_argument("--return-curves", action="store_true", help="also write the metrics at every value of a range")
    g.add_argument("--normalize-weights", action="store_true", help="divide each matrix by its largest absolute weight")
    g.add_argument("--n-random", type=int, default=0, help="random networks for normalization (default: 0, off)")
    g.add_argument("--random-swaps", type=float, default=10, help="swaps per edge when randomizing (default: 10)")
    g.add_argument("--random-seed", type=int, help="seed for the random networks")
    g.add_argument("--exclude-networks", nargs="*", default=["None"], metavar="LABEL",
                   help='network labels left out of the network level (default: None); give no labels to keep all')
    g.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for the node level (default: all but one)")
    return parser


def main(argv: list[str] | None = None) -> int:
    from brainnet3d.graph_theory import compute_graph_metrics

    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    out = pathlib.Path(args.output_dir)
    matrices, atlas = _shared.load_input(args, parser)

    metrics = None
    if args.metrics:
        metrics = "all" if args.metrics == ["all"] else args.metrics
    try:
        result = compute_graph_metrics(
            matrices, atlas, level=args.level, hemi_split={"true": True, "false": False, "both": "both"}[args.hemi_split],
            metrics=metrics, label_col=args.label_col, network_col=args.network_col, hemi_col=args.hemi_col,
            apply_fisher_z=not args.no_fisher_z, graph_method=args.graph_method,
            graph_params=dict(args.graph_param) if args.graph_param else None, sign=args.sign,
            network_graph_method=args.network_graph_method,
            network_graph_params=dict(args.network_graph_param) if args.network_graph_param else None,
            summary=args.summary, return_curves=args.return_curves, normalize_weights=args.normalize_weights,
            n_random=args.n_random, random_swaps=args.random_swaps, random_seed=args.random_seed,
            exclude_networks=args.exclude_networks, n_jobs=args.n_jobs, output_dir=str(out), verbose=not args.quiet,
        )
    except ValueError as exc:
        raise SystemExit(f"{parser.prog}: {exc}") from None

    command = _shared.command_line(parser.prog, argv)
    result.params["command"] = command
    _shared.write_json(out / "parameters.json", result.params)
    _shared.write_description(out, "brainnet3d graph metrics", args.input_dir, command)
    if not args.no_report:
        result.save_report(out / "graph_report.html", nodes=_shared.report_nodes(args, atlas),
                           surfaces=_shared.surfaces(args), static_brain=not args.no_static_brain)
        if not args.quiet:
            print(f"[{parser.prog}] Report: {out / 'graph_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
