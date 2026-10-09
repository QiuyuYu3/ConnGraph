"""
conngraph: graph-theory metrics for each participant, then group tables, statistics and reports, as a BIDS App.
"""

from __future__ import annotations

import argparse
import sys

from conngraph.cli import _shared
from conngraph.graph_theory.sparsify import GRAPH_METHODS, SIGNS

LEVELS = ("participant", "group")


def graph_param(text: str) -> tuple[str, float | list[float]]:
    """Parse KEY=VALUE or KEY=V1,V2,... into a (key, number or list) pair."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"{text!r}: expected KEY=VALUE, e.g. density=0.1")
    name, value = text.split("=", 1)
    try:
        values = [float(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: values must be numbers") from None
    return name, values if len(values) > 1 else values[0]


class _LevelOptions:
    """An argument group whose options belong to one analysis level; their defaults apply only at that level."""

    def __init__(self, parser: argparse.ArgumentParser, level: str, title: str):
        self.level = level
        self.group = parser.add_argument_group(title)
        self.defaults: dict[str, object] = {}
        self.flags: dict[str, str] = {}

    def add(self, *flags: str, default=None, **kwargs) -> None:
        action = self.group.add_argument(*flags, default=argparse.SUPPRESS, **kwargs)
        self.defaults[action.dest] = False if kwargs.get("action") == "store_true" else default
        self.flags[action.dest] = flags[0]

    def resolve(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        given = [dest for dest in self.defaults if dest in vars(args)]
        if args.analysis_level != self.level and given:
            parser.error(f"{self.flags[given[0]]} belongs to the {self.level} level")
        for dest, value in self.defaults.items():
            if dest not in vars(args):
                setattr(args, dest, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conngraph",
        description="participant: compute graph-theory metrics and write one set of files per participant. "
                    "group: collect them into tables and a report, and compare groups when a groups table is given.",
    )
    _shared.add_input_arguments(parser)
    parser.add_argument("analysis_level", choices=LEVELS, help="participant or group")
    s = parser.add_argument_group("both levels")
    s.add_argument("--no-fisher-z", action="store_true",
                   help="participant: average raw r instead of Fisher z at the network level; group: test raw r")
    s.add_argument("--random-seed", type=int,
                   help="participant: seed for the random networks; group: seed for the NBS permutations; a drawn "
                        "seed is recorded when omitted")
    s.add_argument("--n-jobs", "--nprocs", type=int, default=-1,
                   help="parallel workers (default: all but one); results do not depend on it")

    p = _LevelOptions(parser, "participant", "participant level: graph metrics")
    p.add("--level", choices=["node", "network", "both"],
          help="levels to compute (default: both; node for fnirs-pipe, which has no networks)")
    p.add("--hemi-split", choices=["true", "false", "both"], default="true",
          help="network level: one node per network and hemisphere (true, default), per network (false), or both")
    p.add("--metrics", nargs="+", metavar="NAME",
          help='metric or metric.variant names, or "all" (default: the default variant of each metric)')
    p.add("--hemi-col", default="hemisphere", help="hemisphere column (default: hemisphere)")
    p.add("--graph-method", choices=GRAPH_METHODS, default="tmfg", help="graph construction (default: tmfg)")
    p.add("--graph-param", type=graph_param, action="append", metavar="KEY=VALUE",
          help="parameter of the graph method, e.g. density=0.1 or density=0.1,0.2,0.3 for a range")
    p.add("--sign", choices=SIGNS, help="treatment of negative weights (default depends on the method)")
    p.add("--network-graph-method", choices=GRAPH_METHODS, help="graph construction for the network level")
    p.add("--network-graph-param", type=graph_param, action="append", metavar="KEY=VALUE",
          help="parameter of the network-level graph method")
    p.add("--summary", choices=["auc", "mean"], default="auc", help="summary over a parameter range (default: auc)")
    p.add("--return-curves", action="store_true", help="also write the metrics at every value of a range")
    p.add("--normalize-weights", action="store_true", help="divide each matrix by its largest absolute weight")
    p.add("--n-random", type=int, default=0, help="random networks for normalization (default: 0, off)")
    p.add("--random-swaps", type=float, default=10, help="swaps per edge when randomizing (default: 10)")
    p.add("--exclude-networks", nargs="*", default=["None"], metavar="LABEL",
          help="network labels left out of the network level (default: None); give no labels to keep all")

    g = _LevelOptions(parser, "group", "group level: report")
    g.add("--no-report", action="store_true", help="skip the HTML reports")
    g.add("--no-static-brain", action="store_true", help="leave the static brain renderings out of the reports")
    g.add("--coords", help="table with label, x, y, z for the brain figures; Gordon coordinates are added "
                           "automatically")
    g.add("--surfaces", nargs=2, metavar=("LEFT", "RIGHT"),
          help="left and right .surf.gii for the brain figures (default: fsLR 32k midthickness)")
    t = _LevelOptions(parser, "group", "group level: network-based statistic between two groups")
    t.add("--groups", help="table with one row per participant, e.g. participants.tsv")
    t.add("--participant-column", default="participant_id", help="participant column (default: participant_id)")
    t.add("--group-column", help="column holding each participant's group")
    t.add("--contrast", nargs=2, metavar=("GROUP1", "GROUP2"), help="the two groups to compare")
    t.add("--nbs-thresh", type=float, help="t-statistic threshold for keeping an edge; runs NBS")
    t.add("--nbs-perms", type=int, default=5000, help="permutations (default: 5000)")
    t.add("--nbs-tail", choices=["both", "left", "right"], default="both", help="tail of the test (default: both)")
    parser.set_defaults(_level_options=(p, g, t))
    return parser


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = build_parser()
    args = parser.parse_args(argv)
    for options in args._level_options:
        options.resolve(args, parser)
    if args.nbs_thresh is not None and not (args.groups and args.group_column and args.contrast):
        parser.error("--nbs-thresh needs --groups, --group-column and --contrast")
    return args, parser


def main(argv: list[str] | None = None) -> int:
    from conngraph.cli import group, participant

    argv = sys.argv[1:] if argv is None else list(argv)
    args, parser = parse_args(argv)
    command = _shared.command_line(parser.prog, argv)
    run = participant.run if args.analysis_level == "participant" else group.run
    code = _shared.run_all(args, parser, lambda session, variant: run(args, parser, session, variant, command),
                           "logs" if args.analysis_level == "participant" else "group")
    _shared.write_description(args.output_dir, args.input_dir, command)
    return code


if __name__ == "__main__":
    sys.exit(main())
