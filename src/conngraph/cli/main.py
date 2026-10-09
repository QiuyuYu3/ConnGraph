"""
conngraph: graph-theory metrics for each participant, then group tables, statistics and reports, as a BIDS App.
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from conngraph.cli import _shared
from conngraph.graph_theory.compare import COMPARISONS, CORRECTIONS
from conngraph.graph_theory.metrics import PARTITIONS, SIGNED_FALLBACKS
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

    def given(self, args: argparse.Namespace) -> list[str]:
        return [self.flags[dest] for dest in self.defaults if dest in vars(args)]

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
                   help="participant: seed for the random networks; group: seed for the permutations; a drawn "
                        "seed is recorded when omitted")
    s.add_argument("--n-jobs", "--nprocs", type=int, default=-1,
                   help="parallel workers (default: all but one); results do not depend on it")
    s.add_argument("--mem", "--mem-mb", type=int, metavar="MEMORY_MB",
                   help="participant: memory limit in MB for the reports built in parallel, which lowers --n-jobs "
                        "for them when needed (default: 90%% of this machine's memory)")
    s.add_argument("--no-report", action="store_true",
                   help="skip the HTML reports (participant: OUTPUT/sub-<label>.html; group: one per analysis)")
    s.add_argument("--reports-only", action="store_true",
                   help="rebuild the reports from the results already in OUTPUT without recomputing them; the "
                        "analysis options are read from those results")
    s.add_argument("--coords", help="table with label, x, y, z for the brain figures; Gordon coordinates are added "
                                    "automatically")
    s.add_argument("--surfaces", nargs="+", metavar="FILE",
                   help="brain for the figures: left and right .surf.gii, or one skull-stripped brain volume (NIfTI or "
                        "AFNI BRIK/HEAD) whose smoothed outline is cut at x = 0 into hemispheres (default: fsLR 32k "
                        "midthickness)")
    s.add_argument("--interactive-brain", action="store_true",
                   help="also save each brain figure as a rotatable 3-D page in figures/, linked from the report")

    p = _LevelOptions(parser, "participant", "participant level: graph metrics")
    p.add("--level", choices=["node", "network", "both"],
          help="levels to compute (default: both; node for fnirs-pipe, which has no networks)")
    p.add("--hemi-split", choices=["true", "false", "both"], default="true",
          help="network level: one node per network and hemisphere (true, default), per network (false), or both")
    p.add("--metrics", nargs="+", metavar="NAME",
          help='metric or metric.variant names, or "all" (default: the default variant of each metric)')
    p.add("--hemi-col", default="hemisphere", help="hemisphere column (default: hemisphere)")
    p.add("--graph-method", choices=GRAPH_METHODS, help="graph construction, e.g. tmfg; required at this level")
    p.add("--graph-param", type=graph_param, action="append", metavar="KEY=VALUE",
          help="parameter of the graph method, e.g. density=0.1 or density=0.1,0.2,0.3 for a range")
    p.add("--sign", choices=SIGNS, help="treatment of negative weights (default depends on the method)")
    p.add("--signed-fallback", choices=SIGNED_FALLBACKS, default="abs",
          help="when negative weights remain, metrics without a signed form use their absolute values (abs, default) "
               "or only the positive weights (positive)")
    p.add("--partition", nargs="+", choices=PARTITIONS, metavar="{networks,louvain}",
          help="modules for participation, module_z and modularity: the node table's networks, Louvain modules, or "
               "both (default: both; louvain only without networks)")
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

    t = _LevelOptions(parser, "group", "group level: comparing two groups")
    t.add("--groups", help="table with one row per participant, e.g. participants.tsv; runs the comparisons")
    t.add("--participant-column", default="participant_id", help="participant column (default: participant_id)")
    t.add("--group-column", help="column holding each participant's group")
    t.add("--contrast", nargs=2, metavar=("GROUP1", "GROUP2"), help="the two groups to compare")
    t.add("--compare", nargs="*", choices=COMPARISONS, metavar="{metrics,blocks,edges}",
          help="what to compare with permutation t-tests: graph metrics, connectivity within and between networks, "
               "edges (default: metrics blocks); give none to skip")
    t.add("--covariates", nargs="+", metavar="COL",
          help="columns of the groups table to adjust for; text columns become indicator columns (not used by NBS)")
    t.add("--correction", choices=list(CORRECTIONS),
          help="required for the comparisons: which p-value marks a result significant, FDR-corrected, family-wise "
               "(max-T permutations) or uncorrected; the tables hold all three")
    t.add("--alpha", type=float, help="significance level for the comparisons (default: 0.05)")
    t.add("--n-perms", type=int, default=5000, help="permutations for the comparisons and NBS (default: 5000)")
    t.add("--nbs-thresh", type=float, help="t-statistic threshold for keeping an edge; runs NBS")
    t.add("--nbs-tail", choices=["both", "left", "right"], default="both", help="tail of the test (default: both)")
    parser.set_defaults(_level_options=(p, t))
    return parser


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reports_only:
        if args.no_report:
            parser.error("--reports-only and --no-report cannot be combined")
        given = [flag for options in args._level_options for flag in options.given(args)]
        if given:
            parser.error(f"{given[0]} has no effect with --reports-only; the reports use the options saved with the "
                         "results")
    for options in args._level_options:
        options.resolve(args, parser)
    if args.analysis_level == "participant" and args.graph_method is None and not args.reports_only:
        parser.error("--graph-method is required at the participant level")
    if args.nbs_thresh is not None and not (args.groups and args.group_column and args.contrast):
        parser.error("--nbs-thresh needs --groups, --group-column and --contrast")
    if args.groups and not (args.group_column and args.contrast):
        parser.error("--groups needs --group-column and --contrast")
    for flag, value in (("--compare", args.compare), ("--covariates", args.covariates),
                        ("--correction", args.correction), ("--alpha", args.alpha)):
        if value is not None and not args.groups:
            parser.error(f"{flag} needs --groups")
    if args.alpha is None:
        args.alpha = 0.05
    if args.surfaces is not None and len(args.surfaces) > 2:
        parser.error("--surfaces takes a left and a right .surf.gii, or one brain volume")
    if args.n_perms < 1:
        parser.error("--n-perms must be at least 1")
    if args.mem is not None and args.mem < 1:
        parser.error("--mem must be at least 1")
    return args, parser


def main(argv: list[str] | None = None) -> int:
    from conngraph.cli import group, participant

    argv = sys.argv[1:] if argv is None else list(argv)
    args, parser = parse_args(argv)
    command = _shared.command_line(parser.prog, argv)
    if args.reports_only:
        run = participant.rebuild if args.analysis_level == "participant" else group.rebuild
    else:
        run = participant.run if args.analysis_level == "participant" else group.run
    reports = args.analysis_level == "participant" and not args.no_report
    with tempfile.TemporaryDirectory(prefix="conngraph_report_") as work:
        # each session and atlas leaves its report section here; the pages are put together once all have run
        args.report_dir = work if reports else None
        args.failed_reports = []
        code = _shared.run_all(args, parser, lambda session, variant: run(args, parser, session, variant, command),
                               "logs" if args.analysis_level == "participant" else "group")
        if reports:
            participant.write_reports(args, parser)
        if args.failed_reports:
            print(f"{parser.prog}: {len(args.failed_reports)} participant report(s) failed: "
                  f"{', '.join(args.failed_reports)}", file=sys.stderr)
            code = 1
    if not args.reports_only:
        _shared.write_description(args.output_dir, args.input_dir, command)
    return code


if __name__ == "__main__":
    sys.exit(main())
