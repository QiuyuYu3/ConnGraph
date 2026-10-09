"""
Input options, loading and output files shared by the brainnet3d command-line tools.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import shlex

import numpy as np
import pandas as pd

import brainnet3d
from brainnet3d.loaders import INPUT_ATTR


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="XCP-D derivatives folder, a flat folder of XCP-D matrices, or a folder of "
                                          "matrix files (see --input-format)")
    parser.add_argument("output_dir", help="folder for the results; created if missing")
    parser.add_argument("--version", action="version", version=f"%(prog)s {brainnet3d.__version__}")
    parser.add_argument("--quiet", action="store_true", help="print nothing but errors")

    g = parser.add_argument_group("input")
    g.add_argument("--input-format", choices=["xcpd", "xcpd-flat", "csv"], default="xcpd",
                   help="xcpd: XCP-D derivatives tree (default); xcpd-flat: XCP-D matrices collected in one folder; "
                        "csv: one matrix file per participant plus --nodes")
    g.add_argument("--atlas", help="atlas name as in the XCP-D file names, e.g. Gordon (xcpd, xcpd-flat)")
    g.add_argument("--atlas-file", help="atlas dseg.tsv (xcpd-flat)")
    g.add_argument("--session", default="ses-01", help="session label in the file names (default: ses-01)")
    g.add_argument("--task", default="rest", help="task label in the file names (default: rest)")
    g.add_argument("--space", default="fsLR", help="space label in the file names (default: fsLR)")
    g.add_argument("--participant-label", nargs="+", metavar="LABEL", help="participants to include, with or without sub-")
    g.add_argument("--nodes", help="node table for csv input: a label column, network labels and optionally x, y, z")
    g.add_argument("--pattern", default="*_matrix.csv", help="matrix file pattern for csv input (default: *_matrix.csv)")
    g.add_argument("--bad-node-threshold", type=float, default=0.9,
                   help="drop nodes with more than this fraction of missing values (default: 0.9)")
    g.add_argument("--drop-mode", choices=["union", "intersection"], default="union",
                   help="csv input: drop a node missing in any participant (union, default) or in all (intersection)")
    g.add_argument("--label-col", default="label", help="node label column (default: label)")
    g.add_argument("--network-col", default="network_label", help="network column (default: network_label, as in XCP-D)")

    r = parser.add_argument_group("report")
    r.add_argument("--no-report", action="store_true", help="skip the HTML report")
    r.add_argument("--no-static-brain", action="store_true", help="leave the static brain renderings out of the report")
    r.add_argument("--coords", help="table with label, x, y, z for the brain figures; Gordon coordinates are added "
                                    "automatically")
    r.add_argument("--surfaces", nargs=2, metavar=("LEFT", "RIGHT"),
                   help="left and right .surf.gii for the brain figures (default: fsLR 32k midthickness)")


def load_input(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict, pd.DataFrame]:
    """Matrices and node table as the chosen loader returns them, with what it read noted on the table."""
    labels = [s.removeprefix("sub-") for s in args.participant_label] if args.participant_label else None
    verbose = not args.quiet
    if args.input_format == "csv":
        if not args.nodes:
            parser.error("--nodes is required with --input-format csv")
        paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        files = {os.path.basename(p).split("_")[0]: p for p in paths}
        if labels is not None:
            files = {k: p for k, p in files.items() if k.removeprefix("sub-") in labels}
        if not files:
            raise SystemExit(f"{parser.prog}: no files matching {args.pattern} in {args.input_dir}")
        ds = brainnet3d.load_group(files, args.nodes, bad_node_threshold=args.bad_node_threshold, drop_mode=args.drop_mode)
        ds.nodes_df.attrs[INPUT_ATTR]["path"] = os.path.abspath(args.input_dir)
        return ds.matrices, ds.nodes_df
    if not args.atlas:
        parser.error("--atlas is required for XCP-D input")
    from brainnet3d.graph_theory import load_xcpd, load_xcpd_flat

    common = dict(session=args.session, task=args.task, space=args.space, subject_ids=labels,
                  bad_node_threshold=args.bad_node_threshold, verbose=verbose)
    if args.input_format == "xcpd":
        return load_xcpd(args.input_dir, args.atlas, **common)
    if not args.atlas_file:
        parser.error("--atlas-file is required with --input-format xcpd-flat")
    return load_xcpd_flat(args.input_dir, args.atlas_file, args.atlas, **common)


def report_nodes(args: argparse.Namespace, nodes: pd.DataFrame) -> pd.DataFrame | None:
    """The node table with x, y, z for the brain figures, from --coords or, for Gordon, the bundled table."""
    if {"x", "y", "z"} <= set(nodes.columns):
        return None
    if args.coords:
        table = pd.read_csv(args.coords, sep="\t" if args.coords.endswith((".tsv", ".txt")) else ",")
    elif (args.atlas or "").lower() == "gordon":
        try:
            table = brainnet3d.load_gordon_atlas()
        except Exception:
            return None
    else:
        return None
    coords = table[["label", "x", "y", "z"]].rename(columns={"label": args.label_col})
    merged = nodes.merge(coords, on=args.label_col, how="left")
    merged.attrs = dict(nodes.attrs)
    return merged


def surfaces(args: argparse.Namespace) -> tuple[str, str] | None:
    return tuple(args.surfaces) if args.surfaces else None


def command_line(prog: str, argv: list[str]) -> str:
    return " ".join([prog] + [shlex.quote(a) for a in argv])


def fisher_z(matrices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for sid, mat in matrices.items():
        z = np.arctanh(np.clip(mat.to_numpy(dtype=float), -1 + 1e-7, 1 - 1e-7))
        np.fill_diagonal(z, 0)
        out[sid] = pd.DataFrame(z, index=mat.index, columns=mat.columns)
    return out


def write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_description(out_dir: pathlib.Path, name: str, input_dir: str, command: str) -> None:
    """dataset_description.json marking the folder as derived data and naming the tool, version and command."""
    write_json(out_dir / "dataset_description.json", {
        "Name": name,
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "brainnet3d", "Version": brainnet3d.__version__, "Container": {"Type": "none"},
                         "Description": command}],
        "SourceDatasets": [{"URL": pathlib.Path(input_dir).resolve().as_uri()}],
    })
