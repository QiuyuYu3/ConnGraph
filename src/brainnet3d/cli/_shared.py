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
from brainnet3d.loaders import _EXTENSIONS, INPUT_ATTR, subject_id_from_path

NODES_FILE = "nodes.tsv"


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="XCP-D or fnirs-pipe derivatives folder, or a folder of matrix or time "
                                          "series files (see --input-type)")
    parser.add_argument("output_dir", help="folder for the results; created if missing")
    parser.add_argument("--version", action="version", version=f"%(prog)s {brainnet3d.__version__}")
    parser.add_argument("--quiet", action="store_true", help="print nothing but errors")

    g = parser.add_argument_group("input")
    g.add_argument("--input-type", choices=["xcpd", "fnirs-pipe", "matrix", "timeseries"], required=True,
                   help="xcpd: XCP-D derivatives tree, one result folder per atlas; fnirs-pipe: fnirs-pipe "
                        "derivatives tree, one result folder per chromophore; matrix: a folder with nodes.tsv and one "
                        "sub-<label>_matrix.<ext> per participant; timeseries: the same with "
                        "sub-<label>_timeseries.<ext>")
    g.add_argument("--connectivity", choices=["correlation", "partial-correlation"],
                   help="measure computed from time series: for timeseries input (default: correlation); for XCP-D "
                        "input, compute it from the time series files instead of reading XCP-D's matrices")
    g.add_argument("--shrinkage", action="store_true", help="with --connectivity, use Ledoit-Wolf shrinkage")
    g.add_argument("--values", choices=["r", "z"], default="r",
                   help="matrix input: correlations (r, default) or Fisher z values, converted back to r")
    g.add_argument("--mat-key", help="variable to read from .mat files holding more than one array")
    g.add_argument("--atlases", nargs="+", metavar="ATLAS",
                   help="xcpd: atlas names as in the file names, e.g. Gordon; each gets its own result folder")
    g.add_argument("--chromophore", nargs="+", choices=["hbo", "hbr"], default=["hbo", "hbr"],
                   help="fnirs-pipe: chromophores to analyse, each in its own result folder (default: hbo hbr)")
    g.add_argument("--session-id", help="xcpd and fnirs-pipe: session label, with or without ses- (default: each "
                                        "participant's one file, with or without a session)")
    g.add_argument("--task-id", default="rest", help="xcpd and fnirs-pipe: task label in the file names (default: rest)")
    g.add_argument("--space", default="fsLR", help="xcpd: space label in the file names (default: fsLR)")
    g.add_argument("--participant-label", nargs="+", metavar="LABEL", help="participants to include, with or without sub-")
    g.add_argument("--bad-node-threshold", type=float, default=0.9,
                   help="drop nodes with more than this fraction of missing values (default: 0.9)")
    g.add_argument("--drop-mode", choices=["union", "intersection"], default="union",
                   help="matrix, timeseries and fnirs-pipe input: drop a node missing in any participant (union, "
                        "default) or in all (intersection)")

    r = parser.add_argument_group("report")
    r.add_argument("--no-report", action="store_true", help="skip the HTML report")
    r.add_argument("--no-static-brain", action="store_true", help="leave the static brain renderings out of the report")
    r.add_argument("--coords", help="table with label, x, y, z for the brain figures; Gordon coordinates are added "
                                    "automatically")
    r.add_argument("--surfaces", nargs=2, metavar=("LEFT", "RIGHT"),
                   help="left and right .surf.gii for the brain figures (default: fsLR 32k midthickness)")


def input_variants(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str | None]:
    """One entry per separate analysis, named as its result folder: an atlas for XCP-D, a chromophore for fnirs-pipe."""
    if args.input_type == "xcpd":
        if not args.atlases:
            parser.error("--atlases is required for XCP-D input")
        return [f"atlas-{a}" for a in dict.fromkeys(args.atlases)]
    if args.input_type == "fnirs-pipe":
        return [f"chromo-{c}" for c in dict.fromkeys(args.chromophore)]
    return [None]


def load_input(args: argparse.Namespace, parser: argparse.ArgumentParser,
               variant: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Matrices and node table as the chosen loader returns them, with what it read noted on the table."""
    labels = [s.removeprefix("sub-") for s in args.participant_label] if args.participant_label else None
    verbose = not args.quiet
    kind = args.connectivity.replace("-", " ") if args.connectivity else None
    if args.input_type in ("matrix", "fnirs-pipe") and (kind or args.shrinkage):
        parser.error("--connectivity and --shrinkage need timeseries or XCP-D input")
    if args.shrinkage and not kind and args.input_type != "timeseries":
        parser.error("--shrinkage needs --connectivity")
    session = f"ses-{args.session_id.removeprefix('ses-')}" if args.session_id else None
    if args.input_type == "fnirs-pipe":
        return brainnet3d.load_fnirs_pipe(args.input_dir, variant.removeprefix("chromo-"), session=session,
                                          task=args.task_id, subject_ids=labels,
                                          bad_node_threshold=args.bad_node_threshold, drop_mode=args.drop_mode,
                                          verbose=verbose)
    if args.input_type in ("matrix", "timeseries"):
        nodes = os.path.join(args.input_dir, NODES_FILE)
        if not os.path.isfile(nodes):
            raise SystemExit(f"{parser.prog}: {args.input_dir} has no {NODES_FILE}")
        pattern = f"sub-*_{args.input_type}.*"
        paths = sorted(p for p in glob.glob(os.path.join(args.input_dir, pattern)) if p.endswith(_EXTENSIONS))
        files: dict[str, str] = {}
        for p in paths:
            sid = subject_id_from_path(p)
            if sid in files:
                raise SystemExit(f"{parser.prog}: two files for {sid}: {files[sid]} and {p}")
            files[sid] = p
        if labels is not None:
            files = {k: p for k, p in files.items() if k.removeprefix("sub-") in labels}
        if not files:
            raise SystemExit(f"{parser.prog}: no files matching {pattern} in {args.input_dir}")
        common = dict(bad_node_threshold=args.bad_node_threshold, drop_mode=args.drop_mode, mat_key=args.mat_key)
        if args.input_type == "matrix":
            ds = brainnet3d.load_group(files, nodes, values=args.values, **common)
        else:
            ds = brainnet3d.load_timeseries(files, nodes, kind=kind or "correlation", shrinkage=args.shrinkage,
                                            **common)
        ds.nodes_df.attrs[INPUT_ATTR]["path"] = os.path.abspath(args.input_dir)
        return ds.matrices, ds.nodes_df
    return brainnet3d.load_xcpd(args.input_dir, variant.removeprefix("atlas-"), session=session, task=args.task_id,
                                space=args.space, subject_ids=labels, bad_node_threshold=args.bad_node_threshold,
                                verbose=verbose, connectivity=kind, shrinkage=args.shrinkage)


def node_columns(args: argparse.Namespace) -> tuple[str, str]:
    """Label and network columns of the node table: XCP-D's dseg names the network column network_label."""
    return "label", "network_label" if args.input_type == "xcpd" else "network"


def report_nodes(args: argparse.Namespace, nodes: pd.DataFrame, variant: str | None = None) -> pd.DataFrame | None:
    """The node table with x, y, z for the brain figures, from --coords or, for Gordon, the bundled table."""
    if {"x", "y", "z"} <= set(nodes.columns):
        return None
    if args.coords:
        table = pd.read_csv(args.coords, sep="\t" if args.coords.endswith((".tsv", ".txt")) else ",")
    elif (variant or "").lower() == "atlas-gordon":
        try:
            table = brainnet3d.load_gordon_atlas()
        except Exception:
            return None
    else:
        return None
    label_col = node_columns(args)[0]
    coords = table[["label", "x", "y", "z"]].rename(columns={"label": label_col})
    merged = nodes.merge(coords, on=label_col, how="left")
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
