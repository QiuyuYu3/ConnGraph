"""
Input options, loading and output files shared by the conngraph command-line tools.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import pathlib
import shlex
import sys

import numpy as np
import pandas as pd

import conngraph
from conngraph.loaders import _EXTENSIONS, INPUT_ATTR, subject_id_from_path

NODES_FILE = "nodes.tsv"
# file name endings of the folder input types, after sub-<label>[_ses-<label>]_
FOLDER_FILES = {"matrix": "matrix.*", "timeseries": "timeseries.*", "mne-connectivity": "connectivity.nc"}
MODALITIES = ("fmri", "fnirs", "eeg", "meg")
# input types whose modality is known from the input itself
FIXED_MODALITY = {"xcpd": "fmri", "nirspipe": "fnirs"}


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_dir", help="XCP-D or NIRSPipe derivatives folder, or a folder of matrix, time series "
                                          "or MNE-Connectivity files (see --input-type)")
    parser.add_argument("output_dir", help="folder for the results; created if missing")
    parser.add_argument("--version", action="version", version=f"%(prog)s {conngraph.__version__}")
    parser.add_argument("--quiet", action="store_true", help="print nothing but errors")

    g = parser.add_argument_group("input")
    g.add_argument("--input-type", choices=["xcpd", "nirspipe", "matrix", "timeseries", "mne-connectivity"],
                   required=True,
                   help="xcpd: XCP-D derivatives tree, one result folder per atlas; nirspipe: NIRSPipe "
                        "derivatives tree, one result folder per chromophore; matrix: a folder with nodes.tsv and one "
                        "sub-<label>_matrix.<ext> per participant; timeseries: the same with "
                        "sub-<label>_timeseries.<ext>; mne-connectivity: a folder with one sub-<label>_connectivity.nc "
                        "saved by MNE-Connectivity per participant and an optional nodes.tsv, one result folder per "
                        "band")
    g.add_argument("--modality", choices=MODALITIES,
                   help="required for matrix, timeseries and mne-connectivity input (xcpd is fmri, nirspipe is "
                        "fnirs); sets the report's wording, and eeg places electrodes with standard names")
    g.add_argument("--connectivity", choices=["correlation", "partial-correlation"],
                   help="measure computed from time series: for timeseries input (default: correlation); for XCP-D "
                        "input, compute it from the time series files instead of reading XCP-D's matrices")
    g.add_argument("--shrinkage", action="store_true", help="with --connectivity, use Ledoit-Wolf shrinkage")
    g.add_argument("--combine-runs", action="store_true",
                   help="xcpd with --connectivity: z-score each run's time series and concatenate a participant's runs "
                        "before computing connectivity, as XCP-D's --combine-runs does")
    g.add_argument("--values", choices=["r", "z"], default="r",
                   help="matrix input: correlations (r, default) or Fisher z values, converted back to r")
    g.add_argument("--mat-key", help="variable to read from .mat files holding more than one array")
    g.add_argument("--atlases", nargs="+", metavar="ATLAS",
                   help="xcpd: atlas names as in the file names, e.g. Gordon; each gets its own result folder")
    g.add_argument("--chromophore", nargs="+", choices=["hbo", "hbr"], default=["hbo", "hbr"],
                   help="nirspipe: chromophores to analyse, each in its own result folder (default: hbo hbr)")
    g.add_argument("--session-id", nargs="+", metavar="LABEL",
                   help="sessions to analyse, with or without ses-, each in its own result folder (default: every "
                        "session in the input)")
    g.add_argument("--bids-filter-file", type=_read_json, metavar="FILE",
                   help='xcpd and nirspipe: JSON file of BIDS entities the files must carry, as for XCP-D; the "bold" '
                        '(xcpd) or "nirs" (nirspipe) entry is used, e.g. {"bold": {"acquisition": "mb", "run": 1}}')
    g.add_argument("--task-id", default="rest", help="xcpd and nirspipe: task label in the file names (default: rest)")
    g.add_argument("--space", default="fsLR", help="xcpd: space label in the file names (default: fsLR)")
    g.add_argument("--participant-label", nargs="+", metavar="LABEL", help="participants to include, with or without sub-")
    g.add_argument("--bad-node-threshold", type=float, default=0.9,
                   help="drop nodes with more than this fraction of missing values (default: 0.9)")
    g.add_argument("--drop-mode", choices=["union", "intersection"], default="union",
                   help="matrix, timeseries, nirspipe and mne-connectivity input: drop a node missing in any participant (union, "
                        "default) or in all (intersection)")


def input_variants(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str | None]:
    """One entry per separate analysis, named as its result folder: an atlas for XCP-D, a chromophore for NIRSPipe."""
    if args.input_type == "xcpd":
        if not args.atlases:
            parser.error("--atlases is required for XCP-D input")
        return [f"atlas-{a}" for a in dict.fromkeys(args.atlases)]
    if args.input_type == "nirspipe":
        return [f"chromo-{c}" for c in dict.fromkeys(args.chromophore)]
    if args.input_type == "mne-connectivity":
        paths = _input_files(args.input_dir, f"sub-*_{FOLDER_FILES['mne-connectivity']}")
        if not paths:
            raise SystemExit(f"{parser.prog}: no files matching sub-*_{FOLDER_FILES['mne-connectivity']} in "
                             f"{args.input_dir}")
        variants = []
        for label, first in {measure_label(_mne_method(p)): p for p in reversed(paths)}.items():
            try:
                bands = conngraph.mne_connectivity_bands(first)
            except ValueError as exc:
                raise SystemExit(f"{parser.prog}: {exc}") from None
            variants += [f"meas-{label}_{band_folder(b)}" for b in bands] if bands else [f"meas-{label}"]
        return sorted(variants)
    return [None]


def measure_label(method: str) -> str:
    """An MNE-Connectivity method as a file-name label, e.g. wpli2debiased."""
    return "".join(c for c in method if c.isalnum())


@functools.lru_cache(maxsize=None)
def _mne_method(path: str) -> str:
    from conngraph.derivatives import mne_connectivity_method

    return mne_connectivity_method(path)


def band_folder(band: tuple[float, float]) -> str:
    """Result folder of a frequency band, e.g. band-8to13Hz; decimals are written with p (12p5)."""
    return "band-" + "to".join(f"{f:g}".replace(".", "p") for f in band) + "Hz"


def input_sessions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str | None]:
    """Sessions analysed one at a time: those named by --session-id, else every session in the input, else None."""
    filters = bids_filters(args, parser)
    if "ses" in filters and "Query.ANY" not in filters["ses"]:
        if args.session_id:
            parser.error("give sessions either with --session-id or in --bids-filter-file, not both")
        return [f"ses-{v}" if v else None for v in dict.fromkeys(filters["ses"])]
    if args.session_id:
        return [f"ses-{s.removeprefix('ses-')}" for s in dict.fromkeys(args.session_id)]
    if args.input_type in FOLDER_FILES:
        found = {_file_session(p) for p in _input_files(args.input_dir, f"sub-*_{FOLDER_FILES[args.input_type]}")}
        if None in found and len(found) > 1:
            raise SystemExit(f"{parser.prog}: {args.input_dir} mixes files with and without a session (ses-) label")
    else:
        labels = [s.removeprefix("sub-") for s in args.participant_label] if args.participant_label else ["*"]
        datatype = "nirs" if args.input_type == "nirspipe" else "func"
        found = {os.path.basename(os.path.dirname(p)) for lbl in labels
                 for p in glob.glob(os.path.join(args.input_dir, f"sub-{lbl}", "ses-*", datatype))}
    return sorted(s for s in found if s) or [None]


def bids_filters(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, list]:
    """The input type's entry of --bids-filter-file, keyed by file-name entity."""
    from conngraph.derivatives import _check_filters

    if args.bids_filter_file is None:
        return {}
    if args.input_type not in ("xcpd", "nirspipe"):
        parser.error("--bids-filter-file needs xcpd or nirspipe input")
    try:
        return _check_filters(args.bids_filter_file.get("nirs" if args.input_type == "nirspipe" else "bold"))
    except ValueError as exc:
        parser.error(str(exc))


def _read_json(path: str) -> dict:
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError(f"{path} must hold a JSON object")
    return data


def run_all(args: argparse.Namespace, parser: argparse.ArgumentParser, run, error_folder: str) -> int:
    """Call run(session, variant) for each session and variant; with several, a failure is noted and the rest run."""
    kind = args.connectivity
    if args.input_type in ("matrix", "nirspipe", "mne-connectivity") and (kind or args.shrinkage):
        parser.error("--connectivity and --shrinkage need timeseries or XCP-D input")
    if args.shrinkage and not kind and args.input_type != "timeseries":
        parser.error("--shrinkage needs --connectivity")
    if args.combine_runs and (args.input_type != "xcpd" or not kind):
        parser.error("--combine-runs needs XCP-D input with --connectivity")
    variants = input_variants(args, parser)
    runs = [(s, v) for s in input_sessions(args, parser) for v in variants]
    failed = []
    for session, variant in runs:
        name = " ".join(x for x in (session, variant) if x)
        if len(runs) == 1:
            run(session, variant)
            break
        if not args.quiet:
            print(f"[{parser.prog}] {name}")
        try:
            run(session, variant)
        except (Exception, SystemExit) as exc:
            message = str(exc).removeprefix(f"{parser.prog}: ") or type(exc).__name__
            out = run_folder(args.output_dir, error_folder, session, variant)
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(message + "\n", encoding="utf-8")
            print(f"{parser.prog}: {name} failed: {message}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"{parser.prog}: {len(failed)} of {len(runs)} runs failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def run_folder(output_dir: str, folder: str, session: str | None, variant: str | None) -> pathlib.Path:
    # a variant of several entities (meas-wpli_band-8to13Hz) gets one folder level per entity
    return pathlib.Path(output_dir, folder, session or "", *(variant or "").split("_"))


def participant_parts(sid: str) -> tuple[str, str | None]:
    """The participant label without sub-, and the run entity of a run analysed on its own."""
    label, _, run = str(sid).removeprefix("sub-").partition("_")
    return label, run or None


def chosen(matrices: dict, args: argparse.Namespace) -> list[str]:
    """Matrix keys of the participants named by --participant-label, or all of them."""
    if not args.participant_label:
        return list(matrices)
    labels = {s.removeprefix("sub-") for s in args.participant_label}
    return [sid for sid in matrices if participant_parts(sid)[0] in labels]


def _input_files(folder: str, pattern: str) -> list[str]:
    extensions = (".nc",) if pattern.endswith(".nc") else _EXTENSIONS
    return sorted(p for p in glob.glob(os.path.join(folder, pattern)) if p.endswith(extensions))


def _file_session(path: str) -> str | None:
    parts = os.path.basename(path).split("_")
    return parts[1] if len(parts) > 2 and parts[1].startswith("ses-") else None


def _load_mne(args: argparse.Namespace, files: dict[str, str], variant: str | None, nodes: str | None,
              session: str | None) -> tuple[dict, pd.DataFrame]:
    from conngraph.derivatives import MNE_MEASURES

    bands = conngraph.mne_connectivity_bands(next(iter(files.values()))) or []
    band = next((b for b in bands if band_folder(b) == variant.partition("_")[2]), None)
    matrices, table = conngraph.load_mne_connectivity(files, band, nodes, bad_node_threshold=args.bad_node_threshold,
                                                       drop_mode=args.drop_mode, verbose=not args.quiet,
                                                       montage=_montage(args))
    # phase-based and coherence measures are averaged and compared as they are, not as Fisher z
    if not MNE_MEASURES[table.attrs[INPUT_ATTR]["measure"]][2]:
        args.no_fisher_z = True
    table.attrs[INPUT_ATTR].update(path=os.path.abspath(args.input_dir), session=session)
    return matrices, table


def load_input(args: argparse.Namespace, parser: argparse.ArgumentParser,
               variant: str | None = None, session: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Every participant's matrix and the node table, so nodes are dropped the same way whoever is analysed."""
    verbose = not args.quiet
    kind = args.connectivity.replace("-", " ") if args.connectivity else None
    filters = bids_filters(args, parser)
    if "ses" in filters:
        filters["ses"] = [session.removeprefix("ses-") if session else None]
    if args.input_type == "nirspipe":
        return _with_modality(args, conngraph.load_nirspipe(
            args.input_dir, variant.removeprefix("chromo-"), session=session, task=args.task_id,
            bad_node_threshold=args.bad_node_threshold, drop_mode=args.drop_mode, verbose=verbose, bids_filters=filters))
    if args.input_type in FOLDER_FILES:
        nodes = os.path.join(args.input_dir, NODES_FILE)
        if not os.path.isfile(nodes) and args.input_type != "mne-connectivity":
            raise SystemExit(f"{parser.prog}: {args.input_dir} has no {NODES_FILE}")
        ending = FOLDER_FILES[args.input_type]
        pattern = f"sub-*_{session}_{ending}" if session else f"sub-*_{ending}"
        paths = _input_files(args.input_dir, pattern)
        if args.input_type == "mne-connectivity":
            measure = variant.partition("_")[0].removeprefix("meas-")
            paths = [p for p in paths if measure_label(_mne_method(p)) == measure]
        files: dict[str, str] = {}
        for p in paths:
            sid = subject_id_from_path(p)
            if sid in files:
                raise SystemExit(f"{parser.prog}: two files for {sid}: {files[sid]} and {p}")
            files[sid] = p
        if not files:
            raise SystemExit(f"{parser.prog}: no files matching {pattern} in {args.input_dir}")
        if args.input_type == "mne-connectivity":
            return _with_modality(args, _load_mne(args, files, variant, nodes if os.path.isfile(nodes) else None,
                                                  session))
        common = dict(bad_node_threshold=args.bad_node_threshold, drop_mode=args.drop_mode, mat_key=args.mat_key)
        if args.input_type == "matrix":
            ds = conngraph.load_group(files, nodes, values=args.values, **common)
        else:
            ds = conngraph.load_timeseries(files, nodes, kind=kind or "correlation", shrinkage=args.shrinkage,
                                            **common)
        ds.nodes_df.attrs[INPUT_ATTR].update(path=os.path.abspath(args.input_dir), session=session)
        return _with_modality(args, (ds.matrices, ds.nodes_df))
    return _with_modality(args, conngraph.load_xcpd(
        args.input_dir, variant.removeprefix("atlas-"), session=session, task=args.task_id, space=args.space,
        bad_node_threshold=args.bad_node_threshold, verbose=verbose, connectivity=kind, shrinkage=args.shrinkage,
        bids_filters=filters, combine_runs=args.combine_runs))


def _montage(args: argparse.Namespace) -> str | None:
    """Template for electrode positions: only for EEG, and not when --coords gives them."""
    return None if args.coords or args.modality != "eeg" else args.montage or "auto"


def _with_modality(args: argparse.Namespace, loaded: tuple[dict, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    from conngraph.derivatives import _electrode_positions

    matrices, table = loaded
    if args.input_type in ("matrix", "timeseries"):
        table, electrodes = _electrode_positions(table, _montage(args), not args.quiet)
        if electrodes:
            table.attrs[INPUT_ATTR]["electrodes"] = electrodes
    table.attrs[INPUT_ATTR]["modality"] = args.modality
    return matrices, table


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
            table = conngraph.load_gordon_atlas()
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
    """The two hemisphere files of --surfaces; a single volume is turned into them (and cached) on first use."""
    from conngraph.viz.surface import _volume_surfaces, is_volume

    if not args.surfaces:
        return None
    if len(args.surfaces) == 1:
        if not is_volume(args.surfaces[0]):
            raise SystemExit(f"conngraph: --surfaces needs a left and a right .surf.gii, or one NIfTI or AFNI volume; "
                             f"got {args.surfaces[0]}")
        return _volume_surfaces(args.surfaces[0])
    return tuple(args.surfaces)


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


def write_description(output_dir: str, input_dir: str, command: str) -> None:
    """dataset_description.json marking the output folder as derived data and naming the tool, version and command."""
    write_json(pathlib.Path(output_dir) / "dataset_description.json", {
        "Name": "ConnGraph",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "ConnGraph", "Version": conngraph.__version__, "Container": {"Type": "none"},
                         "Description": command}],
        "SourceDatasets": [{"URL": pathlib.Path(input_dir).resolve().as_uri()}],
    })
