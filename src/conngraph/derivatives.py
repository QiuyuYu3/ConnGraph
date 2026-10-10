"""
Loaders for pipeline outputs: XCP-D and NIRSPipe, which name their connectivity
matrices *_relmat.tsv, and MNE-Connectivity's saved files. Each returns matrices
and a node table in the format expected by compute_graph_metrics.
"""

from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd

from conngraph.exceptions import DataValidationError
from conngraph.loaders import _drop_bad_nodes, record_input, series_to_matrices

_RELMAT = "_stat-pearsoncorrelation_relmat.tsv"
_SERIES = "_stat-mean_timeseries.tsv"


def load_xcpd(
    xcpd_dir: str,
    atlas: str,
    session: str | None = None,
    task: str = "rest",
    space: str = "fsLR",
    subject_ids: list[str] | None = None,
    bad_node_threshold: float = 0.9,
    verbose: bool = True,
    connectivity: str | None = None,
    shrinkage: bool = False,
    bids_filters: dict | None = None,
    combine_runs: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load correlation matrices from an XCP-D BIDS derivatives directory.

    Scans ``xcpd_dir`` for all subjects (or the provided ``subject_ids``),
    finds their ``*_stat-pearsoncorrelation_relmat.tsv`` files, and returns
    clean matrices ready to pass to ``compute_graph_metrics``.

    Parameters
    ----------
    xcpd_dir : str
        Root of the XCP-D derivatives folder.
    atlas : str
        Atlas name as it appears in the filename and atlas folder, e.g.
        ``"Gordon"``.  Used to locate both the matrix files and the atlas TSV.
    session : str | None
        BIDS session label, e.g. ``"ses-01"`` or ``"01"``; None (default) takes each
        subject's one matching file, with or without a session.
    task : str
        BIDS task label, e.g. ``"rest"``.
    space : str
        BIDS space label, e.g. ``"fsLR"``.
    subject_ids : list[str] | None
        Explicit list of subject IDs (without the ``sub-`` prefix).
        ``None`` → discover all ``sub-*`` directories automatically.
    bad_node_threshold : float
        Drop ROIs where the fraction of NaN values exceeds this threshold
        across any subject (union strategy, 0–1).
    verbose : bool
        Print how many subjects were found and loaded, and the atlas path.
    connectivity : str | None
        None reads XCP-D's Pearson matrices; "correlation" or "partial correlation"
        computes them from the ``*_stat-mean_timeseries.tsv`` files instead.
    shrinkage : with `connectivity`, estimate the covariance with Ledoit-Wolf shrinkage.
    bids_filters : dict | None
        BIDS entities the file names must carry, as in the "bold" entry of an
        XCP-D filter file, e.g. ``{"acquisition": "mb", "run": 1}``; a list
        accepts any of its values, None requires the entity to be absent.
        Values given here for task and space replace `task` and `space`.
    combine_runs : with `connectivity`, a subject whose files differ only in run gets each run's
        time series z-scored and concatenated in run order before connectivity is computed.
        Without it, each such run is returned on its own as ``"<subject>_run-<label>"``.

    Returns
    -------
    matrices : dict[str, pd.DataFrame]
        ``{subject_id: N×N DataFrame}`` with ROI labels as index and columns.
        Subject IDs are the bare IDs (without ``sub-`` prefix), matching the
        input ``subject_ids`` or whatever was discovered.
    atlas_df : pd.DataFrame
        Atlas label table loaded from
        ``{xcpd_dir}/atlases/atlas-{atlas}/atlas-{atlas}_dseg.tsv``.
        Contains at minimum ``label`` and ``network_label`` columns.

    Examples
    --------
    >>> matrices, atlas = load_xcpd(
    ...     xcpd_dir="derivatives/xcpd",
    ...     atlas="Gordon",
    ... )
    >>> from conngraph.graph_theory import compute_graph_metrics
    >>> results = compute_graph_metrics(matrices=matrices, atlas=atlas)
    """
    xcpd_dir = os.path.abspath(xcpd_dir)
    stem = _stem(connectivity)
    session = _session_label(session)

    # Discover subjects
    if subject_ids is None:
        subject_ids = _discover_subjects(xcpd_dir)
        if not subject_ids:
            raise FileNotFoundError(
                f"No sub-* directories found in: {xcpd_dir}"
            )
        if verbose:
            print(f"[load_xcpd] Found {len(subject_ids)} subject(s)")

    # Load matrices
    matrices: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    filters = _check_filters(bids_filters)
    tail = (f"task-{'*' if 'task' in filters else task}*_space-{'*' if 'space' in filters else space}"
            f"_seg-{atlas}{stem}")

    if combine_runs and connectivity is None:
        raise ValueError("combine_runs concatenates time series; give connectivity as well.")
    combined: dict[str, int] = {}
    split: dict[str, list[str]] = {}

    for sub in subject_ids:
        matches = _filtered(_find(xcpd_dir, sub, session, "func", tail), filters)
        runs = _runs_only(matches) if len(matches) > 1 else []

        if runs and combine_runs:
            matrices[sub] = _concat_runs(runs)
            combined[sub] = len(runs)
            continue
        if runs:
            split[sub] = _add_runs(matrices, sub, runs, lambda p: _read_xcpd_file(p, connectivity))
            continue
        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue

        matrices[sub] = _read_xcpd_file(matches[0], connectivity)

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError(
            "No matrices could be loaded. Check xcpd_dir, atlas, session, task, space." + _first_reasons(skipped)
        )
    if verbose and combined:
        print(f"[load_xcpd] Concatenated the runs of {len(combined)} subject(s)")

    if verbose:
        print(f"[load_xcpd] Loaded {len(matrices)} matrix/matrices")

    # Drop bad nodes (union across subjects)
    before = matrices
    if connectivity is not None:
        matrices = series_to_matrices(matrices, connectivity, shrinkage, bad_node_threshold, "union")
    elif bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold)

    # Load atlas
    atlas_path = os.path.join(
        xcpd_dir, "atlases", f"atlas-{atlas}", f"atlas-{atlas}_dseg.tsv"
    )
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(
            f"Atlas file not found: {atlas_path}\n"
            "If your atlas TSV is elsewhere, load it manually and pass it to "
            "compute_graph_metrics(atlas=...)."
        )
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    if verbose:
        print(f"[load_xcpd] Atlas: {atlas_path} ({len(atlas_df)} ROIs)")

    record_input(atlas_df, before, matrices, bad_node_threshold, "union", source="XCP-D", path=xcpd_dir,
                 atlas=atlas, space=_label(filters, "space", space), task=_label(filters, "task", task),
                 session=session, skipped=skipped, **_series_record(connectivity, shrinkage), **_filter_record(filters),
                 **({"combined_runs": combined} if combine_runs else {}), **({"split_runs": split} if split else {}))
    return matrices, atlas_df


def load_xcpd_flat(
    flat_dir: str,
    atlas_path: str,
    atlas: str,
    session: str = "ses-01",
    task: str = "rest",
    space: str = "fsLR",
    subject_ids: list[str] | None = None,
    bad_node_threshold: float = 0.9,
    verbose: bool = True,
    connectivity: str | None = None,
    shrinkage: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load correlation matrices from a **flat directory** produced by the XCP-D
    prep step (all ``.tsv`` files collected in one folder).

    Parameters
    ----------
    flat_dir : str
        Directory containing flat ``.tsv`` files named like::

            sub-{sub}_{session}_task-{task}[*]_space-{space}_seg-{atlas}
            _stat-pearsoncorrelation_relmat.tsv

    atlas_path : str
        Path to the atlas ``_dseg.tsv`` file (e.g. ``atlas-Gordon_dseg.tsv``).
    atlas : str
        Atlas name, e.g. ``"Gordon"``.  Used to build the glob pattern.
    session, task, space : BIDS entities matching the filenames.
    subject_ids : list[str] | None
        Explicit list of bare subject IDs.  ``None`` → discover from filenames.
    bad_node_threshold : float
        Drop ROIs with NaN fraction above this threshold (union strategy).
    verbose : bool
        Print how many subjects were found and loaded, and the atlas path.
    connectivity, shrinkage : same as :func:`load_xcpd`.

    Returns
    -------
    matrices : dict[str, pd.DataFrame]
    atlas_df : pd.DataFrame

    Examples
    --------
    >>> matrices, atlas = load_xcpd_flat(
    ...     flat_dir="xcpd_flat/Gordon",
    ...     atlas_path="derivatives/xcpd/atlases/atlas-Gordon/atlas-Gordon_dseg.tsv",
    ...     atlas="Gordon",
    ... )
    >>> results = compute_graph_metrics(matrices=matrices, atlas=atlas)
    """
    flat_dir   = os.path.abspath(flat_dir)
    atlas_path = os.path.abspath(atlas_path)

    stem = _stem(connectivity)
    glob_pattern = os.path.join(
        flat_dir,
        f"sub-*_{session}_task-{task}*_space-{space}_seg-{atlas}{stem}",
    )

    if subject_ids is None:
        subject_ids = _discover_subjects_flat(flat_dir, session, task, space, atlas, stem)
        if not subject_ids:
            raise FileNotFoundError(
                f"No matching files found in {flat_dir}\n"
                f"Pattern used: {glob_pattern}"
            )
        if verbose:
            print(f"[load_xcpd_flat] Found {len(subject_ids)} subject(s)")

    matrices: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    file_template = os.path.join(
        flat_dir,
        f"sub-{{sub}}_{session}_task-{task}*_space-{space}_seg-{atlas}{stem}",
    )

    for sub in subject_ids:
        matches = glob.glob(file_template.format(sub=sub))

        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue

        matrices[sub] = _read_xcpd_file(matches[0], connectivity)

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError("No matrices could be loaded. Check flat_dir and parameters.")

    if verbose:
        print(f"[load_xcpd_flat] Loaded {len(matrices)} matrix/matrices")

    before = matrices
    if connectivity is not None:
        matrices = series_to_matrices(matrices, connectivity, shrinkage, bad_node_threshold, "union")
    elif bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold)

    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    if verbose:
        print(f"[load_xcpd_flat] Atlas: {atlas_path} ({len(atlas_df)} ROIs)")

    record_input(atlas_df, before, matrices, bad_node_threshold, "union", source="XCP-D", path=flat_dir,
                 atlas=atlas, space=space, task=task, session=session, skipped=skipped,
                 **_series_record(connectivity, shrinkage))
    return matrices, atlas_df


CHROMOPHORES = ("hbo", "hbr")


def load_nirspipe(
    deriv_dir: str,
    chromophore: str,
    session: str | None = None,
    task: str = "rest",
    subject_ids: list[str] | None = None,
    bad_node_threshold: float = 0.9,
    drop_mode: str = "union",
    verbose: bool = True,
    bids_filters: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load one chromophore's channel-by-channel Pearson matrices from a NIRSPipe derivatives folder.

    Parameters
    ----------
    deriv_dir : NIRSPipe output folder holding ``sub-*/[ses-*/]nirs/``.
    chromophore : "hbo" or "hbr"; the two are never mixed.
    session : session label such as ``"ses-01"`` or ``"01"``; None takes each subject's one matching file, with or without a session.
    task : task label in the file names.
    subject_ids, bad_node_threshold, verbose : as in :func:`load_xcpd`.
    drop_mode : "union" drops a channel rejected in any participant, "intersection" only one rejected in all.
    bids_filters : as in :func:`load_xcpd`, e.g. the "nirs" entry of a filter file.

    Returns
    -------
    matrices : ``{subject_id: channel × channel DataFrame}``, channels named without the chromophore suffix.
    nodes : table with a ``label`` column, one row per channel.
    """
    if chromophore not in CHROMOPHORES:
        raise ValueError(f"chromophore must be one of {CHROMOPHORES}, got {chromophore!r}")
    deriv_dir = os.path.abspath(deriv_dir)
    session = _session_label(session)
    if subject_ids is None:
        subject_ids = _discover_subjects(deriv_dir)
        if not subject_ids:
            raise FileNotFoundError(f"No sub-* directories found in: {deriv_dir}")
    filters = _check_filters(bids_filters)
    tail = f"task-{'*' if 'task' in filters else task}*_chromo-{chromophore}_stat-pearson_relmat.tsv"

    def read(path):
        df = pd.read_csv(path, sep="\t", index_col=0)
        names = [str(c).removesuffix(f" {chromophore}") for c in df.columns]
        return df.set_axis(names, axis=0).set_axis(names, axis=1)

    matrices: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    split: dict[str, list[str]] = {}
    for sub in subject_ids:
        matches = _filtered(_find(deriv_dir, sub, session, "nirs", tail), filters)
        runs = _runs_only(matches) if len(matches) > 1 else []
        if runs:
            split[sub] = _add_runs(matrices, sub, runs, read)
            continue
        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue
        matrices[sub] = read(matches[0])

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError("No matrices could be loaded. Check deriv_dir, chromophore, session, task."
                                  + _first_reasons(skipped))
    if verbose:
        print(f"[load_nirspipe] Loaded {len(matrices)} {chromophore} matrix/matrices")

    before = matrices
    if bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold, drop_mode)
    nodes = pd.DataFrame({"label": list(next(iter(matrices.values())).columns)})
    record_input(nodes, before, matrices, bad_node_threshold, drop_mode, source="NIRSPipe", path=deriv_dir,
                 chromophore=chromophore, task=_label(filters, "task", task), session=session, skipped=skipped,
                 **_filter_record(filters), **({"split_runs": split} if split else {}))
    return matrices, nodes


# undirected MNE-Connectivity methods: full name, short label, Fisher z suits it, absolute value taken
MNE_MEASURES = {
    "coh": ("coherence", "Coh", False, False),
    "imcoh": ("absolute imaginary part of coherency", "|ImCoh|", False, True),
    "plv": ("phase-locking value", "PLV", False, False),
    "ciplv": ("corrected imaginary phase-locking value", "ciPLV", False, False),
    "ppc": ("pairwise phase consistency", "PPC", False, False),
    "pli": ("phase lag index", "PLI", False, False),
    "pli2_unbiased": ("unbiased squared phase lag index", "PLI²", False, False),
    "wpli": ("weighted phase lag index", "wPLI", False, False),
    "wpli2_debiased": ("debiased squared weighted phase lag index", "wPLI² (debiased)", False, False),
    "envelope correlation": ("amplitude envelope correlation", "AEC", True, False),
    "env_corr": ("amplitude envelope correlation", "AEC", True, False),
    "env_corr_orth": ("orthogonalized amplitude envelope correlation", "AEC", True, False),
}


def mne_connectivity_bands(path: str) -> list[tuple[float, float]] | None:
    """Frequency bands of an MNE-Connectivity file as (low, high) Hz; None when it has no frequency axis."""
    return _mne_bands(_read_mne(path), path)


def load_mne_connectivity(
    files: dict[str, str],
    band: tuple[float, float] | None = None,
    nodes: str | pd.DataFrame | None = None,
    bad_node_threshold: float = 0.9,
    drop_mode: str = "union",
    verbose: bool = True,
    montage: str | None = "auto",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """One band's matrices from MNE-Connectivity files ({subject_id: path}), with the node table in file order."""
    # montage: "auto" (colin27_1005), a template from MNI_MONTAGES, or None; used only when the table lacks x, y, z
    matrices: dict[str, pd.DataFrame] = {}
    methods, epochs = set(), []
    for sid, path in files.items():
        conn = _read_mne(path)
        methods.add(conn.method)
        if conn.n_epochs_used is not None:
            epochs.append(int(conn.n_epochs_used))
        full = _mne_square(conn)
        bands = _mne_bands(conn, path)
        if bands is not None:
            if band is None or tuple(band) not in bands:
                raise DataValidationError(f"'{path}' has no {band} Hz band; it holds {bands}.")
            full = full[..., bands.index(tuple(band))]
        elif full.ndim == 3:
            full = full[..., 0]
        matrices[sid] = pd.DataFrame(full, index=list(conn.names), columns=list(conn.names))
    if len(methods) > 1:
        raise DataValidationError(f"The files mix connectivity methods: {sorted(methods)}.")
    method = methods.pop()
    name, label, _, absolute = MNE_MEASURES[method]
    names = list(dict.fromkeys(n for m in matrices.values() for n in m.columns))
    matrices = {sid: (m.abs() if absolute else m).reindex(index=names, columns=names) for sid, m in matrices.items()}
    if verbose:
        print(f"[load_mne_connectivity] Loaded {len(matrices)} {method} matrix/matrices")

    before = matrices
    if bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold, drop_mode)
    kept = list(next(iter(matrices.values())).columns)
    table = pd.DataFrame({"label": kept}) if nodes is None else _mne_nodes(nodes, kept)
    table, electrodes = _electrode_positions(table, montage, verbose)
    record_input(table, before, matrices, bad_node_threshold, drop_mode, source="MNE-Connectivity",
                 measure=method, measure_name=name, measure_label=label, band=list(band) if band else None,
                 epochs=[min(epochs), max(epochs)] if epochs else None,
                 **({"electrodes": electrodes} if electrodes else {}))
    return matrices, table


# MNE templates with positions in MNI space; MNE 1.13 renamed the standard_* ones to colin27_*
MNI_MONTAGES = ("colin27_1005", "colin27_1020", "colin27_alphabetic", "colin27_postfixed", "colin27_prefixed",
                "colin27_primed", "mgh60", "mgh70")


def montage_coordinates(names: list[str], montage: str = "colin27_1005") -> pd.DataFrame:
    """MNI x, y, z in mm of the named electrodes in an MNE template, matched in any case; unknown names are left out."""
    import mne

    template = _mni_montage(montage)
    installed = template if template in mne.channels.get_builtin_montages() else template.replace("colin27_", "standard_")
    positions = mne.channels.make_standard_montage(installed).get_positions()["ch_pos"]
    lookup = {k.lower(): 1000 * np.asarray(v) for k, v in positions.items()}
    rows = [(n, *lookup[str(n).lower()]) for n in names if str(n).lower() in lookup]
    return pd.DataFrame(rows, columns=["label", "x", "y", "z"])


def _mni_montage(montage: str) -> str:
    template = "colin27_" + montage.removeprefix("standard_") if montage.startswith("standard_") else montage
    if template not in MNI_MONTAGES:
        raise DataValidationError(f"{montage!r} is not an MNE template in MNI space; use one of "
                                  f"{', '.join(MNI_MONTAGES)}, or give the positions as x, y, z in the node table "
                                  "(--coords on the command line).")
    return template


def _electrode_positions(table: pd.DataFrame, montage: str | None, verbose: bool) -> tuple[pd.DataFrame, str | None]:
    if montage is None:
        return table, None
    if {"x", "y", "z"} <= set(table.columns):
        if montage != "auto":
            raise DataValidationError("The node table already has x, y, z; leave out the montage.")
        return table, None
    template = "colin27_1005" if montage == "auto" else _mni_montage(montage)
    coords = montage_coordinates(list(table["label"]), template)
    if coords.empty:
        if montage != "auto":
            raise DataValidationError(f"None of the node names are in the {template} template.")
        return table, None
    missing = [n for n in table["label"] if n not in set(coords["label"])]
    if verbose:
        print(f"[load_mne_connectivity] Electrode positions from MNE's {template} template for {len(coords)} of "
              f"{len(table)} nodes" + (f"; not in it: {', '.join(map(str, missing))}" if missing else ""))
    return table.merge(coords, on="label", how="left"), template


def _read_mne(path: str):
    from mne_connectivity import read_connectivity

    conn = read_connectivity(path)
    if conn.is_epoched:
        raise DataValidationError(f"'{path}' holds one matrix per epoch; average them first, e.g. with "
                                  "conn.combine(), and save the result.")
    if conn.method not in MNE_MEASURES:
        raise DataValidationError(f"'{path}' holds {conn.method!r} connectivity; supported undirected methods are "
                                  f"{', '.join(MNE_MEASURES)}.")
    if "times" in conn.dims and conn.get_data("raveled").shape[-1] > 1:
        raise DataValidationError(f"'{path}' holds time-resolved connectivity; average it over time first.")
    return conn


def _mne_bands(conn, path: str) -> list[tuple[float, float]] | None:
    if "freqs" not in conn.dims:
        return None
    used = conn.attrs.get("freqs_used")
    if used is None or any(np.ndim(b) != 1 or len(b) != 2 for b in used):
        raise DataValidationError(f"'{path}' holds single frequencies; compute connectivity with faverage=True "
                                  "to average them into bands.")
    return [(float(lo), float(hi)) for lo, hi in used]


def _mne_square(conn) -> np.ndarray:
    """Node × node (× band) array from the stored connections, each pair filled from whichever side was stored."""
    n, data = conn.n_nodes, np.asarray(conn.get_data("raveled"), dtype=float)
    indices = conn.indices
    if isinstance(indices, tuple):
        rows, cols = (np.asarray(i) for i in indices)
    elif indices == "symmetric":
        rows, cols = np.triu_indices(n)
    elif indices in ("lower", "upper"):
        rows, cols = np.tril_indices(n, -1) if indices == "lower" else np.triu_indices(n, 1)
    else:
        rows, cols = np.unravel_index(np.arange(n * n), (n, n))
    full = np.full((n, n, *data.shape[1:]), np.nan)
    full[rows, cols] = data
    # all-to-all results store one triangle and leave the other at zero or missing
    lower, upper = np.tril_indices(n, -1), np.triu_indices(n, 1)
    unset = lambda tri: bool(np.all((full[tri] == 0) | np.isnan(full[tri])))
    if unset(upper) and not unset(lower):
        full[lower[::-1]] = full[lower]
    elif unset(lower) and not unset(upper):
        full[upper[::-1]] = full[upper]
    mirror = np.swapaxes(full, 0, 1)
    gap = np.isnan(full) & ~np.isnan(mirror)
    full[gap] = mirror[gap]
    full[np.arange(n), np.arange(n)] = 0.0
    return full


def _mne_nodes(nodes: str | pd.DataFrame, names: list[str]) -> pd.DataFrame:
    from conngraph.loaders import _read_nodes

    table = _read_nodes(nodes)
    missing = [n for n in names if n not in set(table["label"])]
    if missing:
        raise DataValidationError(f"The node table has no row for {', '.join(missing[:10])}"
                                  + (" …" if len(missing) > 10 else "") + ".")
    return table.set_index("label").loc[names].reset_index()


def _find(root: str, sub: str, session: str | None, datatype: str, tail: str) -> list[str]:
    """A subject's files ending in tail, in the given session or, without one, in no or any session."""
    folder = os.path.join(root, f"sub-{sub}")
    if session:
        return glob.glob(os.path.join(folder, session, datatype, f"sub-{sub}_{session}_{tail}"))
    return (glob.glob(os.path.join(folder, datatype, f"sub-{sub}_{tail}"))
            + glob.glob(os.path.join(folder, "ses-*", datatype, f"sub-{sub}_ses-*_{tail}")))


# BIDS entity names as pybids and XCP-D filter files spell them, and the keys they take in file names
ENTITIES = {
    "subject": "sub", "session": "ses", "task": "task", "acquisition": "acq", "ceagent": "ce",
    "reconstruction": "rec", "direction": "dir", "run": "run", "echo": "echo", "part": "part", "space": "space",
    "cohort": "cohort", "resolution": "res", "density": "den", "desc": "desc", "atlas": "atlas",
    "segmentation": "seg", "statistic": "stat", "chromophore": "chromo",
}
_ABSENT = (None, "Query.NONE")
_PRESENT = "Query.ANY"


def _check_filters(filters: dict | None) -> dict[str, list]:
    """Filters keyed by file-name entity: accepted values without key-, None for absent, "Query.ANY" for any."""
    out = {}
    for name, value in (filters or {}).items():
        key = ENTITIES.get(name, name if name in ENTITIES.values() else None)
        if key is None:
            raise ValueError(f"Unknown BIDS entity {name!r} in the filters; use one of {sorted(ENTITIES)}.")
        if value == "Query.OPTIONAL":
            continue
        values = value if isinstance(value, list) else [value]
        out[key] = [None if v in _ABSENT else v if v == _PRESENT else str(v).removeprefix(f"{key}-") for v in values]
    return out


def _entities(path: str) -> dict[str, str]:
    parts = os.path.basename(path).split(".")[0].split("_")
    return dict(p.split("-", 1) for p in parts if "-" in p)


def _same(key: str, a: str, b: str) -> bool:
    if b == _PRESENT:
        return True
    if key in ("run", "echo") and a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return a == b


def _filtered(paths: list[str], filters: dict[str, list]) -> list[str]:
    def keep(path):
        found = _entities(path)
        return all(any(found.get(k) is None if v is None else k in found and _same(k, found[k], v) for v in values)
                   for k, values in filters.items())
    return [p for p in paths if keep(p)]


def _label(filters: dict[str, list], key: str, default: str) -> str:
    return " or ".join(str(v) for v in filters[key]) if key in filters else default


def _filter_record(filters: dict[str, list]) -> dict:
    return {"bids_filters": filters} if filters else {}


def _session_label(session: str | None) -> str | None:
    return f"ses-{session.removeprefix('ses-')}" if session else None


def _stem(connectivity: str | None) -> str:
    if connectivity is None:
        return _RELMAT
    from conngraph.connectivity import check_kind

    check_kind(connectivity)
    return _SERIES


def _read_xcpd_file(path: str, connectivity: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t") if connectivity else pd.read_csv(path, sep="\t", index_col=0)
    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _series_record(connectivity: str | None, shrinkage: bool) -> dict:
    return {} if connectivity is None else {"connectivity": connectivity, "shrinkage": shrinkage}


def _discover_subjects_flat(
    flat_dir: str,
    session: str,
    task: str,
    space: str,
    atlas: str,
    stem: str,
) -> list[str]:
    pattern = os.path.join(
        flat_dir,
        f"sub-*_{session}_task-{task}*_space-{space}_seg-{atlas}{stem}",
    )
    subs = []
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        # fname starts with "sub-XXX_"
        sub = fname.split("_")[0].removeprefix("sub-")
        subs.append(sub)
    return sorted(set(subs))


def _skip_reason(sub: str, matches: list[str]) -> str:
    if not matches:
        return f"sub-{sub}: no file found"
    runs = _runs_only(matches)
    if runs:
        return (f"sub-{sub}: {len(runs)} runs ({', '.join(_run_labels(runs))}); merge them upstream "
                "(XCP-D --combine-runs) or pick one with a BIDS filter on run")
    return f"sub-{sub}: {len(matches)} files matched, be more specific: " + ", ".join(matches)


def _run_labels(paths: list[str]) -> list[str]:
    return [f"run-{_entities(p)['run']}" for p in paths]


def _add_runs(matrices: dict, sub: str, runs: list[str], read) -> list[str]:
    """Each run as its own entry, named <subject>_run-<label>; returns the run labels."""
    labels = _run_labels(runs)
    for label, path in zip(labels, runs):
        matrices[f"{sub}_{label}"] = read(path)
    return labels


def _first_reasons(skipped: list[str], n: int = 3) -> str:
    return "".join(f"\n  {s}" for s in skipped[:n]) + (f"\n  ... and {len(skipped) - n} more" if len(skipped) > n else "")


def _runs_only(paths: list[str]) -> list[str]:
    """The files in run order when they differ in nothing but their run, else an empty list."""
    found = [_entities(p) for p in paths]
    if any("run" not in e for e in found):
        return []
    rest = {tuple(sorted((k, v) for k, v in e.items() if k != "run")) for e in found}
    if len(rest) != 1:
        return []
    order = lambda r: (not r.isdigit(), int(r) if r.isdigit() else 0, r)  # noqa: E731
    return [p for _, p in sorted(zip(found, paths), key=lambda pair: order(pair[0]["run"]))]


def _concat_runs(paths: list[str]) -> pd.DataFrame:
    """Each run's regional time series z-scored, then stacked in run order, as XCP-D's --combine-runs does."""
    runs = [_read_xcpd_file(p, "series") for p in paths]
    return pd.concat([(ts - ts.mean()) / ts.std(ddof=0) for ts in runs], ignore_index=True)


def _warn_skipped(skipped: list[str]) -> None:
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} subject(s) without a unique matching file:\n  " + "\n  ".join(skipped),
            stacklevel=3,
        )


def _discover_subjects(xcpd_dir: str) -> list[str]:
    dirs = glob.glob(os.path.join(xcpd_dir, "sub-*"))
    return sorted(
        os.path.basename(d).removeprefix("sub-")
        for d in dirs
        if os.path.isdir(d)
    )
