"""
Loaders for BIDS derivatives that name their connectivity matrices *_relmat.tsv:
XCP-D and fnirs-pipe. Each returns matrices and a node table in the format
expected by compute_graph_metrics.
"""

from __future__ import annotations

import glob
import os
import warnings

import pandas as pd

from brainnet3d.exceptions import DataValidationError
from brainnet3d.loaders import _drop_bad_nodes, record_input, series_to_matrices

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
    >>> from brainnet3d.graph_theory import compute_graph_metrics
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

    for sub in subject_ids:
        matches = _filtered(_find(xcpd_dir, sub, session, "func", tail), filters)

        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue

        matrices[sub] = _read_xcpd_file(matches[0], connectivity)

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError(
            "No matrices could be loaded. Check xcpd_dir, atlas, session, task, space."
        )

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
                 session=session, skipped=skipped, **_series_record(connectivity, shrinkage), **_filter_record(filters))
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


def load_fnirs_pipe(
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
    Load one chromophore's channel-by-channel Pearson matrices from a fnirs-pipe derivatives folder.

    Parameters
    ----------
    deriv_dir : fnirs-pipe output folder holding ``sub-*/[ses-*/]nirs/``.
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

    matrices: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for sub in subject_ids:
        matches = _filtered(_find(deriv_dir, sub, session, "nirs", tail), filters)
        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue
        df = pd.read_csv(matches[0], sep="\t", index_col=0)
        names = [str(c).removesuffix(f" {chromophore}") for c in df.columns]
        matrices[sub] = df.set_axis(names, axis=0).set_axis(names, axis=1)

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError("No matrices could be loaded. Check deriv_dir, chromophore, session, task.")
    if verbose:
        print(f"[load_fnirs_pipe] Loaded {len(matrices)} {chromophore} matrix/matrices")

    before = matrices
    if bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold, drop_mode)
    nodes = pd.DataFrame({"label": list(next(iter(matrices.values())).columns)})
    record_input(nodes, before, matrices, bad_node_threshold, drop_mode, source="fnirs-pipe", path=deriv_dir,
                 chromophore=chromophore, task=_label(filters, "task", task), session=session, skipped=skipped,
                 **_filter_record(filters))
    return matrices, nodes


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
    from brainnet3d.connectivity import check_kind

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
    return f"sub-{sub}: {len(matches)} files matched, be more specific: " + ", ".join(matches)


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
