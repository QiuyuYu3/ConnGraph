"""
XCP-D loader: read correlation matrices directly from an XCP-D derivatives
directory and return them in the format expected by compute_graph_metrics.
"""

from __future__ import annotations

import glob
import os
import warnings

import pandas as pd

from brainnet3d.exceptions import DataValidationError
from brainnet3d.loaders import _drop_bad_nodes


def load_xcpd(
    xcpd_dir: str,
    atlas: str,
    session: str = "ses-01",
    task: str = "rest",
    space: str = "fsLR",
    subject_ids: list[str] | None = None,
    bad_node_threshold: float = 0.9,
    verbose: bool = True,
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
    session : str
        BIDS session label, e.g. ``"ses-01"``.
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
    pattern_template = os.path.join(
        xcpd_dir,
        "sub-{sub}",
        session,
        "func",
        f"sub-{{sub}}_{session}_task-{task}*_space-{space}_seg-{atlas}"
        f"_stat-pearsoncorrelation_relmat.tsv",
    )

    for sub in subject_ids:
        pattern = pattern_template.format(sub=sub)
        matches = glob.glob(pattern)

        if len(matches) != 1:
            skipped.append(_skip_reason(sub, matches))
            continue

        df = pd.read_csv(matches[0], sep="\t", index_col=0)
        df.index   = df.index.astype(str)
        df.columns = df.columns.astype(str)
        matrices[sub] = df

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError(
            "No matrices could be loaded. Check xcpd_dir, atlas, session, task, space."
        )

    if verbose:
        print(f"[load_xcpd] Loaded {len(matrices)} matrix/matrices")

    # Drop bad nodes (union across subjects)
    if bad_node_threshold < 1.0:
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

    stem = "_stat-pearsoncorrelation_relmat.tsv"
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

        df = pd.read_csv(matches[0], sep="\t", index_col=0)
        df.index   = df.index.astype(str)
        df.columns = df.columns.astype(str)
        matrices[sub] = df

    _warn_skipped(skipped)
    if not matrices:
        raise DataValidationError("No matrices could be loaded. Check flat_dir and parameters.")

    if verbose:
        print(f"[load_xcpd_flat] Loaded {len(matrices)} matrix/matrices")

    if bad_node_threshold < 1.0:
        matrices = _drop_bad_nodes(matrices, bad_node_threshold)

    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    atlas_df = pd.read_csv(atlas_path, sep="\t")
    if verbose:
        print(f"[load_xcpd_flat] Atlas: {atlas_path} ({len(atlas_df)} ROIs)")

    return matrices, atlas_df


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
