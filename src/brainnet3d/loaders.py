"""
Loaders: read user files → ConnectivityDataset.

Entry points:
  - load()               : single subject (one matrix file)
  - load_group()         : multiple subjects (dict or directory of matrix files)
  - load_gordon_atlas()  : Gordon 333-parcel node table, fetched on first use
"""

from __future__ import annotations

import os
import glob
import warnings

import pandas as pd

from brainnet3d.core.dataset import ConnectivityDataset


def load(
    matrix: str | pd.DataFrame,
    nodes:  str | pd.DataFrame,
    subject_id: str = "single",
    bad_node_threshold: float = 1.0,
) -> ConnectivityDataset:
    """
    Load a single-subject connectivity dataset.

    Parameters
    ----------
    matrix : str | pd.DataFrame
        Path to a square CSV/TSV with ROI labels as the first column (index)
        and matching column headers.
    nodes : str | pd.DataFrame
        Path to a CSV/TSV with at minimum: label, x, y, z.
    subject_id : label for this subject inside the dataset.
    bad_node_threshold : drop ROIs where NaN fraction exceeds this (0–1).

    Returns
    -------
    ConnectivityDataset
    """
    mat_df   = _read_matrix(matrix)
    nodes_df = _read_nodes(nodes)

    mat_df   = _drop_bad_nodes({subject_id: mat_df}, bad_node_threshold)[subject_id]
    nodes_df = _align_nodes(nodes_df, mat_df)

    return ConnectivityDataset(matrices={subject_id: mat_df}, nodes_df=nodes_df)


def load_group(
    matrices: str | dict[str, str | pd.DataFrame],
    nodes:    str | pd.DataFrame,
    pattern:  str = "*_matrix.csv",
    bad_node_threshold: float = 0.9,
    drop_mode: str = "union",
) -> ConnectivityDataset:
    """
    Load a multi-subject connectivity dataset.

    Parameters
    ----------
    matrices : str | dict
        Directory path → scans for files matching `pattern`.
        Subject IDs are extracted from the filename stem before the first
        underscore (e.g. ``sub-001_matrix.csv`` → ``sub-001``).
        Dict → ``{subject_id: path_or_DataFrame}``.
    nodes : str | pd.DataFrame
        Same as :func:`load`.
    pattern : glob pattern used when `matrices` is a directory.
    bad_node_threshold : nodes with NaN fraction > threshold are dropped.
    drop_mode : "union" removes a node if bad in ANY subject.
                "intersection" removes only nodes bad in ALL subjects.

    Returns
    -------
    ConnectivityDataset
    """
    raw: dict[str, pd.DataFrame] = {}

    if isinstance(matrices, str) and os.path.isdir(matrices):
        paths = sorted(glob.glob(os.path.join(matrices, pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {matrices}"
            )
        for p in paths:
            stem   = os.path.basename(p)
            sub_id = stem.split("_")[0]
            raw[sub_id] = _read_matrix(p)
    elif isinstance(matrices, dict):
        for sub_id, src in matrices.items():
            raw[sub_id] = _read_matrix(src)
    else:
        raise TypeError("`matrices` must be a directory path (str) or a dict.")

    raw = _drop_bad_nodes(raw, bad_node_threshold, drop_mode)

    nodes_df = _read_nodes(nodes)
    ref_mat  = next(iter(raw.values()))
    nodes_df = _align_nodes(nodes_df, ref_mat)

    return ConnectivityDataset(matrices=raw, nodes_df=nodes_df)


def _read_matrix(src: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()

    sep = "\t" if src.endswith((".tsv", ".txt")) else ","
    df  = pd.read_csv(src, sep=sep, index_col=0)
    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Matrix in '{src}' is not square: {df.shape}. "
            "Make sure the first column is used as the row index."
        )
    return df


def _read_nodes(src: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()

    sep = "\t" if src.endswith((".tsv", ".txt")) else ","
    df  = pd.read_csv(src, sep=sep)
    df["label"] = df["label"].astype(str)
    return df


def _drop_bad_nodes(
    matrices: dict[str, pd.DataFrame],
    threshold: float,
    mode: str = "union",
) -> dict[str, pd.DataFrame]:
    bad = _detect_bad_nodes(matrices, threshold, mode)
    if not bad:
        return matrices

    warnings.warn(
        f"Dropping {len(bad)} node(s) with NaN fraction above {threshold}: {sorted(bad)}",
        stacklevel=3,
    )
    return {
        sid: mat.drop(index=bad, columns=bad, errors="ignore")
        for sid, mat in matrices.items()
    }


def _detect_bad_nodes(
    raw: dict[str, pd.DataFrame],
    threshold: float,
    mode: str,
) -> set:
    sets = []
    for mat in raw.values():
        nan_row = mat.isna().mean(axis=1)
        nan_col = mat.isna().mean(axis=0)
        bad = set(mat.index[nan_row > threshold]) | \
              set(mat.columns[nan_col > threshold])
        sets.append(bad)

    if not sets:
        return set()

    return set.union(*sets) if mode == "union" else set.intersection(*sets)


def load_gordon_atlas(
    community: str = "gordon",
    drop_none: bool = False,
    parcels_xlsx: str | None = None,
    data_dir: str | None = None,
) -> pd.DataFrame:
    """
    Return a nodes DataFrame for the Gordon 333-parcel atlas, downloaded from the official release on first use.

    Parameters
    ----------
    community : "gordon" → use the original Gordon Community labels (``network``).
                "perino2021" → use Community_Perino2021 labels (``network_perino2021``).
    drop_none : if True, drop parcels whose network label is "None".
    parcels_xlsx : local Parcels.xlsx from the Gordon release; skips the download.
    data_dir : cache folder; defaults to $BRAINNET3D_DATA or ~/brainnet3d_data.

    Returns
    -------
    pd.DataFrame
        Columns: label, hemisphere, x, y, z, network, network_perino2021,
                 surface_area_mm2, parcel_id.
        ``network`` always reflects the chosen ``community`` scheme so that
        downstream visualisation code can use it without extra wiring.
    """
    from brainnet3d.datasets import gordon_nodes

    df = gordon_nodes(parcels_xlsx, data_dir)

    if community == "perino2021":
        df["network"] = df["network_perino2021"]
    elif community != "gordon":
        raise ValueError(f"community must be 'gordon' or 'perino2021', got '{community}'")

    if drop_none:
        df = df[df["network"] != "None"].reset_index(drop=True)

    return df


def _align_nodes(nodes_df: pd.DataFrame, mat_df: pd.DataFrame) -> pd.DataFrame:
    roi_labels = list(mat_df.columns)
    known      = set(nodes_df["label"].tolist())
    missing    = [r for r in roi_labels if r not in known]

    if missing:
        warnings.warn(
            f"{len(missing)} ROI(s) in matrix not found in nodes file "
            f"(will be excluded from visualisation): {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
            stacklevel=3,
        )
        roi_labels = [r for r in roi_labels if r in known]

    return nodes_df.set_index("label").loc[roi_labels].reset_index()
