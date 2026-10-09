"""
Loaders: read user files → ConnectivityDataset.

Entry points:
  - load()               : single subject (one matrix file)
  - load_group()         : multiple subjects (dict or directory of matrix files)
  - load_timeseries()    : multiple subjects' regional time series, turned into matrices
  - load_gordon_atlas()  : Gordon 333-parcel node table, fetched on first use
"""

from __future__ import annotations

import os
import glob
import warnings

import numpy as np
import pandas as pd

from brainnet3d.core.dataset import ConnectivityDataset
from brainnet3d.exceptions import DataValidationError

# pandas' default missing-value strings minus "None", which atlases such as Gordon use as a network label
_NODE_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null",
]
# Key in a node or atlas table's attrs where loaders note what they read; compute_graph_metrics copies it to params
INPUT_ATTR = "brainnet3d_input"


def record_input(table: pd.DataFrame, before: dict, after: dict, threshold: float, mode: str, **fields) -> pd.DataFrame:
    """Note on the table how many matrices were read and which nodes the missing-value rule dropped."""
    labels = lambda mats: set().union(*(m.columns for m in mats.values())) if mats else set()
    table.attrs[INPUT_ATTR] = {
        **fields, "n_loaded": len(after), "bad_node_threshold": threshold, "drop_mode": mode,
        "dropped": sorted(labels(before) - labels(after), key=str),
    }
    return table


def load(
    matrix: str | pd.DataFrame | np.ndarray,
    nodes:  str | pd.DataFrame,
    subject_id: str = "single",
    bad_node_threshold: float = 1.0,
    values: str = "r",
    mat_key: str | None = None,
) -> ConnectivityDataset:
    """
    Load a single-subject connectivity dataset.

    Parameters
    ----------
    matrix : str | pd.DataFrame | np.ndarray
        A labelled square CSV/TSV (ROI labels as the first column and as
        column headers), an unlabelled square matrix (.csv, .tsv, .txt, .1D,
        .npy, .mat or an array) whose rows follow the order of `nodes`, or a
        CIFTI .pconn.nii, which carries its own parcel names.
    nodes : str | pd.DataFrame
        Path to a CSV/TSV with at minimum: label, x, y, z.
    subject_id : label for this subject inside the dataset.
    bad_node_threshold : drop ROIs where NaN fraction exceeds this (0–1).
    values : "r" for correlations, "z" for Fisher z values (converted back to r).
    mat_key : variable to read from a .mat file holding more than one square matrix.

    Returns
    -------
    ConnectivityDataset
    """
    _check_values(values)
    nodes_df = _read_nodes(nodes)
    mat_df   = _read_matrix(matrix, nodes_df["label"].tolist(), mat_key, values)

    raw      = {subject_id: mat_df}
    kept     = _drop_bad_nodes(raw, bad_node_threshold)
    mat_df   = kept[subject_id]
    nodes_df = _align_nodes(nodes_df, mat_df)
    record_input(nodes_df, raw, kept, bad_node_threshold, "union", source="matrix files", values=values)

    return ConnectivityDataset(matrices={subject_id: mat_df}, nodes_df=nodes_df)


def load_group(
    matrices: str | dict[str, str | pd.DataFrame | np.ndarray],
    nodes:    str | pd.DataFrame,
    pattern:  str = "*_matrix.csv",
    bad_node_threshold: float = 0.9,
    drop_mode: str = "union",
    values: str = "r",
    mat_key: str | None = None,
) -> ConnectivityDataset:
    """
    Load a multi-subject connectivity dataset.

    Parameters
    ----------
    matrices : str | dict
        Directory path → scans for files matching `pattern`.
        Subject IDs are extracted from the file name, without its extension,
        before the first underscore (e.g. ``sub-001_matrix.csv`` → ``sub-001``).
        Dict → ``{subject_id: path_DataFrame_or_array}``.
        Each matrix may be in any form :func:`load` accepts.
    nodes : str | pd.DataFrame
        Same as :func:`load`.
    pattern : glob pattern used when `matrices` is a directory.
    bad_node_threshold : nodes with NaN fraction > threshold are dropped.
    drop_mode : "union" removes a node if bad in ANY subject.
                "intersection" removes only nodes bad in ALL subjects.
    values : same as :func:`load`.
    mat_key : same as :func:`load`.

    Returns
    -------
    ConnectivityDataset
    """
    _check_values(values)
    nodes_df = _read_nodes(nodes)
    labels   = nodes_df["label"].tolist()
    raw: dict[str, pd.DataFrame] = {}

    if isinstance(matrices, str) and os.path.isdir(matrices):
        paths = sorted(glob.glob(os.path.join(matrices, pattern)))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {matrices}"
            )
        for p in paths:
            raw[subject_id_from_path(p)] = _read_matrix(p, labels, mat_key, values)
    elif isinstance(matrices, dict):
        for sub_id, src in matrices.items():
            raw[sub_id] = _read_matrix(src, labels, mat_key, values)
    else:
        raise TypeError("`matrices` must be a directory path (str) or a dict.")

    before = raw
    raw = _drop_bad_nodes(raw, bad_node_threshold, drop_mode)

    ref_mat  = next(iter(raw.values()))
    nodes_df = _align_nodes(nodes_df, ref_mat)
    record_input(nodes_df, before, raw, bad_node_threshold, drop_mode, source="matrix files", values=values)

    return ConnectivityDataset(matrices=raw, nodes_df=nodes_df)


def load_timeseries(
    timeseries: str | dict[str, str | pd.DataFrame | np.ndarray],
    nodes: str | pd.DataFrame,
    pattern: str = "*_timeseries.csv",
    kind: str = "correlation",
    shrinkage: bool = False,
    bad_node_threshold: float = 0.9,
    drop_mode: str = "union",
    mat_key: str | None = None,
) -> ConnectivityDataset:
    """
    Load regional time series and turn them into connectivity matrices.

    Parameters
    ----------
    timeseries : str | dict
        Directory path or ``{subject_id: source}``, as in :func:`load_group`.
        Each source holds time points in rows and regions in columns: a CSV/TSV
        whose header names the regions, an unlabelled table (.csv, .tsv, .txt,
        .1D, .npy, .mat or an array) whose columns follow the order of `nodes`,
        or a CIFTI .ptseries.nii, which carries its own parcel names.
    nodes : str | pd.DataFrame
        Same as :func:`load`.
    pattern : glob pattern used when `timeseries` is a directory.
    kind, shrinkage : see :func:`brainnet3d.compute_connectivity`.
    bad_node_threshold : below 1, regions with missing or constant time series are dropped.
    drop_mode : "union" drops a region unusable in ANY subject, "intersection" only one unusable in ALL.
    mat_key : variable to read from a .mat file holding more than one 2-D array.

    Returns
    -------
    ConnectivityDataset
    """
    from brainnet3d.connectivity import check_kind

    check_kind(kind)
    nodes_df = _read_nodes(nodes)
    labels   = nodes_df["label"].tolist()
    if isinstance(timeseries, str) and os.path.isdir(timeseries):
        paths = sorted(glob.glob(os.path.join(timeseries, pattern)))
        if not paths:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {timeseries}")
        sources = {subject_id_from_path(p): p for p in paths}
    elif isinstance(timeseries, dict):
        sources = timeseries
    else:
        raise TypeError("`timeseries` must be a directory path (str) or a dict.")
    series = {sid: _read_timeseries(src, labels, mat_key) for sid, src in sources.items()}
    matrices = series_to_matrices(series, kind, shrinkage, bad_node_threshold, drop_mode)

    nodes_df = _align_nodes(nodes_df, next(iter(matrices.values())))
    record_input(nodes_df, series, matrices, bad_node_threshold, drop_mode,
                 source="time series", connectivity=kind, shrinkage=shrinkage)
    return ConnectivityDataset(matrices=matrices, nodes_df=nodes_df)


def series_to_matrices(
    series: dict[str, pd.DataFrame], kind: str, shrinkage: bool, threshold: float, mode: str,
) -> dict[str, pd.DataFrame]:
    """Connectivity of each time series after dropping, for everyone, the regions the drop rule removes."""
    from brainnet3d.connectivity import compute_connectivity, unusable_regions

    unusable = [set(unusable_regions(ts)) for ts in series.values()]
    bad = set()
    if threshold < 1 and unusable:
        bad = set.union(*unusable) if mode == "union" else set.intersection(*unusable)
    if bad:
        warnings.warn(f"Dropping {len(bad)} node(s) with missing or constant time series: {sorted(bad)}", stacklevel=3)
    kept = {sid: ts.drop(columns=list(bad), errors="ignore") for sid, ts in series.items()}
    return compute_connectivity(kept, kind, shrinkage)


def _read_timeseries(src, labels: list[str], mat_key: str | None) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.set_axis(src.columns.astype(str), axis=1)
    if isinstance(src, np.ndarray):
        return _labelled_series(src, labels, "array")
    if src.endswith(".npy"):
        return _labelled_series(np.load(src), labels, src)
    if src.endswith(".mat"):
        return _labelled_series(_read_mat(src, mat_key, square=False), labels, src)
    if src.endswith(".ptseries.nii"):
        import nibabel as nib

        img = nib.load(src)
        return pd.DataFrame(np.asarray(img.get_fdata()), columns=[str(n) for n in img.header.get_axis(1).name])

    delimiter = {".csv": ",", ".tsv": "\t"}.get(os.path.splitext(src)[1])
    if _first_row_is_header(src, delimiter, labels):
        df = pd.read_csv(src, sep=delimiter or r"\s+", comment="#")
        return df.set_axis(df.columns.astype(str), axis=1)
    return _labelled_series(np.loadtxt(src, delimiter=delimiter, ndmin=2), labels, src)


def _first_row_is_header(path: str, delimiter: str | None, labels: list[str]) -> bool:
    with open(path, encoding="utf-8") as f:
        line = next((ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")), "")
    tokens = [t.strip().strip('"') for t in line.split(delimiter)]
    try:
        [float(t) for t in tokens]
    except ValueError:
        return True
    return set(tokens) <= set(labels) and len(set(tokens)) == len(tokens)


def _labelled_series(arr: np.ndarray, labels: list[str], src: str) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(labels):
        raise DataValidationError(
            f"Time series in '{src}' has shape {arr.shape}, but the node table has {len(labels)} labels; "
            "an unlabelled time series needs time points in rows and one column per node table row."
        )
    return pd.DataFrame(arr, columns=labels)


_EXTENSIONS =(".pconn.nii", ".ptseries.nii", ".npy", ".mat", ".csv", ".tsv", ".txt", ".1D")


def subject_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    for ext in _EXTENSIONS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.split("_")[0]


def _check_values(values: str) -> None:
    if values not in ("r", "z"):
        raise ValueError(f"values must be 'r' or 'z', got {values!r}")


def _read_matrix(
    src: str | pd.DataFrame | np.ndarray,
    labels: list[str] | None = None,
    mat_key: str | None = None,
    values: str = "r",
) -> pd.DataFrame:
    df = _read_square(src, labels, mat_key)
    if df.shape[0] != df.shape[1]:
        raise DataValidationError(
            f"Matrix in '{src}' is not square: {df.shape}. "
            "Make sure the first column is used as the row index."
        )
    return np.tanh(df) if values == "z" else df


def _read_square(src, labels: list[str] | None, mat_key: str | None) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    if isinstance(src, np.ndarray):
        return _labelled(src, labels, "array")
    if src.endswith(".npy"):
        return _labelled(np.load(src), labels, src)
    if src.endswith(".mat"):
        return _labelled(_read_mat(src, mat_key), labels, src)
    if src.endswith(".pconn.nii"):
        return _read_pconn(src)

    headerless = _read_headerless(src)
    if headerless is not None and labels is not None and headerless.shape == (len(labels), len(labels)):
        return _labelled(headerless, labels, src)

    sep = "\t" if src.endswith((".tsv", ".txt")) else ","
    df  = pd.read_csv(src, sep=sep, index_col=0)
    df.index   = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _read_headerless(path: str) -> np.ndarray | None:
    """The file as a plain numeric array, or None when it has a header row or label column."""
    delimiter = {".csv": ",", ".tsv": "\t"}.get(os.path.splitext(path)[1])
    try:
        return np.loadtxt(path, delimiter=delimiter, ndmin=2)
    except ValueError:
        return None


def _labelled(arr: np.ndarray, labels: list[str] | None, src: str) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise DataValidationError(f"Matrix in '{src}' is not square: {arr.shape}.")
    if labels is None or len(labels) != arr.shape[0]:
        raise DataValidationError(
            f"Matrix in '{src}' has {arr.shape[0]} rows but the node table has "
            f"{0 if labels is None else len(labels)} labels; an unlabelled matrix must follow the node table order."
        )
    return pd.DataFrame(arr, index=labels, columns=labels)


def _read_mat(path: str, key: str | None, square: bool = True) -> np.ndarray:
    from scipy.io import loadmat

    try:
        data = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError as e:
        raise DataValidationError(f"Cannot read '{path}': MATLAB v7.3 files are not supported; save with -v7.") from e
    if key is not None:
        if key not in data:
            raise DataValidationError(f"No variable '{key}' in '{path}'; it holds {sorted(data)}.")
        return data[key]
    found = sorted(k for k, v in data.items()
                   if isinstance(v, np.ndarray) and v.ndim == 2 and min(v.shape) > 1
                   and (v.shape[0] == v.shape[1] or not square))
    if len(found) != 1:
        kind = "square matrices" if square else "2-D arrays"
        raise DataValidationError(
            f"'{path}' holds {len(found)} {kind}; choose one with mat_key (variables: {sorted(data)})."
        )
    return data[found[0]]


def _read_pconn(path: str) -> pd.DataFrame:
    import nibabel as nib

    img = nib.load(path)
    names = [str(n) for n in img.header.get_axis(0).name]
    cols  = [str(n) for n in img.header.get_axis(1).name]
    return pd.DataFrame(np.asarray(img.get_fdata()), index=names, columns=cols)


def _read_nodes(src: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()

    sep = "\t" if src.endswith((".tsv", ".txt")) else ","
    df  = pd.read_csv(src, sep=sep, keep_default_na=False, na_values=_NODE_NA_VALUES)
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
