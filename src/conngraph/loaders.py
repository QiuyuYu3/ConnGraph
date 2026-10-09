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
import pathlib
import warnings

import numpy as np
import pandas as pd

from conngraph.core.dataset import ConnectivityDataset
from conngraph.exceptions import DataValidationError

# pandas' default missing-value strings minus "None", which atlases such as Gordon use as a network label
_NODE_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null",
]
# Key in a node or atlas table's attrs where loaders note what they read; compute_graph_metrics copies it to params
INPUT_ATTR = "conngraph_input"


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
        .npy, .mat or an array) whose rows follow the order of `nodes`, a
        CIFTI .pconn.nii or an AFNI 3dNetCorr .netcc, which carry their own region names.
    nodes : str | pd.DataFrame
        Path to a CSV/TSV with a label column; x, y, z are needed for the brain figures.
    subject_id : label for this subject inside the dataset.
    bad_node_threshold : drop ROIs where NaN fraction exceeds this (0–1).
    values : "r" for correlations, "z" for Fisher z values (converted back to r);
        for a .netcc file, "r" reads its CC matrix and "z" its FZ matrix.
    mat_key : variable to read from a .mat file holding more than one square matrix,
        or the .netcc matrix to read (CC, FZ, PC or PCB).

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
    subject_ids: list[str] | None = None,
) -> ConnectivityDataset:
    """
    Load a multi-subject connectivity dataset.

    Parameters
    ----------
    matrices : str | dict | np.ndarray | list
        Directory path → scans for files matching `pattern`.
        Subject IDs are extracted from the file name, without its extension,
        before the first underscore (e.g. ``sub-001_matrix.csv`` → ``sub-001``),
        or, when `pattern` has a folder part (``*/corr_000.netcc``), are the folder names.
        Dict → ``{subject_id: path_DataFrame_or_array}``.
        A subjects × nodes × nodes array or a list of arrays, as nilearn returns them, is named by `subject_ids`.
        Each matrix may be in any form :func:`load` accepts.
    nodes : str | pd.DataFrame
        Same as :func:`load`.
    pattern : glob pattern used when `matrices` is a directory.
    bad_node_threshold : nodes with NaN fraction > threshold are dropped.
    drop_mode : "union" removes a node if bad in ANY subject.
                "intersection" removes only nodes bad in ALL subjects.
    values : same as :func:`load`.
    mat_key : same as :func:`load`.
    subject_ids : names for an array or list input; defaults to sub-01, sub-02, ...

    Returns
    -------
    ConnectivityDataset
    """
    _check_values(values)
    nodes_df = _read_nodes(nodes)
    labels   = nodes_df["label"].tolist()
    sources  = _sources(matrices, pattern, subject_ids, "matrices")
    raw = {sid: _read_matrix(src, labels, mat_key, values) for sid, src in sources.items()}

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
    subject_ids: list[str] | None = None,
) -> ConnectivityDataset:
    """
    Load regional time series and turn them into connectivity matrices.

    Parameters
    ----------
    timeseries : str | dict | np.ndarray | list
        Directory path, ``{subject_id: source}``, a subjects × time × regions
        array or a list of arrays, as in :func:`load_group`.
        Each source holds time points in rows and regions in columns: a CSV/TSV
        whose header names the regions, an unlabelled table (.csv, .tsv, .txt,
        .1D, .npy, .mat or an array) whose columns follow the order of `nodes`,
        a CIFTI .ptseries.nii, which carries its own parcel names, or an AFNI
        3dNetCorr .netts file (one region per row, in the order of `nodes`).
    nodes : str | pd.DataFrame
        Same as :func:`load`.
    pattern : glob pattern used when `timeseries` is a directory.
    kind, shrinkage : see :func:`conngraph.compute_connectivity`.
    bad_node_threshold : below 1, regions with missing or constant time series are dropped.
    drop_mode : "union" drops a region unusable in ANY subject, "intersection" only one unusable in ALL.
    mat_key : variable to read from a .mat file holding more than one 2-D array.
    subject_ids : names for an array or list input; defaults to sub-01, sub-02, ...

    Returns
    -------
    ConnectivityDataset
    """
    from conngraph.connectivity import check_kind

    check_kind(kind)
    nodes_df = _read_nodes(nodes)
    labels   = nodes_df["label"].tolist()
    sources  = _sources(timeseries, pattern, subject_ids, "timeseries")
    series ={sid: _read_timeseries(src, labels, mat_key) for sid, src in sources.items()}
    matrices = series_to_matrices(series, kind, shrinkage, bad_node_threshold, drop_mode)

    nodes_df = _align_nodes(nodes_df, next(iter(matrices.values())))
    record_input(nodes_df, series, matrices, bad_node_threshold, drop_mode,
                 source="time series", connectivity=kind, shrinkage=shrinkage)
    return ConnectivityDataset(matrices=matrices, nodes_df=nodes_df)


def series_to_matrices(
    series: dict[str, pd.DataFrame], kind: str, shrinkage: bool, threshold: float, mode: str,
) -> dict[str, pd.DataFrame]:
    """Connectivity of each time series after dropping, for everyone, the regions the drop rule removes."""
    from conngraph.connectivity import compute_connectivity, unusable_regions

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
    if src.endswith(".netts"):
        return _read_netts(src, labels)

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


def _read_netts(path: str, labels: list[str]) -> pd.DataFrame:
    """AFNI 3dNetCorr time series: one region per row, led by its ROI value when written with -ts_label."""
    arr = np.loadtxt(path, ndmin=2)
    with open(path, encoding="utf-8") as f:
        first = f.readline().split()[0]
    if "." not in first:
        arr = arr[:, 1:]
    return _labelled_series(arr.T, labels, path)


def _read_netcc(path: str, block: str) -> pd.DataFrame:
    """One matrix block (CC, FZ, PC or PCB) of an AFNI 3dNetCorr .netcc file, named by its ROI labels."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    names, blocks, current, i = None, {}, None, 0
    while i < len(lines):
        line = lines[i]
        head = line.lstrip("#").strip()
        if line.startswith("# WITH_ROI_LABELS"):
            names = [s.strip() for s in lines[i + 1].split("\t")]
            i += 3
            continue
        if line.startswith("#"):
            current = head if head.isalpha() else None
            if current:
                blocks[current] = []
        elif current is not None:
            blocks[current].append([float(x) for x in line.split()])
        elif names is None:
            names = line.split()
        i += 1
    if block not in blocks:
        raise DataValidationError(f"No matrix '{block}' in '{path}'; it holds {list(blocks)}.")
    return pd.DataFrame(np.array(blocks[block]), index=names, columns=names)


def _labelled_series(arr: np.ndarray, labels: list[str], src: str) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(labels):
        raise DataValidationError(
            f"Time series in '{src}' has shape {arr.shape}, but the node table has {len(labels)} labels; "
            "an unlabelled time series needs time points in rows and one column per node table row."
        )
    return pd.DataFrame(arr, columns=labels)


_EXTENSIONS = (".pconn.nii", ".ptseries.nii", ".netcc", ".netts", ".npy", ".mat", ".csv", ".tsv", ".txt", ".1D")


def subject_id_from_path(path: str, root: str | None = None) -> str:
    """The folder name for files in per-subject folders under root, else the file name before its first underscore."""
    if root is not None:
        parts = pathlib.Path(os.path.relpath(path, root)).parts
        if len(parts) > 1:
            return parts[0]
    name = os.path.basename(path)
    for ext in _EXTENSIONS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.split("_")[0]


def subject_files(root: str, pattern: str) -> dict[str, str]:
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {root}")
    files: dict[str, str] = {}
    for p in paths:
        sid = subject_id_from_path(p, root)
        if sid in files:
            raise DataValidationError(
                f"Two files give subject ID '{sid}': {files[sid]} and {p}; narrow the pattern."
            )
        files[sid] = p
    return files


def _sources(given, pattern: str, subject_ids: list[str] | None, name: str) -> dict:
    if isinstance(given, str) and os.path.isdir(given):
        return subject_files(given, pattern)
    if isinstance(given, dict):
        return given
    if isinstance(given, (np.ndarray, list, tuple)):
        items = list(given)
        ids = subject_ids if subject_ids is not None else [f"sub-{i:02d}" for i in range(1, len(items) + 1)]
        if len(ids) != len(items):
            raise ValueError(f"subject_ids has {len(ids)} names for {len(items)} subjects")
        return dict(zip(ids, items))
    raise TypeError(f"`{name}` must be a directory path, a dict, an array or a list of arrays.")


def _check_values(values: str) -> None:
    if values not in ("r", "z"):
        raise ValueError(f"values must be 'r' or 'z', got {values!r}")


def _read_matrix(
    src: str | pd.DataFrame | np.ndarray,
    labels: list[str] | None = None,
    mat_key: str | None = None,
    values: str = "r",
) -> pd.DataFrame:
    df = _read_square(src, labels, mat_key, values)
    if df.shape[0] != df.shape[1]:
        raise DataValidationError(
            f"Matrix in '{src}' is not square: {df.shape}. "
            "Make sure the first column is used as the row index."
        )
    return np.tanh(df) if values == "z" else df


def _read_square(src, labels: list[str] | None, mat_key: str | None, values: str = "r") -> pd.DataFrame:
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
    if src.endswith(".netcc"):
        return _read_netcc(src, mat_key or ("FZ" if values == "z" else "CC"))

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
    data_dir : cache folder; defaults to $CONNGRAPH_DATA or ~/conngraph_data.

    Returns
    -------
    pd.DataFrame
        Columns: label, hemisphere, x, y, z, network, network_perino2021,
                 surface_area_mm2, parcel_id.
        ``network`` always reflects the chosen ``community`` scheme so that
        downstream visualisation code can use it without extra wiring.
    """
    from conngraph.datasets import gordon_nodes

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
