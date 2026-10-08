"""
Aggregate ROI-level connectivity matrices to network-level average correlations.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def build_net2rois(
    atlas_df: pd.DataFrame,
    label_col: str = "label",
    network_col: str = "network_label",
) -> dict:
    """Return {network_name: [roi_label, ...]} from atlas."""
    df = atlas_df.dropna(subset=[network_col])
    return {
        net: df.loc[df[network_col] == net, label_col].tolist()
        for net in sorted(df[network_col].unique())
    }


def build_net_hemi2rois(
    atlas_df: pd.DataFrame,
    label_col: str = "label",
    network_col: str = "network_label",
    hemi_col: str = "hemisphere",
) -> dict:
    """Return {hemi_network_name: [roi_label, ...]}; hemisphere comes from hemi_col if present, else the label prefix.

    Column values L/left/lh and R/right/rh (any case), or labels starting with "L_"/"R_",
    give L and R; everything else is grouped as bilateral (B_).
    """
    df = atlas_df.dropna(subset=[network_col])
    if hemi_col in df.columns:
        hemis = [_HEMI_NAMES.get(str(h).strip().lower(), "B") for h in df[hemi_col]]
    else:
        hemis = [_hemi_from_prefix(roi) for roi in df[label_col]]

    if df.shape[0] and not {"L", "R"} & set(hemis):
        warnings.warn(
            f"No ROI could be assigned to a hemisphere (no '{hemi_col}' column and no L_/R_ label prefix); "
            "all ROIs are grouped as bilateral (B_).",
            stacklevel=3,
        )

    mapping: dict = {}
    for roi, net, hemi in zip(df[label_col], df[network_col], hemis):
        mapping.setdefault(f"{hemi}_{net}", []).append(roi)
    return dict(sorted(mapping.items()))


_HEMI_NAMES = {"l": "L", "left": "L", "lh": "L", "r": "R", "right": "R", "rh": "R"}


def _hemi_from_prefix(roi) -> str:
    if isinstance(roi, str) and roi.startswith("L_"):
        return "L"
    if isinstance(roi, str) and roi.startswith("R_"):
        return "R"
    return "B"


def _fisher_z(corr_df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-7
    clipped = np.clip(corr_df.to_numpy(dtype=float, copy=True), -1 + eps, 1 - eps)
    z = 0.5 * np.log((1 + clipped) / (1 - clipped))
    np.fill_diagonal(z, 0)
    return pd.DataFrame(z, index=corr_df.index, columns=corr_df.columns)


def compute_net_corr(
    matrices: dict[str, pd.DataFrame],
    net2rois: dict,
    apply_fisher_z: bool = True,
) -> dict[str, pd.DataFrame]:
    """Average Fisher-z (or raw) correlations within and between networks.

    Parameters
    ----------
    matrices : {subject_id: N×N DataFrame with ROI labels as index/columns}
    net2rois : output of build_net2rois or build_net_hemi2rois
    apply_fisher_z : transform correlations before averaging (recommended)

    Returns
    -------
    {subject_id: (n_networks × n_networks) DataFrame of averaged z-values}
    """
    matrices_z = {
        sid: _fisher_z(mat) if apply_fisher_z else mat.copy()
        for sid, mat in matrices.items()
    }

    networks = list(net2rois.keys())
    result: dict[str, pd.DataFrame] = {}

    for sub_id, z_mat in matrices_z.items():
        roi_labels = z_mat.columns.tolist()
        z_vals = z_mat.values
        pairs: dict = {}

        for net1, rois1 in net2rois.items():
            idx1 = [roi_labels.index(r) for r in rois1 if r in roi_labels]
            for net2, rois2 in net2rois.items():
                idx2 = [roi_labels.index(r) for r in rois2 if r in roi_labels]

                if not idx1 or not idx2 or (net1 == net2 and len(idx1) < 2):
                    pairs[(net1, net2)] = np.nan
                elif net1 == net2:
                    sub = z_vals[np.ix_(idx1, idx1)]
                    triu = np.triu_indices_from(sub, k=1)
                    pairs[(net1, net2)] = np.nanmean(sub[triu])
                else:
                    pairs[(net1, net2)] = np.nanmean(z_vals[np.ix_(idx1, idx2)])

        df = pd.DataFrame(index=networks, columns=networks, dtype=float)
        for (n1, n2), val in pairs.items():
            df.loc[n1, n2] = val
        result[sub_id] = df

    return result
