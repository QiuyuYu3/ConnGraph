"""
Functional connectivity matrices from regional time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brainnet3d.exceptions import DataValidationError

KINDS = ("correlation", "partial correlation")


def compute_connectivity(
    timeseries: dict[str, pd.DataFrame | np.ndarray],
    kind: str = "correlation",
    shrinkage: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Connectivity matrix of each participant's time series (time points in rows, regions in columns).

    Parameters
    ----------
    timeseries : ``{subject_id: DataFrame or array}``; DataFrame columns name the regions.
    kind : "correlation" (Pearson) or "partial correlation".
    shrinkage : estimate the covariance with Ledoit-Wolf shrinkage instead of the sample covariance.

    Returns
    -------
    dict
        ``{subject_id: region × region DataFrame}``. Time points missing in every
        region are skipped; a region with missing or constant values gets an
        all-missing row and column.
    """
    check_kind(kind)
    return {sid: _connectivity(pd.DataFrame(ts), kind, shrinkage, sid) for sid, ts in timeseries.items()}


def check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def unusable_regions(ts: pd.DataFrame) -> list:
    """Regions with missing values at kept time points, or with no variance."""
    ts = ts.dropna(how="all")
    bad = ts.isna().any() | (ts.std() == 0)
    return ts.columns[bad.to_numpy()].tolist()


def _connectivity(ts: pd.DataFrame, kind: str, shrinkage: bool, sid: str) -> pd.DataFrame:
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import EmpiricalCovariance, LedoitWolf

    ts = ts.dropna(how="all")
    good = [c for c in ts.columns if c not in set(unusable_regions(ts))]
    n_time, n_good = len(ts), len(good)
    if n_time < 3:
        raise DataValidationError(f"{sid}: {n_time} time point(s) left; at least 3 are needed.")
    if kind == "partial correlation" and not shrinkage and n_time <= n_good:
        raise DataValidationError(
            f"{sid}: partial correlation needs more time points ({n_time}) than regions ({n_good}); "
            "use shrinkage=True."
        )
    estimator = LedoitWolf(store_precision=False) if shrinkage else EmpiricalCovariance()
    measure = ConnectivityMeasure(kind=kind, cov_estimator=estimator)
    values = measure.fit_transform([ts[good].to_numpy(dtype=float)])[0]
    out = pd.DataFrame(np.nan, index=ts.columns, columns=ts.columns)
    out.loc[good, good] = values
    return out
