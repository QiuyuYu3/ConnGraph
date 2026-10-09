"""
ConnectivityDataset: the single internal format every loader produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from conngraph.exceptions import DataValidationError


@dataclass
class ConnectivityDataset:
    """
    Central data container shared by all loaders, the graph-theory engine,
    and the visualisation layer.

    Parameters
    ----------
    matrices : dict[str, pd.DataFrame]
        {subject_id: N×N DataFrame with ROI labels as index AND columns}.
        Single-subject data should use subject_id = "single".
    nodes_df : pd.DataFrame
        Must contain a label column; x, y, z are needed for the brain figures.
        Optional columns: network, hemisphere, + any user-defined columns
        that can later be mapped to node_size / node_color.
    """

    matrices: dict[str, pd.DataFrame]
    nodes_df: pd.DataFrame

    def __post_init__(self):
        self._validate()

    def _validate(self):
        required_node_cols = {"label"}
        missing = required_node_cols - set(self.nodes_df.columns)
        if missing:
            raise DataValidationError(f"nodes_df is missing required columns: {missing}")

        for sub_id, mat in self.matrices.items():
            if mat.shape[0] != mat.shape[1]:
                raise DataValidationError(f"Matrix for '{sub_id}' is not square: {mat.shape}")
            if list(mat.index) != list(mat.columns):
                raise DataValidationError(
                    f"Matrix for '{sub_id}': row index and column names must match."
                )

    @property
    def subject_ids(self) -> list:
        return list(self.matrices.keys())

    @property
    def n_subjects(self) -> int:
        return len(self.matrices)

    @property
    def roi_labels(self) -> list:
        return self.nodes_df["label"].tolist()

    @property
    def n_rois(self) -> int:
        return len(self.nodes_df)

    @property
    def has_network(self) -> bool:
        return "network" in self.nodes_df.columns

    @property
    def has_hemisphere(self) -> bool:
        return "hemisphere" in self.nodes_df.columns

    def get_matrix(self, subject_id: str) -> np.ndarray:
        return self.matrices[subject_id].values.astype(float)

    def mean_matrix(self) -> pd.DataFrame:
        arrays = np.array([m.values.astype(float) for m in self.matrices.values()])
        mean   = np.nanmean(arrays, axis=0)
        labels = list(next(iter(self.matrices.values())).columns)
        return pd.DataFrame(mean, index=labels, columns=labels)

    def get_positions(self) -> np.ndarray:
        return self.nodes_df[["x", "y", "z"]].values.astype(float)

    def summary(self):
        print("=" * 45)
        print("ConnectivityDataset")
        print("=" * 45)
        print(f"  Subjects  : {self.n_subjects}")
        print(f"  ROIs      : {self.n_rois}")
        print(f"  Network   : {'yes' if self.has_network else 'no'}")
        print(f"  Hemisphere: {'yes' if self.has_hemisphere else 'no'}")
        extra = [
            c for c in self.nodes_df.columns
            if c not in {"label", "x", "y", "z", "network", "hemisphere"}
        ]
        if extra:
            print(f"  Extra cols: {extra}")
        print("=" * 45)
