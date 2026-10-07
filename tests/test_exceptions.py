import urllib.request

import numpy as np
import pandas as pd
import pytest

import brainnet3d as bnv
from brainnet3d import datasets
from brainnet3d.exceptions import BrainNet3DError, DataValidationError, DownloadError
from brainnet3d.graph_theory import compute_graph_metrics, load_xcpd_flat

LABELS = list("abcd")


def _square():
    return pd.DataFrame(np.eye(4), index=LABELS, columns=LABELS)


def _nodes(**extra):
    return pd.DataFrame({"label": LABELS, "x": 0.0, "y": 0.0, "z": 0.0, **extra})


def _nan_matrix():
    arr = np.eye(4)
    arr[0, 1] = arr[1, 0] = np.nan
    return pd.DataFrame(arr, index=LABELS, columns=LABELS)


@pytest.mark.parametrize("make", [
    lambda: bnv.ConnectivityDataset({"s": _square()}, _nodes().drop(columns="z")),
    lambda: bnv.ConnectivityDataset({"s": _square().iloc[:3]}, _nodes()),
    lambda: bnv.ConnectivityDataset({"s": _square().set_axis(list("wxyz"), axis=0)}, _nodes()),
    lambda: compute_graph_metrics(
        {"s": _nan_matrix()}, pd.DataFrame({"label": LABELS, "network_label": list("aabb")}), level="node",
    ),
    lambda: bnv.BrainNetPlotter(bnv.load(_square(), _nodes(hemisphere="L"))).plot(show_hemisphere="R"),
], ids=["missing-columns", "not-square", "label-mismatch", "nan-input", "empty-hemisphere"])
def test_data_problems_raise_data_validation_error(make):
    with pytest.raises(DataValidationError) as info:
        make()
    assert isinstance(info.value, ValueError) and isinstance(info.value, BrainNet3DError)


def test_non_square_matrix_file_raises(tmp_path):
    path = tmp_path / "m.csv"
    _square().iloc[:3].to_csv(path)
    with pytest.raises(DataValidationError, match="not square"):
        bnv.load(str(path), _nodes())


@pytest.mark.filterwarnings("ignore:Skipped")
def test_xcpd_nothing_loaded_raises(tmp_path):
    atlas_path = tmp_path / "atlas.tsv"
    pd.DataFrame({"label": LABELS, "network_label": list("aabb")}).to_csv(atlas_path, sep="\t", index=False)
    with pytest.raises(DataValidationError, match="No matrices"):
        load_xcpd_flat(str(tmp_path), str(atlas_path), atlas="Gordon", subject_ids=["99"], verbose=False)


def test_failed_download_raises_download_error(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(DownloadError, match="offline") as info:
        datasets._fetch_gordon_parcels(tmp_path)
    assert isinstance(info.value, RuntimeError)
