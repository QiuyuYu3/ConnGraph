import numpy as np
import pandas as pd
import pytest

import conngraph as bnv
from conngraph import datasets

pytest.importorskip("openpyxl")


@pytest.fixture
def parcels_xlsx(tmp_path):
    rows = [
        (1, "L", "Default",         "Default",          "-10 50 20", 100.0),
        (2, "L", "CinguloParietal", "CinguloParietal",  "-5 -40 30", 80.5),
        (3, "L", "None",            "None",             "-30 0 -30", 60.0),
        (4, "L", "Default",         "Default",          "-12 40 10", 90.0),
        (5, "R", "Default",         "Default",          "10 50 20",  110.0),
        (6, "R", "None",            "MedVisual",        "30 0 -30",  70.0),
    ]
    df = pd.DataFrame(rows, columns=["ParcelID", "Hem", "Community", "Community_Perino2021", "Centroid (MNI)", "Surface area (mm2)"])
    path = tmp_path / "Parcels.xlsx"
    df.sample(frac=1, random_state=0).to_excel(path, index=False)
    return path


def test_build_gordon_nodes_uses_xcpd_labels(parcels_xlsx):
    nodes = datasets.gordon_nodes(parcels_xlsx)
    assert nodes["parcel_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert nodes["label"].tolist() == [
        "L_Default_1", "L_MedialParietal_1", "L_None_1", "L_Default_2", "R_Default_3", "R_None_2",
    ]
    assert nodes["network_perino2021"].tolist()[1] == "CinguloParietal"
    assert nodes.loc[4, ["x", "y", "z"]].tolist() == [10.0, 50.0, 20.0]
    assert nodes.loc[1, "surface_area_mm2"] == 80.5


def test_gordon_nodes_cached_after_first_build(parcels_xlsx, tmp_path, monkeypatch):
    calls = []

    def fake_fetch(dest_dir):
        calls.append(dest_dir)
        return parcels_xlsx

    monkeypatch.setattr(datasets, "_fetch_gordon_parcels", fake_fetch)
    monkeypatch.setenv("CONNGRAPH_DATA", str(tmp_path / "cache"))

    first = datasets.gordon_nodes()
    second = datasets.gordon_nodes()

    assert len(calls) == 1
    assert (tmp_path / "cache" / "gordon" / "gordon_nodes.tsv").exists()
    pd.testing.assert_frame_equal(first, second)
    assert (second["network"] == "None").sum() == 2


def test_load_gordon_atlas_options(parcels_xlsx):
    assert len(bnv.load_gordon_atlas(parcels_xlsx=str(parcels_xlsx), drop_none=True)) == 4
    perino = bnv.load_gordon_atlas(community="perino2021", parcels_xlsx=str(parcels_xlsx))
    assert perino["network"].tolist()[5] == "MedVisual"


@pytest.fixture
def mock_nodes():
    nets = ["Default", "Salience", "Visual"]
    return pd.DataFrame({
        "label":      [f"{n}_{i}" for n in nets for i in range(6)],
        "network":    [n for n in nets for _ in range(6)],
        "hemisphere": ["L", "R"] * 9,
    })


def _within(matrix, nodes, network):
    keep = (nodes["network"] == network).to_numpy()
    block = matrix.to_numpy()[np.ix_(keep, keep)]
    return block[np.triu_indices(keep.sum(), 1)].mean()


def test_mock_dataset_groups_differ_within_the_named_networks(mock_nodes):
    matrices, nodes, participants = datasets.make_mock_dataset(n_per_group=8, nodes=mock_nodes)

    assert participants["group"].value_counts().to_dict() == {"A": 8, "B": 8}
    assert list(matrices) == participants["participant_id"].tolist()
    for m in matrices.values():
        assert m.index.tolist() == m.columns.tolist() == mock_nodes["label"].tolist()
        np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)
        np.testing.assert_allclose(np.diag(m), 1)
    group = dict(zip(participants["participant_id"], participants["group"]))
    for network, sign in [("Default", 1), ("Visual", -1), ("Salience", 0)]:
        a = np.mean([_within(m, nodes, network) for s, m in matrices.items() if group[s] == "A"])
        b = np.mean([_within(m, nodes, network) for s, m in matrices.items() if group[s] == "B"])
        assert np.sign(round(b - a, 1)) == sign, network


def test_mock_dataset_writes_a_matrix_folder(mock_nodes, tmp_path):
    matrices, _, _ = datasets.make_mock_dataset(n_per_group=2, nodes=mock_nodes, out_dir=tmp_path)

    loaded = bnv.load_group(str(tmp_path), str(tmp_path / "nodes.tsv"), pattern="*_matrix.tsv")
    assert sorted(loaded.matrices) == sorted(matrices)
    np.testing.assert_allclose(np.asarray(loaded.matrices["sub-03"]), matrices["sub-03"].to_numpy(), atol=1e-12)
    assert pd.read_csv(tmp_path / "participants.tsv", sep="\t").shape == (4, 3)


def test_mock_dataset_rejects_an_unknown_network(mock_nodes):
    with pytest.raises(ValueError, match="'Limbic'"):
        datasets.make_mock_dataset(nodes=mock_nodes, weaker="Limbic")
