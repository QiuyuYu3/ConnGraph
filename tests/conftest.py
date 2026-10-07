import os

os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

import matplotlib

matplotlib.use("Agg")

import pathlib

import numpy as np
import pandas as pd
import pytest

import brainnet3d as bnv

NETWORKS = {
    "Default":  dict(center=(0,  50, 20), spread=25, hemi_offset=20),
    "Salience": dict(center=(0,  20,  0), spread=20, hemi_offset=35),
    "Visual":   dict(center=(0, -80, 10), spread=20, hemi_offset=20),
}
N_PER_NET = 8
N_SUBJECTS = 6


def _make_nodes(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for net, cfg in NETWORKS.items():
        cx, cy, cz = cfg["center"]
        sp, ox = cfg["spread"], cfg["hemi_offset"]
        for hemi, sign in [("L", -1), ("R", 1)]:
            for _ in range(N_PER_NET // 2):
                rows.append(dict(
                    label=f"{net[:3]}_{hemi}_{len(rows):02d}",
                    x=round(sign * (ox + rng.uniform(0, sp * 0.5)), 1),
                    y=round(cy + rng.uniform(-sp, sp), 1),
                    z=round(cz + rng.uniform(-sp * 0.5, sp * 0.5), 1),
                    network=net,
                    hemisphere=hemi,
                ))
    return pd.DataFrame(rows)


def _make_matrix(nodes_df: pd.DataFrame, group: int, rng: np.random.Generator) -> np.ndarray:
    n = len(nodes_df)
    networks = nodes_df["network"].tolist()
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if networks[i] == networks[j]:
                w = (0.55 if group == 1 else 0.65) + rng.uniform(-0.1, 0.1)
            else:
                w = rng.uniform(-0.05, 0.2)
            mat[i, j] = mat[j, i] = w
    np.fill_diagonal(mat, 1.0)
    return mat


@pytest.fixture(scope="session")
def mock_dir(tmp_path_factory) -> pathlib.Path:
    rng = np.random.default_rng(42)
    out = tmp_path_factory.mktemp("mock_data")
    nodes = _make_nodes(rng)
    nodes.to_csv(out / "nodes.csv", index=False)
    labels = nodes["label"].tolist()
    for i in range(N_SUBJECTS):
        group = 1 if i < N_SUBJECTS // 2 else 2
        mat = _make_matrix(nodes, group, rng)
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(out / f"sub-{i + 1:02d}_matrix.csv")
    return out


@pytest.fixture(scope="session")
def dataset(mock_dir) -> bnv.ConnectivityDataset:
    return bnv.load_group(str(mock_dir), str(mock_dir / "nodes.csv"))


@pytest.fixture(scope="session")
def groups(dataset):
    ids = sorted(dataset.matrices)
    half = len(ids) // 2
    return (
        {s: dataset.matrices[s] for s in ids[:half]},
        {s: dataset.matrices[s] for s in ids[half:]},
    )


@pytest.fixture(scope="session")
def out_dir(tmp_path_factory) -> pathlib.Path:
    """Render output folder: $BRAINNET3D_TEST_OUTPUT if set (to inspect figures), else a pytest temp dir."""
    target = os.environ.get("BRAINNET3D_TEST_OUTPUT")
    if not target:
        return tmp_path_factory.mktemp("render_output")
    path = pathlib.Path(target)
    path.mkdir(parents=True, exist_ok=True)
    return path
