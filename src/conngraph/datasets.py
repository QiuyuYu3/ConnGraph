from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from conngraph.exceptions import DownloadError

PathLike = str | os.PathLike

_CONFIG_DIR = pathlib.Path(__file__).parent / "config"
_XCPD_GORDON_NETWORKS = {"CinguloParietal": "MedialParietal", "RetrosplenialTemporal": "ParietoOccip"}


def get_data_dir(data_dir: PathLike | None = None) -> pathlib.Path:
    """Return the cache folder: data_dir, else $CONNGRAPH_DATA, else ~/conngraph_data."""
    path = pathlib.Path(data_dir or os.environ.get("CONNGRAPH_DATA") or pathlib.Path.home() / "conngraph_data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gordon_nodes(parcels_xlsx: PathLike | None = None, data_dir: PathLike | None = None) -> pd.DataFrame:
    """Gordon 333-parcel node table, built from parcels_xlsx or from the cached official release."""
    if parcels_xlsx is not None:
        return _build_gordon_nodes(parcels_xlsx)

    cache = get_data_dir(data_dir) / "gordon" / "gordon_nodes.tsv"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        nodes = _build_gordon_nodes(_fetch_gordon_parcels(cache.parent))
        nodes.to_csv(cache, sep="\t", index=False)
    return pd.read_csv(cache, sep="\t", keep_default_na=False)


def _fetch_gordon_parcels(dest_dir: pathlib.Path) -> pathlib.Path:
    entry = json.loads((_CONFIG_DIR / "datasets.json").read_text(encoding="utf-8"))["gordon"]
    manual = f"Download it manually from {entry['page']} and pass the Parcels.xlsx path as parcels_xlsx."
    try:
        with urllib.request.urlopen(entry["url"], timeout=60) as resp:
            archive = resp.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            content = zf.read(entry["member"])
    except (OSError, zipfile.BadZipFile, KeyError) as e:
        raise DownloadError(f"Could not download the Gordon parcellation ({e}). {manual}") from e

    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise DownloadError(f"The downloaded Parcels.xlsx does not match the expected release. {manual}")

    path = dest_dir / "Parcels.xlsx"
    path.write_bytes(content)
    return path


def _build_gordon_nodes(parcels_xlsx: PathLike) -> pd.DataFrame:
    try:
        xl = pd.read_excel(parcels_xlsx, keep_default_na=False)
    except ImportError:
        raise ImportError("Reading Parcels.xlsx needs openpyxl: pip install openpyxl") from None

    xl = xl.sort_values("ParcelID").reset_index(drop=True)
    network = xl["Community"].astype(str).replace(_XCPD_GORDON_NETWORKS)
    # XCP-D numbers parcels within each network across both hemispheres
    counter = network.groupby(network).cumcount() + 1
    coords = xl["Centroid (MNI)"].astype(str).str.split(expand=True).astype(float)
    area_col = next(c for c in xl.columns if str(c).lower().startswith("surface area"))

    return pd.DataFrame({
        "parcel_id":          xl["ParcelID"].astype(int),
        "label":              xl["Hem"] + "_" + network + "_" + counter.astype(str),
        "hemisphere":         xl["Hem"],
        "x":                  coords[0],
        "y":                  coords[1],
        "z":                  coords[2],
        "network":            network,
        "network_perino2021": xl["Community_Perino2021"].astype(str),
        "surface_area_mm2":   xl[area_col].astype(float),
    })


def make_mock_dataset(
    n_per_group: int = 15,
    nodes: pd.DataFrame | None = None,
    stronger: str = "Default",
    weaker: str = "Visual",
    seed: int = 0,
    out_dir: PathLike | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Simulated groups A and B (B more connected within `stronger`, less within `weaker`) on Gordon or `nodes`."""
    if nodes is None:
        from conngraph.loaders import load_gordon_atlas

        nodes = load_gordon_atlas()[["label", "network", "hemisphere", "x", "y", "z"]]
    nets = sorted(set(nodes["network"]))
    missing = sorted({stronger, weaker} - set(nets))
    if missing:
        raise ValueError(f"The node table has no network named {', '.join(map(repr, missing))}.")

    labels = nodes["label"].tolist()
    idx = nodes["network"].map(nets.index).to_numpy()
    rng = np.random.default_rng(seed)
    # the network signals share three common signals, which correlates some networks positively, some negatively
    mix = rng.normal(0, 1.0, (len(nets), 3))
    scale = np.sqrt(1 + (mix ** 2).sum(axis=1))
    matrices, rows = {}, []
    for s in range(2 * n_per_group):
        group = "A" if s < n_per_group else "B"
        # each network has one signal; its loading sets the within-network connectivity
        load = rng.uniform(0.55, 0.8, len(nets))
        if "None" in nets:
            load[nets.index("None")] = 0
        if group == "B":
            load[nets.index(stronger)] += 0.25
            load[nets.index(weaker)] -= 0.2
        signals = (rng.standard_normal((400, len(nets))) + rng.standard_normal((400, 3)) @ mix.T) / scale
        ts = signals[:, idx] * load[idx] + rng.standard_normal((400, len(labels)))
        sid = f"sub-{s + 1:02d}"
        matrices[sid] = pd.DataFrame(np.corrcoef(ts.T), index=labels, columns=labels)
        rows.append({"participant_id": sid, "group": group, "age": round(float(rng.uniform(20, 40)), 1)})
    participants = pd.DataFrame(rows)

    if out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes.to_csv(out / "nodes.tsv", sep="\t", index=False)
        participants.to_csv(out / "participants.tsv", sep="\t", index=False)
        for sid, matrix in matrices.items():
            matrix.to_csv(out / f"{sid}_matrix.tsv", sep="\t")
    return matrices, nodes, participants
