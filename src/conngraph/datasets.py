from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import urllib.request
import zipfile

import pandas as pd

from brainnet3d.exceptions import DownloadError

PathLike = str | os.PathLike

_CONFIG_DIR = pathlib.Path(__file__).parent / "config"
_XCPD_GORDON_NETWORKS = {"CinguloParietal": "MedialParietal", "RetrosplenialTemporal": "ParietoOccip"}


def get_data_dir(data_dir: PathLike | None = None) -> pathlib.Path:
    """Return the cache folder: data_dir, else $BRAINNET3D_DATA, else ~/brainnet3d_data."""
    path = pathlib.Path(data_dir or os.environ.get("BRAINNET3D_DATA") or pathlib.Path.home() / "brainnet3d_data")
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
