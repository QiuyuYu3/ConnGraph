"""
Surface rendering: load .surf.gii brain meshes → vedo Mesh objects.
"""

from __future__ import annotations

import pathlib


def get_fsLR_surface() -> tuple[str, str]:
    """Return (path_L, path_R) for the fsLR 32k midthickness surfaces, fetched via TemplateFlow on first use."""
    try:
        from templateflow import api as tflow
    except ImportError:
        raise ImportError("get_fsLR_surface needs templateflow: pip install templateflow") from None

    left, right = (
        str(tflow.get("fsLR", hemi=h, density="32k", suffix="midthickness", extension=".surf.gii"))
        for h in ("L", "R")
    )
    return left, right


VOLUME_SUFFIXES = (".nii", ".nii.gz", ".HEAD", ".BRIK", ".BRIK.gz")


def is_volume(path) -> bool:
    return str(path).endswith(VOLUME_SUFFIXES)


def _volume_surfaces(path, data_dir=None) -> tuple[str, str]:
    """Left and right .surf.gii of a brain volume's smoothed outline, cut at x = 0 and cached by content."""
    import hashlib

    import nibabel as nib
    import numpy as np

    from conngraph.datasets import get_data_dir

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    data = data[..., 0] if data.ndim == 4 else data
    key = hashlib.sha256(np.ascontiguousarray(data != 0).tobytes() + img.affine.tobytes()).hexdigest()[:16]
    folder = get_data_dir(data_dir) / "surfaces"
    targets = tuple(str(folder / f"{key}_hemi-{h}.surf.gii") for h in ("L", "R"))
    if all(pathlib.Path(t).exists() for t in targets):
        return targets
    folder.mkdir(parents=True, exist_ok=True)
    mesh = _shell_mesh(data != 0, img.affine)
    for target, normal in zip(targets, ((-1, 0, 0), (1, 0, 0))):
        half = mesh.clone().cut_with_plane(origin=(0, 0, 0), normal=normal).triangulate().clean()
        nib.save(nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(np.asarray(half.vertices, dtype=np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(np.asarray(half.cells, dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE")]), target)
    return targets


def _shell_mesh(mask, affine):
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    from vedo import Volume

    # the blurred mask's half-height surface is the brain's outline without the voxel staircase
    blurred = ndimage.gaussian_filter(mask.astype(float) * 100, 1.5)
    mesh = Volume(blurred).isosurface(50).extract_largest_region().decimate(0.15).smooth(niter=30)
    mesh.vertices = nib.affines.apply_affine(affine, np.asarray(mesh.vertices))
    return mesh


def load_surface(
    surface_L: str | None = None,
    surface_R: str | None = None,
    color:     str | tuple = (0.93, 0.90, 0.84),
    alpha:     float = 0.15,
    smooth:    int   = 50,
) -> list:
    """
    Load one or both hemisphere surface meshes.

    Parameters
    ----------
    surface_L, surface_R : path to .surf.gii files.  Pass None to skip.
    color  : RGB tuple or colour name for the surface.
    alpha  : transparency (0 = invisible, 1 = opaque).
    smooth : number of Laplacian smoothing iterations.

    Returns
    -------
    list of vedo Mesh objects (0, 1, or 2 elements).
    """
    meshes = []
    for path in (surface_L, surface_R):
        if path is not None:
            meshes.append(_build_mesh(path, color, alpha, smooth))
    return meshes


def _build_mesh(path: str, color, alpha: float, smooth: int):
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for surface loading: pip install nibabel")

    try:
        from vedo import Mesh
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")

    gii      = nib.load(path)
    vertices = gii.darrays[0].data.astype(float)
    faces    = gii.darrays[1].data.astype(int)

    mesh = (
        Mesh([vertices, faces])
        .clean()
        .triangulate()
        .smooth(smooth)
        .extract_largest_region()
        # vertex normals give smooth shading; without them folds seen edge-on show the triangle grid
        .compute_normals(points=True, cells=False)
    )
    mesh.c(color).alpha(alpha).lighting("default")
    return mesh
