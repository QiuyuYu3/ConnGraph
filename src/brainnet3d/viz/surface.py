"""
Surface rendering: load .surf.gii brain meshes → vedo Mesh objects.
"""

from __future__ import annotations


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
