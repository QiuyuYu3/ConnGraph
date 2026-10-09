from __future__ import annotations

import numpy as np


def load_nifti_atlas(
    atlas_path: str,
    highlight: list[int] | None = None,
    highlight_color: str | tuple = "tomato",
    highlight_alpha: float = 0.9,
    default_alpha: float = 0.35,
    smooth_sigma: float = 1.2,
    isosurface_value: float = 0.3,
    apply_affine: bool = True,
) -> list:
    """
    Load a NIfTI parcellation atlas and return one vedo Mesh per ROI.

    Each integer voxel value (excluding 0) is treated as a distinct ROI.
    Non-highlighted ROIs are coloured with the tab20 colourmap; highlighted
    ROIs use highlight_color at a higher opacity so they stand out.

    After calling this, combine the returned meshes with node Spheres and
    edge Lines from other conngraph functions — all are vedo actors and
    can be passed together to vedo.Plotter.show().

    Parameters
    ----------
    atlas_path       : path to a .nii / .nii.gz file.
    highlight        : ROI integer label(s) to emphasise (e.g. [3, 7]).
    highlight_color  : colour for highlighted ROIs.
    highlight_alpha  : opacity for highlighted ROIs.
    default_alpha    : opacity for non-highlighted ROIs.
    smooth_sigma     : Gaussian sigma applied per ROI before isosurface;
                       higher values give smoother surfaces.
    isosurface_value : isosurface threshold on the smoothed 0-1 mask.
    apply_affine     : transform mesh vertices from voxel index space to
                       scanner/MNI coordinates using the image affine.
                       Set True when overlaying with MNI-space node spheres.

    Returns
    -------
    list of vedo Mesh objects.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")
    try:
        from vedo import Volume
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")
    import matplotlib.pyplot as plt

    img = nib.load(atlas_path)
    data = img.get_fdata()
    affine = img.affine.astype(float)

    roi_labels = np.unique(data).astype(int)
    roi_labels = roi_labels[roi_labels != 0]

    cmap = plt.get_cmap("tab20")
    highlight_set = set(highlight) if highlight else set()

    meshes = []
    for i, roi_label in enumerate(roi_labels):
        mask = (data == roi_label).astype(np.float32)
        smoothed = gaussian_filter(mask, sigma=smooth_sigma)
        vol = Volume(smoothed)
        mesh = vol.isosurface(value=isosurface_value)
        if not mesh.npoints:
            continue

        if apply_affine:
            mesh.apply_transform(affine)

        if roi_label in highlight_set:
            mesh.c(highlight_color).alpha(highlight_alpha).lighting("default")
        else:
            mesh.c(cmap(i % cmap.N)[:3]).alpha(default_alpha).lighting("off")

        meshes.append(mesh)

    return meshes
