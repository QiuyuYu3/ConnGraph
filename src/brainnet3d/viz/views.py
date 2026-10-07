from __future__ import annotations

import numpy as np


_DEFAULT_CAMERAS: dict[str, dict] = {
    "Left":   dict(pos=(-380, 0, 40), focalPoint=(0, 0, 0), viewup=(0, 0, 1)),
    "Right":  dict(pos=( 380, 0, 40), focalPoint=(0, 0, 0), viewup=(0, 0, 1)),
    "Dorsal": dict(pos=(  0, 0, 450), focalPoint=(0, 0, 0), viewup=(0, 1, 0)),
}


def save_three_views(
    actors: list,
    output_path: str = "three_views.png",
    panel_size: tuple[int, int] = (600, 500),
    cameras: dict[str, dict] | None = None,
    bg: str = "white",
) -> None:
    """
    Render a list of vedo actors from three orthogonal viewpoints and save
    as a single side-by-side PNG.

    Parameters
    ----------
    actors      : list of vedo actor objects (Mesh, Sphere, Line, …).
    output_path : destination file path.
    panel_size  : (width, height) per panel in pixels.
    cameras     : dict of {label: camera_dict} to override the defaults.
                  Each camera_dict must have keys: pos, focalPoint, viewup.
                  Default views are Left-lateral, Right-lateral, and Dorsal.
    bg          : background colour passed to vedo Plotter.
    """
    try:
        from vedo import Plotter as VPlotter
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")

    views  = cameras if cameras is not None else _DEFAULT_CAMERAS
    w, h   = panel_size
    canvas = Image.new("RGB", (len(views) * w, h), bg)
    draw   = ImageDraw.Draw(canvas)

    for col, (label, cam) in enumerate(views.items()):
        plt = VPlotter(offscreen=True, size=panel_size, bg=bg)
        plt.show(actors, camera=cam, interactive=False)
        img = plt.screenshot(asarray=True)
        plt.close()

        canvas.paste(Image.fromarray(img), (col * w, 0))
        draw.text((col * w + 10, 10), label, fill="black")

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def make_axis_arrows(
    axes: str | list[str] = "all",
    origin: tuple[float, float, float] = (-95, -115, -70),
    length: float = 30.0,
) -> list:
    """
    Create anatomical orientation arrows for a MNI-space vedo scene.

    Parameters
    ----------
    axes   : axes to draw. "all" draws all three. Otherwise a list of any
             subset of ["LR", "AP", "SI"].
    origin : arrow base in MNI coordinates. Default is lower-left-anterior
             corner of a typical whole-brain view.
    length : arrow length in mm.

    Returns
    -------
    List of vedo Arrow + Text3D objects. Add to your actors list before
    calling Plotter.show().

    Example
    -------
    >>> actors = surface_meshes + node_spheres + make_axis_arrows()
    >>> Plotter(bg="white", axes=0).show(actors)
    """
    try:
        from vedo import Arrow, Text3D
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")

    if axes == "all":
        axes = ["LR", "AP", "SI"]

    ox, oy, oz = origin
    axis_defs = {
        "LR": (np.array([length, 0, 0]), "R", "red4"),
        "AP": (np.array([0, length, 0]), "A", "green4"),
        "SI": (np.array([0, 0, length]), "S", "blue4"),
    }

    actors = []
    for ax in axes:
        if ax not in axis_defs:
            raise ValueError(f"axes value '{ax}' not recognised. Choose from: LR, AP, SI.")
        direction, label, color = axis_defs[ax]
        end = (ox + direction[0], oy + direction[1], oz + direction[2])
        actors.append(Arrow(origin, end, c=color, s=0.04).lighting("ambient"))
        actors.append(
            Text3D(label, pos=end, s=length * 0.25, c=color, justify="center")
        )

    return actors


def save_orbit_gif(
    actors: list,
    output_path: str = "orbit.gif",
    n_frames: int = 36,
    distance: float = 380,
    elevation: float = 40,
    fps: int = 12,
    panel_size: tuple[int, int] = (800, 600),
    bg: str = "white",
    focal_point: tuple[float, float, float] = (0, 0, 0),
    verbose: bool = True,
) -> None:
    """
    Render a 360° camera orbit around a set of vedo actors and save as GIF.

    Works with any actors list: brain network nodes/edges, atlas meshes,
    filtration simplices, etc.

    Parameters
    ----------
    actors      : list of vedo actor objects.
    output_path : destination .gif file path.
    n_frames    : number of frames (higher = smoother, larger file).
    distance    : camera distance from focal_point.
    elevation   : fixed camera height (z) above focal_point.
    fps         : playback frame rate.
    panel_size  : (width, height) per frame in pixels.
    bg          : background colour.
    focal_point : point the camera looks at throughout the orbit.
    verbose     : print per-frame progress.
    """
    try:
        from vedo import Plotter as VPlotter
    except ImportError:
        raise ImportError("vedo is required: pip install vedo")

    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required: pip install imageio")

    fx, fy, fz = focal_point
    frames = []

    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        cam = dict(
            pos=(fx + distance * np.sin(angle), fy + distance * np.cos(angle), fz + elevation),
            focalPoint=focal_point,
            viewup=(0, 0, 1),
        )
        if verbose:
            print(f"  [{i + 1}/{n_frames}]")

        plt = VPlotter(offscreen=True, size=panel_size, bg=bg)
        plt.show(actors, camera=cam, interactive=False)
        frames.append(plt.screenshot(asarray=True))
        plt.close()

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    if verbose:
        print(f"Saved: {output_path}")


def _finish_render(
    plt,
    actors: list,
    interactive: bool,
    screenshot: str | None = None,
    html: str | None = None,
    camera: dict | None = None,
) -> np.ndarray | None:
    """Show actors, write the requested files, and return the image unless a window was opened."""
    show_kwargs: dict = {"interactive": False}
    if camera:
        show_kwargs["camera"] = camera
    plt.show(actors, **show_kwargs)

    # vedo ignores show(screenshot=...) without an interactor, so save explicitly
    if screenshot is not None:
        plt.screenshot(screenshot)
    if html is not None:
        _export_html(plt, html)

    image = None
    if interactive:
        plt.interactive()
    else:
        image = np.asarray(plt.screenshot(asarray=True))
    plt.close()
    return image


def _export_html(plt, path: str) -> None:
    try:
        import k3d  # noqa: F401
    except ImportError:
        raise ImportError("html export needs k3d: pip install 'brainnet3d[html]'") from None
    import vedo

    s = vedo.settings
    saved = (s.k3d_grid_visible, s.k3d_axes_helper, s.backend_autoclose)
    # the k3d export closes the current plotter unless autoclose is off
    s.k3d_grid_visible, s.k3d_axes_helper, s.backend_autoclose = False, 0, False
    try:
        vedo.file_io.export_window(str(path), plt=plt)
    finally:
        s.k3d_grid_visible, s.k3d_axes_helper, s.backend_autoclose = saved
