import nibabel as nib
import numpy as np
import pytest

from conngraph.cli import main as cli
from conngraph.viz import surface


def _ellipsoid(path, shape=(60, 70, 60)):
    """A brain-like blob centred on x = 0, saved with a 2 mm affine."""
    grid = np.indices(shape).astype(float)
    centre = (np.array(shape) - 1) / 2
    inside = sum(((g - c) / (s * 0.4)) ** 2 for g, c, s in zip(grid, centre, shape)) <= 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -2 * centre
    nib.save(nib.Nifti1Image((inside * 120).astype(np.int16), affine), path)


def test_a_volume_becomes_left_and_right_surfaces(tmp_path):
    volume = tmp_path / "template.nii.gz"
    _ellipsoid(volume)
    left, right = surface._volume_surfaces(volume, data_dir=tmp_path / "cache")
    lx = nib.load(left).darrays[0].data[:, 0]
    rx = nib.load(right).darrays[0].data[:, 0]
    assert lx.max() <= 1e-3 and rx.min() >= -1e-3
    assert lx.min() == pytest.approx(-48, abs=4) and rx.max() == pytest.approx(48, abs=4)
    assert len(surface.load_surface(left, right)) == 2


def test_a_converted_volume_is_reused(tmp_path, monkeypatch):
    volume = tmp_path / "template.nii.gz"
    _ellipsoid(volume)
    first = surface._volume_surfaces(volume, data_dir=tmp_path / "cache")
    monkeypatch.setattr(surface, "_shell_mesh", lambda *a, **k: pytest.fail("converted again"))
    assert surface._volume_surfaces(volume, data_dir=tmp_path / "cache") == first


def test_the_cli_takes_one_volume_or_two_surfaces(tmp_path, monkeypatch, capsys):
    from conngraph.cli import _shared

    volume = tmp_path / "template.nii.gz"
    _ellipsoid(volume)
    monkeypatch.setenv("CONNGRAPH_DATA", str(tmp_path / "cache"))
    args, _ = cli.parse_args(["in", "out", "group", "--input-type", "matrix", "--surfaces", str(volume)])
    left, right = _shared.surfaces(args)
    assert left.endswith("hemi-L.surf.gii") and right.endswith("hemi-R.surf.gii")
    args, _ = cli.parse_args(["in", "out", "group", "--input-type", "matrix", "--surfaces", "l.surf.gii", "r.surf.gii"])
    assert _shared.surfaces(args) == ("l.surf.gii", "r.surf.gii")
    with pytest.raises(SystemExit):
        cli.parse_args(["in", "out", "group", "--input-type", "matrix", "--surfaces", "a", "b", "c"])
    assert "--surfaces" in capsys.readouterr().err
