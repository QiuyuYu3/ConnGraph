import numpy as np
import pytest

from brainnet3d.viz.bundling import bundle_paths


def _edges():
    # two long parallel edges 8 mm apart and a third far from both
    starts = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 80.0, 0.0]])
    ends = starts + [100.0, 0.0, 0.0]
    return starts, ends


def _mid_gap(paths, a=0, b=1):
    mid = paths.shape[1] // 2
    return float(np.linalg.norm(paths[a, mid] - paths[b, mid]))


def test_zero_strength_gives_straight_lines():
    starts, ends = _edges()
    paths = bundle_paths(starts, ends, strength=0)
    t = np.linspace(0, 1, paths.shape[1])[None, :, None]
    np.testing.assert_allclose(paths, starts[:, None] + t * (ends - starts)[:, None])


def test_endpoints_stay_on_their_nodes():
    starts, ends = _edges()
    paths = bundle_paths(starts, ends, strength=1.0)
    np.testing.assert_array_equal(paths[:, 0], starts)
    np.testing.assert_array_equal(paths[:, -1], ends)


def test_nearby_edges_join_and_far_edges_stay():
    starts, ends = _edges()
    straight = bundle_paths(starts, ends, strength=0)
    light, strong = bundle_paths(starts, ends, strength=0.5), bundle_paths(starts, ends, strength=1.0)
    assert _mid_gap(strong) < _mid_gap(light) < _mid_gap(straight)
    assert np.abs(strong[2] - straight[2]).max() < 0.5


def test_edge_direction_does_not_matter():
    starts, ends = _edges()
    flipped_starts, flipped_ends = starts.copy(), ends.copy()
    flipped_starts[1], flipped_ends[1] = ends[1], starts[1]
    paths = bundle_paths(starts, ends)
    flipped = bundle_paths(flipped_starts, flipped_ends)
    np.testing.assert_allclose(flipped[1], paths[1][::-1], atol=1e-4)


def test_blocks_do_not_change_the_result():
    rng = np.random.default_rng(0)
    starts, ends = rng.uniform(-60, 60, (50, 3)), rng.uniform(-60, 60, (50, 3))
    np.testing.assert_allclose(bundle_paths(starts, ends, block=7), bundle_paths(starts, ends), atol=1e-4)


def test_negative_strength_raises():
    starts, ends = _edges()
    with pytest.raises(ValueError, match="strength"):
        bundle_paths(starts, ends, strength=-1)
