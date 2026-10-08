"""
Edge bundling: bend edges that run close to each other into shared bundles.
"""

from __future__ import annotations

import numpy as np

_RANGE_PER_STRENGTH = 0.05
_STEP = 0.35


def bundle_paths(
    starts: np.ndarray,
    ends: np.ndarray,
    strength: float = 1.0,
    n_points: int = 24,
    iterations: int = 12,
    block: int = 1024,
) -> np.ndarray:
    """Return (E, n_points, 3) curves from starts to ends, pulled towards nearby edges; strength 0 gives straight lines."""
    if strength < 0:
        raise ValueError(f"strength must be 0 or more, got {strength}.")
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    t = np.linspace(0.0, 1.0, n_points)
    paths = starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]
    if strength == 0 or len(paths) < 2:
        return paths

    # the attraction range follows the size of the layout, so MNI and graph layouts behave alike
    span = float(np.linalg.norm(np.ptp(np.vstack([starts, ends]), axis=0))) or 1.0
    sigma = _RANGE_PER_STRENGTH * span * strength
    # 0 at both ends and 1 in the middle, so endpoints stay on their nodes
    free = (1 - np.abs(2 * t - 1) ** 4)[None, :, None].astype(np.float32)

    pts = paths.astype(np.float32)
    e = len(pts)
    for _ in range(iterations):
        x = pts.reshape(e, -1)
        xr = pts[:, ::-1].reshape(e, -1)
        sq = (x ** 2).sum(axis=1)
        target = np.empty_like(x)
        for a in range(0, e, block):
            b = min(a + block, e)
            d_same = (sq[a:b, None] + sq[None, :] - 2 * x[a:b] @ x.T) / n_points
            d_flip = (sq[a:b, None] + sq[None, :] - 2 * x[a:b] @ xr.T) / n_points
            # compare each pair in whichever direction they line up best
            flipped = d_flip < d_same
            k = np.exp(-np.minimum(d_same, d_flip).clip(min=0) / (2 * sigma ** 2))
            target[a:b] = (np.where(flipped, 0, k) @ x + np.where(flipped, k, 0) @ xr) / k.sum(axis=1, keepdims=True)
        pts = pts + _STEP * free * (target.reshape(pts.shape) - pts)
        pts[:, 1:-1] = (pts[:, :-2] + 2 * pts[:, 1:-1] + pts[:, 2:]) / 4

    out = pts.astype(float)
    out[:, 0], out[:, -1] = starts, ends
    return out
