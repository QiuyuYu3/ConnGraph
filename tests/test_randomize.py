import numpy as np

from conngraph.graph_theory.randomize import randomize_signed


def _signed_graph(n: int = 40, density: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((200, 4))
    corr = np.corrcoef((factors[:, np.arange(n) % 4] + rng.standard_normal((200, n))).T)
    upper = np.triu(corr - corr[np.triu_indices(n, 1)].mean(), 1)  # about half the weights negative
    cutoff = np.quantile(np.abs(upper[np.triu_indices(n, 1)]), 1 - density)
    upper[np.abs(upper) < cutoff] = 0
    return upper + upper.T


def _degrees(A: np.ndarray, sign: int) -> np.ndarray:
    return (np.sign(A) == sign).sum(axis=1)


def _weights(A: np.ndarray, sign: int) -> np.ndarray:
    return np.sort(A[np.triu(np.sign(A) == sign, 1)])


def test_keeps_positive_and_negative_degrees_and_weights():
    A = _signed_graph()
    R = randomize_signed(A, rng=0)
    assert np.array_equal(R, R.T)
    assert not np.diag(R).any()
    for sign in (1, -1):
        np.testing.assert_array_equal(_degrees(R, sign), _degrees(A, sign))
        np.testing.assert_array_equal(_weights(R, sign), _weights(A, sign))
    moved = np.triu((A != 0) & (R == 0), 1).sum() / np.triu(A != 0, 1).sum()
    assert moved > 0.5


def test_keeps_strength_roughly():
    A = _signed_graph(n=60, density=0.2)
    R = randomize_signed(A, rng=1)
    for sign in (1, -1):
        s_a = np.abs(np.where(np.sign(A) == sign, A, 0)).sum(axis=1)
        s_r = np.abs(np.where(np.sign(R) == sign, R, 0)).sum(axis=1)
        assert np.corrcoef(s_a, s_r)[0, 1] > 0.95


def test_randomizes_a_complete_signed_graph():
    A = _signed_graph(n=12, density=1.0)
    R = randomize_signed(A, rng=2)
    assert (np.sign(R) != np.sign(A)).any()
    for sign in (1, -1):
        np.testing.assert_array_equal(_degrees(R, sign), _degrees(A, sign))


def test_seed_makes_it_reproducible():
    A = _signed_graph()
    np.testing.assert_array_equal(randomize_signed(A, rng=3), randomize_signed(A, rng=3))
    assert not np.array_equal(randomize_signed(A, rng=3), randomize_signed(A, rng=4))
