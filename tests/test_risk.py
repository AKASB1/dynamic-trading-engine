"""Check 12: risk models."""

import numpy as np
import pytest
from helpers import truncate_market

from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.risk.models import (
    RiskModel,
    ewma_cov,
    floor_psd,
    ledoit_wolf,
    sample_cov,
    shrink_to_identity,
)
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.state.view import build_state

MODELS = ("sample", "ewma", "lw", "pca")


def _cols(st):
    return np.nonzero(np.asarray(st.n_known) >= 60)[0]


@pytest.mark.parametrize("name", MODELS)
def test_symmetric_psd_deterministic(name):
    m, _ = generate(load_market_config("tiny"), 2)
    st = build_state(m, int(m.ts_event[200]), 1e6, np.zeros(m.n))
    rm = RiskModel(name)
    a = rm.estimate(st, _cols(st))
    b = RiskModel(name).estimate(st, _cols(st))
    assert np.array_equal(a.sigma, b.sigma)
    assert np.array_equal(a.sigma, a.sigma.T)
    lam = np.linalg.eigvalsh(a.sigma)
    n = a.sigma.shape[0]
    assert lam.min() >= 1e-10 * np.trace(a.sigma) / n * (1 - 1e-6)
    assert np.allclose(a.factor.T @ a.factor, a.sigma, rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("name", MODELS)
def test_no_look_ahead_bit_for_bit(name):
    m, _ = generate(load_market_config("tiny"), 3)
    g = stream(4, "test.risk.instants")
    for k in g.integers(80, m.n_bars - 1, 50):
        k = int(k)
        t = int(m.ts_event[k])
        cut = truncate_market(m, k)
        s_full = build_state(m, t, 1e6, np.zeros(m.n))
        s_cut = build_state(cut, t, 1e6, np.zeros(m.n))
        cols = _cols(s_full)
        if len(cols) < 2:
            continue
        a = RiskModel(name).estimate(s_full, cols)
        b = RiskModel(name).estimate(s_cut, cols)
        assert np.array_equal(a.sigma, b.sigma)


def test_ewma_hand_case():
    R = np.array([[0.01, 0.0], [-0.02, 0.01], [0.03, -0.01]])
    hl = 1.0
    w = np.array([0.25, 0.5, 1.0]) / 1.75
    mu = w @ R
    want = sum(w[j] * np.outer(R[j] - mu, R[j] - mu) for j in range(3))
    assert np.allclose(ewma_cov(R, hl), want, rtol=1e-14)
    # the same by the normalized recursion: W_j = lam W_{j-1} + 1, mean and co-moment updated
    lam = 0.5
    Wn, mean, M = 0.0, np.zeros(2), np.zeros((2, 2))
    for r in R:
        W_new = lam * Wn + 1.0
        mean_new = (lam * Wn * mean + r) / W_new
        M = lam * (M + Wn * np.outer(mean - mean_new, mean - mean_new)) + np.outer(
            r - mean_new, r - mean_new
        )
        Wn, mean = W_new, mean_new
    assert np.allclose(M / Wn, want, rtol=1e-12)


def test_lw_intensity_and_limits():
    g = stream(1, "test.lw")
    R = g.standard_normal((40, 10)) * 0.01
    S_lw, delta = ledoit_wolf(R)
    assert 0.0 <= delta <= 1.0
    X = R - R.mean(axis=0)
    S = X.T @ X / R.shape[0]
    assert np.allclose(shrink_to_identity(S, 0.0), S)
    assert np.allclose(shrink_to_identity(S, 1.0), np.trace(S) / 10 * np.eye(10))
    assert np.allclose(S_lw, shrink_to_identity(S, delta))


def _base_variances(g, n):
    beta = g.uniform(0.6, 1.4, n)
    idio = g.uniform(0.15, 0.35, n)
    return (beta**2 * 0.16**2 + 0.10**2 + idio**2) / 252


@pytest.mark.parametrize("n", [30, 100])
@pytest.mark.parametrize("ratio", [2, 1])
@pytest.mark.parametrize("truth", ["base", "clean"])
def test_lw_beats_sample_on_diagonal_truth(n, ratio, truth):
    g = stream(n * 10 + ratio, f"test.lw.{truth}")
    var = _base_variances(g, n) if truth == "base" else np.full(n, 0.25**2 / 252)
    W = ratio * n
    wins = 0
    for _ in range(100):
        R = g.standard_normal((W, n)) * np.sqrt(var)
        true = np.diag(var)
        e_s = np.linalg.norm(sample_cov(R) - true)
        e_lw = np.linalg.norm(ledoit_wolf(R)[0] - true)
        wins += e_lw < e_s
    assert wins >= 95


def test_pca_reconstruction_and_diagonal():
    g = stream(2, "test.pca")
    R = g.standard_normal((200, 6)) @ g.standard_normal((6, 6)) * 0.01
    rm = RiskModel("pca", n_pc=6)
    est = rm.estimate_from_returns(R)
    assert np.allclose(est.sigma, sample_cov(R), rtol=1e-8, atol=1e-14)
    rm3 = RiskModel("pca", n_pc=3)
    est3 = rm3.estimate_from_returns(R)
    assert np.allclose(np.diag(est3.sigma), np.diag(sample_cov(R)), rtol=1e-10)


def test_floor_applies():
    S = np.array([[1.0, 1.0], [1.0, 1.0]])
    Sf, F = floor_psd(S)
    assert np.linalg.eigvalsh(Sf).min() >= 1e-10 * 2 / 2 * (1 - 1e-9)


def _lw_brute_force(R):
    """Ledoit and Wolf (2004), Lemma 3.2 / Theorem 3.1, written out with loops."""
    T, n = R.shape
    X = R - R.mean(axis=0)
    S = sum(np.outer(X[t], X[t]) for t in range(T)) / T
    norm2 = lambda A: float(np.trace(A @ A.T)) / n  # noqa: E731 - the paper's norm
    m = float(np.trace(S)) / n
    d2 = norm2(S - m * np.eye(n))
    b_bar2 = sum(norm2(np.outer(X[t], X[t]) - S) for t in range(T)) / T**2
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    return b2 / d2 * m * np.eye(n) + a2 / d2 * S, b2 / d2


def test_lw_matches_an_independent_brute_force():
    g = stream(5, "test.lw.brute")
    for T, n in ((20, 5), (60, 12), (12, 30)):
        R = g.standard_normal((T, n)) * 0.01 + g.standard_normal((T, 1)) * 0.005
        S, delta = ledoit_wolf(R)
        S_ref, delta_ref = _lw_brute_force(R)
        assert abs(delta - delta_ref) <= 1e-12
        assert np.allclose(S, S_ref, rtol=1e-12, atol=1e-18)
