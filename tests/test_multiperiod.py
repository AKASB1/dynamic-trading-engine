"""Check 9: multi-period control against the exact references (R6, R7)."""

import numpy as np
import pytest

from dynamic_trading_engine.contracts.costs import CostConfig
from dynamic_trading_engine.optimization.optimizers import Optimizer, OptimizerConfig
from dynamic_trading_engine.policies.dp import no_trade_band, solve_dp
from dynamic_trading_engine.policies.gp import GPParams, gp_closed_form, stationary_sd
from dynamic_trading_engine.policies.single import (
    mpc_coefficients,
    mpc_first_action_cvxpy,
    mpc_first_action_la,
)
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.rng import stream

P = GPParams()


def test_gp_known_answer():
    s = gp_closed_form(P)
    for got, want in (
        (s.A, 0.14129787310950273),
        (s.B, 1.1625476588156523),
        (s.a, 0.3532446827737568),
        (s.aim, 8.227637353852478),
    ):
        assert abs(got - want) <= 1e-12 * abs(want)
    assert abs(s.residual) <= 1e-15
    assert s.aim < 1 / P.g


@pytest.fixture(scope="module")
def dp_quadratic():
    sd = stationary_sd(P)
    return solve_dp(P.g, P.phi_gp, P.sigma_eps, P.delta, "quadratic", P.Lam, 0.8, 241, 5 * sd, 61)


def test_grid_dp_reproduces_gp_policy(dp_quadratic):
    res = dp_quadratic
    s = gp_closed_form(P)
    step = res.x_grid[1] - res.x_grid[0]
    sd = stationary_sd(P)
    g = stream(1, "test.dp.states")
    checked = 0
    for _ in range(40):
        i = int(g.integers(60, 181))  # interior positions
        j = int(np.argmin(np.abs(res.mu_grid - g.uniform(-1.5, 1.5) * sd)))
        xp, mu = res.x_grid[i], res.mu_grid[j]
        want = (1 - s.a) * xp + s.a * s.aim * mu
        got = res.x_grid[res.policy_idx[i, j]]
        assert abs(got - want) <= step + 1e-12, (xp, mu, got, want)
        checked += 1
    assert checked >= 10


def test_controller_h60_reproduces_gp_first_action():
    s = gp_closed_form(P)
    g = stream(2, "test.mpc.gp")
    sd = stationary_sd(P)
    for _ in range(12):
        xp, mu = float(g.uniform(-0.3, 0.3)), float(g.normal(0, sd))
        want = (1 - s.a) * xp + s.a * s.aim * mu
        got = mpc_first_action_cvxpy(xp, mu, P.g, P.Lam, P.phi_gp, 0.98, 60)
        assert abs(got - want) <= 1e-6 * abs(want), (got, want)
        la = mpc_first_action_la(xp, mu, P.g, P.Lam, P.phi_gp, 0.98, 60)
        assert abs(la - want) <= 1e-9 * abs(want)


def test_linear_algebra_controller_matches_cvxpy():
    g = stream(3, "test.mpc.la")
    for _ in range(20):
        H = int(g.integers(1, 21))
        phi = float(g.uniform(0.05, 0.7))
        lam = float(g.uniform(0.1, 1.6))
        xp, mu = float(g.uniform(-0.3, 0.3)), float(g.normal(0, 0.014))
        a = mpc_first_action_la(xp, mu, P.g, lam, phi, 0.98, H)
        b = mpc_first_action_cvxpy(xp, mu, P.g, lam, phi, 0.98, H)
        assert abs(a - b) <= 1e-8 * max(1.0, abs(a)), (H, a, b)


def test_controller_h1_equals_mean_variance():
    from dynamic_trading_engine.optimization.testing import random_forecast, random_state

    g = stream(4, "test.mpc.h1")
    st = random_state(g, 12)
    cols = np.arange(12)
    risk = RiskModel("lw").estimate(st, cols)
    fc = random_forecast(g, 12)
    mpc = Optimizer(OptimizerConfig("mpc", gamma=5.0, H=1), "LS", CostConfig(), 5)
    mv = Optimizer(OptimizerConfig("mean_variance", gamma=5.0), "LS", CostConfig(), 5)
    a, b = mpc.solve(st, cols, fc, risk), mv.solve(st, cols, fc, risk)
    assert a.problem_key == b.problem_key  # routed to the same problem object
    assert np.array_equal(a.weights, b.weights)
    # and the H = 1 coefficients of the scalar controller are the one-period solution
    al, be = mpc_coefficients(P.g, P.Lam, P.phi_gp, 0.98, 1)
    assert al == pytest.approx(P.Lam / (P.g + P.Lam)) and be == pytest.approx(1 / (P.g + P.Lam))


@pytest.fixture(scope="module")
def dp_prop():
    sd = stationary_sd(P)
    out = {}
    for c in (1e-7, 1e-4, 4e-4, 1.6e-3):
        out[c] = solve_dp(
            P.g, P.phi_gp, P.sigma_eps, P.delta, "proportional", c, 0.8, 241, 5 * sd, 61
        )
    return out


def test_no_trade_band_properties(dp_prop):
    res = dp_prop[4e-4]
    step = res.x_grid[1] - res.x_grid[0]
    mid = len(res.mu_grid) // 2
    for j in range(mid - 15, mid + 16, 3):
        lo, hi = no_trade_band(res, j)
        pol = res.policy()[:, j]
        x = res.x_grid
        inside = (x >= lo) & (x <= hi)
        assert np.all(pol[inside] == x[inside])
        assert np.all(np.abs(pol[(x < lo)] - lo) <= step + 1e-12)
        assert np.all(np.abs(pol[(x > hi)] - hi) <= step + 1e-12)
        assert np.all(np.diff(pol) >= -1e-12)  # nondecreasing in the held position


def test_band_widens_with_cost_and_closes_as_cost_vanishes(dp_prop):
    mid = len(dp_prop[4e-4].mu_grid) // 2
    for j in range(mid - 10, mid + 11, 5):
        widths = []
        for c in (1e-7, 1e-4, 4e-4, 1.6e-3):
            lo, hi = no_trade_band(dp_prop[c], j)
            widths.append(hi - lo)
        assert all(a <= b + 1e-12 for a, b in zip(widths, widths[1:])), widths
    res = dp_prop[1e-7]
    step = res.x_grid[1] - res.x_grid[0]
    for j in range(mid - 10, mid + 11, 5):
        mu = res.mu_grid[j]
        pol = res.policy()[60:181, j]
        assert np.all(np.abs(pol - mu / P.g) <= step + 1e-12)
