"""Tier-2 checks: the Wasserstein-robust mean-CVaR (eps = 0, the linear-loss closed form,
monotonicity in eps), the multi-asset Garleanu-Pedersen rule against the scalar reference, the
adaptive execution policy, and the export consumer."""

import dataclasses

import cvxpy as cp
import numpy as np
import pytest

from dynamic_trading_engine.contracts.costs import CostConfig
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization import problems as PR
from dynamic_trading_engine.optimization.optimizers import Optimizer, OptimizerConfig
from dynamic_trading_engine.optimization.testing import random_forecast, random_state
from dynamic_trading_engine.policies.gp import GPParams, gp_closed_form
from dynamic_trading_engine.policies.gp_aim import gp_target
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.rng import stream

COSTS = CostConfig()


def _setup(g, n=8):
    st = random_state(g, n, L=200)
    cols = np.arange(n)
    fc = random_forecast(g, n, 5e-4)
    fc = Forecast(fc.mu, fc.se, 0.5, False, "t")
    return st, cols, fc


@pytest.mark.parametrize("seed", range(5))
def test_wdro_with_zero_radius_is_the_empirical_mean_cvar(seed):
    """eps = 0: the worst case is the empirical E[-w'xi] + eta CVaR(-w'xi) with xi = mu_h + R, which
    is the cvar optimizer with the forecast multiplied by (1 + eta) (CVaR is translation
    equivariant)."""
    g = stream(seed, "test.wdro.eps0")
    st, cols, fc = _setup(g)
    eta, alpha = 0.01, 0.95
    w = Optimizer(
        OptimizerConfig("wdro", eps_w=0.0, eta_cvar=eta, alpha=alpha), "LS", COSTS, 5
    ).solve(st, cols, fc, None)
    fc2 = Forecast(fc.mu * (1 + eta), fc.se, fc.decay, False, "t")
    c = Optimizer(OptimizerConfig("cvar", eta_cvar=eta, alpha=alpha), "LS", COSTS, 5).solve(
        st, cols, fc2, None
    )
    assert w.success and c.success
    assert np.max(np.abs(w.weights - c.weights)) <= 1e-4
    assert abs(w.objective - c.objective) * PR.SCALE <= 1e-6 * max(1.0, abs(c.objective) * PR.SCALE)


@pytest.mark.parametrize("seed", range(5))
def test_wdro_linear_loss_is_empirical_plus_eps_times_dual_norm(seed):
    """eta = 0 (a linear loss): the worst-case expectation is the empirical one plus
    eps * ||w||_2, so the optimum equals max mu_h'w - eps ||w||_2 - c(z) under the book."""
    g = stream(seed, "test.wdro.lin")
    st, cols, fc = _setup(g)
    eps = 2e-3
    opt = Optimizer(OptimizerConfig("wdro", eps_w=eps, eta_cvar=0.0), "LS", COSTS, 5)
    sol = opt.solve(st, cols, fc, None)
    N = PR.ladder_size(st.n)
    vals, _ = opt._values(st, cols, fc, None, N, "cvar", opt.cfg.extras())
    w = cp.Variable(N)
    z = w - vals["w0"]
    obj = (
        vals["mu"][0] @ w
        - PR.SCALE * eps * cp.norm(w, 2)
        - PR.cost_expression(z, vals["ca"], vals["cb"], "power15")
    )
    bk = PR.BOOKS["LS"]
    cons = [
        w >= vals["lo"],
        w <= vals["hi"],
        cp.sum(cp.abs(w)) <= bk.gross,
        cp.sum(w) >= bk.net[0],
        cp.sum(w) <= bk.net[1],
    ]
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver="CLARABEL", **PR.CLARABEL_TIGHT)
    assert abs(sol.objective * PR.SCALE - prob.value) <= 1e-6 * max(1.0, abs(prob.value))


def test_wdro_objective_monotone_in_eps():
    g = stream(9, "test.wdro.mono")
    st, cols, fc = _setup(g)
    prev = np.inf
    for eps in (0.0, 1e-4, 5e-4, 2e-3, 1e-2):
        sol = Optimizer(OptimizerConfig("wdro", eps_w=eps, eta_cvar=0.01), "LS", COSTS, 5).solve(
            st, cols, fc, None
        )
        v = sol.objective * PR.SCALE
        assert v <= prev + 1e-6 * max(1.0, abs(prev) if np.isfinite(prev) else 1.0)
        prev = v


def test_gp_target_decouples_into_the_scalar_reference():
    """With a diagonal covariance, each instrument follows the scalar rule of R6 with g = gamma,
    Lam = lam_gp, and expected return mu_i / sigma_i (in the coordinates sqrt(k Sigma) x)."""
    g = stream(3, "test.gp.diag")
    n = 6
    var = g.uniform(1e-4, 1e-3, n) * 5
    Sk = np.diag(var)
    mu = g.standard_normal(n) * 1e-3
    x0 = g.uniform(-0.05, 0.05, n)
    gamma, lam, d = 4.0, 2.0, 0.6
    tgt, a, aim = gp_target(mu, Sk, x0, gamma, lam, d, 0.98)
    s = gp_closed_form(GPParams(g=gamma, Lam=lam, rho_gp=0.02, phi_gp=1 - d, sigma_eps=0.0))
    assert a == pytest.approx(s.a) and aim == pytest.approx(s.aim)
    for i in range(n):
        y_prev = np.sqrt(var[i]) * x0[i]
        y = (1 - s.a) * y_prev + s.a * s.aim * mu[i] / np.sqrt(var[i])
        assert tgt[i] == pytest.approx(y / np.sqrt(var[i]), rel=1e-12)


def test_gp_aim_respects_the_book():
    g = stream(4, "test.gp.book")
    st, cols, fc = _setup(g, 12)
    risk = RiskModel("lw").estimate(st, cols)
    sol = Optimizer(OptimizerConfig("gp_aim", gamma=0.5, lam_mult=1.0), "LS", COSTS, 5).solve(
        st, cols, fc, risk
    )
    assert sol.success and sol.max_violation <= 1e-6
    assert np.max(np.abs(sol.weights)) <= 0.10 + 1e-6 and abs(np.sum(sol.weights)) <= 0.02 + 1e-6


def test_dataclass_defaults_keep_tier1_hashes():
    from dynamic_trading_engine.contracts.canonical import canonical_dict

    cfg = OptimizerConfig("mean_variance")
    d = canonical_dict(cfg)
    assert "eps_w" not in d and "lam_mult" not in d and "delta_gp" not in d
    assert dataclasses.replace(cfg, eps_w=0.001) != cfg


# ---------------------------------------------------------------- adaptive execution


def _exec_order(K=10):
    from dynamic_trading_engine.execution.simulator import ParentOrder, u_profile

    adv = 1e6
    return ParentOrder(1, 3e4, 50.0, 0.02, adv, tuple([1.0 / K] * K), u_profile(K, adv, 1.0))


def test_adaptive_ac_is_time_consistent_in_the_exact_world():
    """Without volume information to learn from and with the true linear impact, re-planning the
    remaining Almgren-Chriss schedule every interval reproduces the static schedule."""
    from dynamic_trading_engine.execution.adaptive import AdaptiveACPolicy
    from dynamic_trading_engine.execution.simulator import ExecConfig, ac_policy, draws, simulate

    o = _exec_order()
    cfg = ExecConfig(
        impact="direct",
        eta_ac=2.5e-6,
        gamma_ac=2.5e-7,
        half_spread_bps=0.0,
        commission_bps=0.0,
        participation_cap=None,
        volume_noise_sd=0.0,
    )
    xi, zv = draws(1, 20, o.K)
    a = simulate(o, ac_policy(o, 1e-6, 2.5e-6, 2.5e-7), cfg, xi, zv, record=True)
    b = simulate(o, AdaptiveACPolicy(o, cfg, 1e-6), cfg, xi, zv, record=True)
    assert np.allclose(a["fills"], b["fills"], rtol=1e-9, atol=1e-6)


def test_adaptive_ac_has_no_look_ahead_and_an_exact_decomposition():
    from dynamic_trading_engine.execution.adaptive import AdaptiveACPolicy
    from dynamic_trading_engine.execution.simulator import ExecConfig, draws, simulate

    o = _exec_order()
    cfg = ExecConfig(volume_level_sd=0.5)
    pol = AdaptiveACPolicy(o, cfg, 1e-6)
    g = stream(5, "test.adapt")
    xi, zv = draws(2, 8, o.K)
    zl = g.standard_normal(8)
    a = simulate(o, pol, cfg, xi, zv, record=True, zl=zl)
    total = sum(
        a[k] for k in ("timing", "permanent", "spread", "temporary", "opportunity", "commission")
    )
    assert np.max(np.abs(total - a["IS"])) <= 1e-12 * o.Q * o.S0
    for k in range(1, o.K):
        xi2, zv2 = xi.copy(), zv.copy()
        xi2[:, k:] = g.standard_normal(xi2[:, k:].shape)
        zv2[:, k:] = g.standard_normal(zv2[:, k:].shape)
        b = simulate(o, pol, cfg, xi2, zv2, record=True, zl=zl)
        assert np.array_equal(a["decided"][:, : k + 1], b["decided"][:, : k + 1])


# ---------------------------------------------------------------- fat tails and jumps


def test_student_t_variant_has_fat_tails_and_unit_variance():
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.generator import generate

    k_vals, v_vals = [], []
    for seed in range(1, 6):
        m, tr = generate(load_market_config("base_t5"), seed)
        sd = tr.sig_pre[None, :] * tr.v[1:1260, None]
        z = (tr.e[1:] / sd - 0.02 * tr.s[:-1]) / np.sqrt(1 - 0.02**2)
        z = z.ravel()
        v_vals.append(np.var(z))
        k_vals.append(np.mean((z - z.mean()) ** 4) / np.var(z) ** 2)
    assert abs(np.mean(v_vals) - 1.0) < 0.05
    assert np.mean(k_vals) > 4.0  # Gaussian: 3; Student-t with 5 degrees of freedom: 9


def test_tail_variants_leave_the_shift_pairing_and_the_default_market_intact():
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.generator import generate

    a, _ = generate(load_market_config("base_t5"), 3)
    b, _ = generate(load_market_config("shift_t5"), 3)
    assert np.array_equal(a.close[:630], b.close[:630], equal_nan=True)
    g1, _ = generate(load_market_config("base"), 3)
    j1, _ = generate(load_market_config("base_jump"), 3)
    assert not np.array_equal(g1.returns, j1.returns, equal_nan=True)
    d = np.abs(np.nan_to_num(j1.returns - g1.returns))
    share = np.mean(d > 1e-12)
    assert 0.5 * 5 / 252 < share < 2 * 5 / 252  # about 5 jumps per instrument and year


def test_export_round_trip_runs_the_loop_identically(tmp_path):
    from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.export import ExportError, read_export, write_export
    from dynamic_trading_engine.market.generator import generate
    from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

    m, _ = generate(load_market_config("tiny"), 4)
    write_export(m, str(tmp_path))
    m2, man, _f = read_export(str(tmp_path))
    spec = StrategySpec("EW", forecast="none", optimizer=OptimizerConfig("equal_weight"), book="LO")
    logs = [
        run_loop(
            x, PipelineStrategy(spec, COSTS, 5), COSTS, LoopConfig(warmup_bars=100), write_logs=True
        ).logs
        for x in (m, m2)
    ]
    assert logs[0] == logs[1]
    # a tampered file is refused
    p = tmp_path / "bars.csv"
    data = p.read_bytes()
    p.write_bytes(data.replace(b",1", b",2", 1))
    with pytest.raises(ExportError):
        read_export(str(tmp_path))
