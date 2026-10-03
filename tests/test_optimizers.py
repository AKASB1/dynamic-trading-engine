"""Checks 5 (constraints), 6 (optimality and solver agreement), 7 (closed forms R1-R3),
8 (CVaR and robust, R4 and R5), the solve-A-B-A part of check 15, and the optimizer-loop leg
of the cost parity of check 3."""

import dataclasses
import math

import cvxpy as cp
import numpy as np
import pytest
from scipy.optimize import minimize

from dynamic_trading_engine.contracts.costs import (
    CostConfig,
    ImpactConfig,
    convex_coefficients,
    fill_cost,
)
from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.optimization import problems as PR
from dynamic_trading_engine.optimization.optimizers import (
    Optimizer,
    OptimizerConfig,
    cvar_scenarios,
)
from dynamic_trading_engine.optimization.problems import SCALE, cost_expression, get_problem
from dynamic_trading_engine.optimization.testing import random_forecast, random_state
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

COSTS = CostConfig()
SOLVED = ("mean_variance", "min_variance", "cvar", "robust_box", "robust_ell", "mpc")


def _problem(g, n, book, extras_kind):
    if book == "LO":
        w0 = np.full(n, 1.0 / n)
    else:
        w0 = np.zeros(n)
    st = random_state(g, n, w0=w0, missing=int(g.integers(0, 2)))
    cols = np.nonzero(np.asarray(st.n_known) >= 60)[0]
    if book == "LO":
        # w0 must be feasible: equal weight over the optimized instruments
        w0 = np.zeros(n)
        w0[cols] = 1.0 / len(cols)
        st = dataclasses.replace(st, portfolio=dataclasses.replace(st.portfolio, weights=w0))
    ex = {}
    if extras_kind in ("all", "turn"):
        ex["turnover"] = float(g.uniform(0.05, 1.5))
    if extras_kind in ("all", "secbeta"):
        ex["sector_max"] = 1.0 if book == "LO" else float(g.uniform(0.05, 0.3))
        R = np.asarray(st.returns)[-200:, cols]
        from dynamic_trading_engine.optimization.optimizers import market_beta

        bw = float(market_beta(R) @ w0[cols])
        ex["beta_band"] = (bw - 0.1, bw + 0.1)
    if extras_kind == "all":
        ex["participation"] = float(g.uniform(0.01, 0.2))
    return st, cols, ex


def _opt(name, book, g, ex, **kw):
    cfg = OptimizerConfig(
        name=name,
        gamma=float(math.exp(g.uniform(math.log(0.5), math.log(200)))),
        cost_scale=float(g.uniform(0.25, 4)),
        eta_cvar=float(math.exp(g.uniform(math.log(0.002), math.log(0.5)))),
        kappa_rob=float(g.uniform(0.1, 3)),
        H=3,
        **ex,
        **kw,
    )
    return Optimizer(cfg, book, COSTS, 5)


@pytest.mark.parametrize("block", range(4))
def test_constraints_hold_on_random_problems(block):
    """Check 5: 200 random problems (4 blocks of 50), every optimizer, every constraint to 1e-6."""
    g = stream(block, "test.check5")
    counts = {"optimal": 0, "optimal_inaccurate": 0, "failed": 0}
    for _ in range(50):
        n = int(g.integers(3, 31))
        book = "LO" if g.random() < 0.5 else "LS"
        exk = ("none", "all", "turn", "secbeta")[int(g.integers(0, 4))]
        st, cols, ex = _problem(g, n, book, exk)
        fc = random_forecast(g, n)
        risk = RiskModel("lw").estimate(st, cols)
        for name in SOLVED:
            opt = _opt(name, book, g, ex)
            sol = opt.solve(st, cols, fc, risk)
            counts[sol.status] = counts.get(sol.status, 0) + 1
            assert sol.status in ("optimal", "optimal_inaccurate"), (
                name,
                book,
                exk,
                sol.raw_status,
            )
            assert sol.max_violation <= 1e-6
            assert np.all(
                sol.weights[np.setdiff1d(np.arange(n), cols)] == pytest.approx(0, abs=1e-6)
            )
    for key in PR._CACHE:
        assert PR._CACHE[key].problem.is_dpp()
    print("statuses", counts)


def _independent_violation(w, st, cols, book, ex):
    """Every constraint of section 3 recomputed from the weights and the state (not the
    optimizer's own measurement)."""
    w = np.asarray(w, float)
    n = len(w)
    w0 = np.asarray(st.portfolio.weights, float)
    out = [np.max(np.abs(np.delete(w, cols)))] if len(cols) < n else [0.0]
    wc = w[cols]
    if book == "LO":
        hi = max(0.10, 2.0 / len(cols))
        out += [abs(w.sum() - 1.0), float(np.max(-wc)), float(np.max(wc - hi))]
    else:
        out += [
            float(np.max(np.abs(wc))) - 0.10,
            float(np.sum(np.abs(w))) - 2.0,
            abs(w.sum()) - 0.02,
        ]
    z = w - w0
    if "turnover" in ex:
        out.append(float(np.sum(np.abs(z))) - ex["turnover"])
    if "sector_max" in ex:
        for s in sorted(set(st.sector)):
            idx = [i for i in cols if st.sector[i] == s]
            out.append(abs(float(np.sum(w[idx]))) - ex["sector_max"])
    if "beta_band" in ex:
        from dynamic_trading_engine.optimization.optimizers import market_beta
        from dynamic_trading_engine.state.view import common_window

        h = common_window(np.asarray(st.returns), cols, 250)
        b = market_beta(np.asarray(st.returns)[-h:, cols])
        bw = float(b @ wc)
        out += [ex["beta_band"][0] - bw, bw - ex["beta_band"][1]]
    if "participation" in ex:
        lim = (
            ex["participation"]
            * np.asarray(st.adv)
            * np.asarray(st.last_close)
            / st.portfolio.equity
        )
        out.append(float(np.max(np.abs(z) - lim)))
    return max(0.0, max(out))


def test_non_finite_weights_fail():
    from dynamic_trading_engine.optimization.optimizers import Solution

    g = stream(2, "test.nan")
    st = random_state(g, 6)
    cols = np.arange(6)
    opt = Optimizer(OptimizerConfig("mean_variance"), "LS", COSTS, 5)
    risk = RiskModel("sample").estimate(st, cols)
    sol = opt.solve(st, cols, random_forecast(g, 6), risk)
    assert isinstance(sol, Solution) and sol.success
    w = np.full(PR.ladder_size(6), np.nan)
    prob = get_problem("mv", PR.ladder_size(6), "LS")
    orig = prob.solve
    prob.solve = lambda *a, **k: PR.RawSolution(w, "optimal", 0.0, 1, "CLARABEL")
    try:
        bad = opt.solve(st, cols, random_forecast(g, 6), risk)
    finally:
        prob.solve = orig
    assert bad.status == "failed" and bad.weights is None


def test_mpc_h1_routes_to_mean_variance_for_any_delta():
    # the discount of a routed controller does not create another problem object
    a = get_problem("mv", 8, "LS", "power15", (False,) * 4, 1, 0.9)
    b = get_problem("mv", 8, "LS", "power15", (False,) * 4, 1, 1.0)
    assert a is b
    g = stream(3, "test.route.delta")
    st = random_state(g, 6)
    cols = np.arange(6)
    risk = RiskModel("lw").estimate(st, cols)
    fc = random_forecast(g, 6)
    x = Optimizer(OptimizerConfig("mpc", gamma=5.0, H=1, delta=0.9), "LS", COSTS, 5).solve(
        st, cols, fc, risk
    )
    y = Optimizer(OptimizerConfig("mean_variance", gamma=5.0), "LS", COSTS, 5).solve(
        st, cols, fc, risk
    )
    assert np.array_equal(x.weights, y.weights)


def test_cash_reference_holds_nothing():
    g = stream(4, "test.cash")
    st = random_state(g, 6)
    sol = Optimizer(OptimizerConfig("cash"), "LS", COSTS, 5).solve(st, np.arange(6), None, None)
    assert sol.success and not sol.weights.any()


def test_infeasible_problem_is_a_flagged_hold():
    g = stream(1, "test.infeasible")
    st = random_state(g, 6)
    cols = np.arange(6)
    risk = RiskModel("sample").estimate(st, cols)
    # long-only fully invested from cash with zero turnover allowed: infeasible
    opt = Optimizer(OptimizerConfig("mean_variance", turnover=0.0), "LO", COSTS, 5)
    sol = opt.solve(st, cols, random_forecast(g, 6), risk)
    assert sol.status == "failed" and sol.weights is None and not sol.success


class FailOnce(PipelineStrategy):
    def __init__(self, *a, fail_at=2, **kw):
        super().__init__(*a, **kw)
        self.n = 0
        self.fail_at = fail_at
        self.ok_opt = self.optimizer
        cfg = dataclasses.replace(self.spec.optimizer, turnover=0.0)
        self.bad_opt = Optimizer(cfg, "LO", COSTS, 5)

    def decide(self, state):
        self.optimizer = self.bad_opt if self.n == self.fail_at else self.ok_opt
        self.n += 1
        return super().decide(state)


def test_run_with_one_failed_decision_completes():
    m, _ = generate(load_market_config("tiny"), 1)
    spec = StrategySpec("F", forecast="none", optimizer=OptimizerConfig("min_variance"), book="LO")
    st = FailOnce(spec, COSTS, 5)
    res = run_loop(m, st, COSTS, LoopConfig(warmup_bars=100))
    assert res.status_counts["failed"] == 1 and res.n_holds == 1
    assert res.n_decisions == len(res.decision_bars)


# ---------------------------------------------------------------- check 6


def _kkt_case(g, n):
    st = random_state(g, n, w0=np.full(n, 1.0 / n))
    cols = np.arange(n)
    costs = CostConfig(half_spread_bps=0.0, commission_bps=0.0, impact=ImpactConfig("linear", 0.5))
    fc = random_forecast(g, n, 5e-4)
    risk = RiskModel("sample").estimate(st, cols)
    opt = Optimizer(OptimizerConfig("mean_variance", gamma=float(g.uniform(1, 50))), "LO", costs, 5)
    return st, cols, fc, risk, opt


def test_kkt_conditions_quadratic_program():
    g = stream(2, "test.kkt")
    for _ in range(20):
        n = int(g.integers(3, 12))
        st, cols, fc, risk, opt = _kkt_case(g, n)
        sol = opt.solve(st, cols, fc, risk)
        assert sol.status == "optimal"
        prob = get_problem(*_key(opt, st))
        P = prob.P
        w = prob.ws[0].value
        mu, F, cb = P["mu"][0].value, P["F"].value, P["cb"].value
        w0 = P["w0"].value
        grad = mu - 2 * F.T @ (F @ w) - 2 * cb * (w - w0)
        lam_lo = prob.named["lo"].dual_value
        lam_hi = prob.named["hi"].dual_value
        r = grad + lam_lo - lam_hi
        act = np.arange(n)
        assert np.all(lam_lo >= -1e-6) and np.all(lam_hi >= -1e-6)
        assert np.max(np.abs(r[act] - np.mean(r[act]))) <= 1e-6 * max(1.0, np.max(np.abs(grad)))
        assert np.max(np.abs(lam_lo * (w - P["lo"].value))) <= 1e-6
        assert np.max(np.abs(lam_hi * (P["hi"].value - w))) <= 1e-6


def _key(opt, st):
    return ("mv", PR.ladder_size(st.n), opt.book_name, opt.cost_form, (False,) * 4, 1, 1.0)


SLSQP_FEASIBLE = 1e-8  # largest constraint violation (weights) of an SLSQP point that is checked


def _slsqp(prob, cost_form, book):
    """Independent solver: SciPy SLSQP on the variables (w+, w-, z+, z-), all >= 0, of the problem
    as CVXPY last solved it. Returns ``run(shift)``, which starts SLSQP at the CVXPY solution split
    into its parts plus ``shift`` and returns the objective of the point SLSQP stops at (scaled
    units) and that point's largest constraint violation, computed here from the constraints (the
    SLSQP status is not trusted). With ``za = z+ + z- >= |z|`` and ``w+ + w- >= |w|`` every feasible
    point maps to feasible weights whose objective is at least as high, so no feasible SLSQP point
    can beat the optimum of the convex problem."""
    P = prob.P
    N = prob.N
    mu, F, ca, cb, w0 = (
        P["mu"][0].value,
        P["F"].value,
        P["ca"].value,
        P["cb"].value,
        P["w0"].value,
    )
    lo, hi = P["lo"].value, P["hi"].value
    Q = F.T @ F

    def split(x):
        return x[:N], x[N : 2 * N], x[2 * N : 3 * N], x[3 * N :]

    def f(x):
        wp, wm, zp, zm = split(x)
        w = wp - wm
        za = zp + zm
        c = ca @ za + (cb @ za**1.5 if cost_form == "power15" else cb @ za**2)
        return -(mu @ w - w @ Q @ w - c)

    def jac(x):
        wp, wm, zp, zm = split(x)
        w = wp - wm
        za = zp + zm
        gw = mu - 2 * Q @ w
        gz = ca + (1.5 * cb * np.sqrt(za) if cost_form == "power15" else 2 * cb * za)
        return -np.concatenate([gw, -gw, -gz, -gz])

    cons = [
        {
            "type": "eq",
            "fun": lambda x: (split(x)[0] - split(x)[1]) - w0 - (split(x)[2] - split(x)[3]),
        },
        {"type": "ineq", "fun": lambda x: (split(x)[0] - split(x)[1]) - lo},
        {"type": "ineq", "fun": lambda x: hi - (split(x)[0] - split(x)[1])},
    ]
    if book.budget == "eq":
        cons.append({"type": "eq", "fun": lambda x: np.sum(split(x)[0] - split(x)[1]) - 1.0})
    if book.gross is not None:
        cons.append(
            {"type": "ineq", "fun": lambda x: book.gross - np.sum(split(x)[0] + split(x)[1])}
        )
    if book.net is not None:
        cons.append(
            {"type": "ineq", "fun": lambda x: np.sum(split(x)[0] - split(x)[1]) - book.net[0]}
        )
        cons.append(
            {"type": "ineq", "fun": lambda x: book.net[1] - np.sum(split(x)[0] - split(x)[1])}
        )
    w_cv = np.array(prob.ws[0].value, float)
    z_cv = w_cv - w0
    x_cv = np.concatenate(
        [np.maximum(w_cv, 0), np.maximum(-w_cv, 0), np.maximum(z_cv, 0), np.maximum(-z_cv, 0)]
    )

    def violation(x):
        out = float(np.max(-x, initial=0.0))
        for c in cons:
            r = np.atleast_1d(c["fun"](x))
            out = max(out, float(np.max(np.abs(r) if c["type"] == "eq" else -r, initial=0.0)))
        return out

    def run(shift):
        res = minimize(
            f,
            x_cv + shift,
            jac=jac,
            method="SLSQP",
            bounds=[(0, None)] * (4 * N),
            constraints=cons,
            options={"ftol": 1e-15, "maxiter": 2000},
        )
        return -float(f(res.x)), violation(res.x)

    return run


@pytest.mark.parametrize("cost_form", ["power15", "quadratic"])
def test_solver_agreement(cost_form):
    """50 random problems of at most 10 instruments, Clarabel at 1e-10, tolerance
    1e-6 * max(1, |objective|) in the scaled units:
    (a) SciPy SLSQP started away from the optimum: no point it returns that is feasible to 1e-8
        (measured here) has a higher objective than Clarabel's optimum (a local solver may stall
        short of the optimum, so only this side is a check);
    (b) SLSQP started at Clarabel's solution finds no improvement (local optimality, which for a
        convex problem is global optimality);
    (c) a second CVXPY solver (OSQP for the quadratic form, SCS for both) agrees on the optimal
        objective in both directions."""
    g = stream(3, f"test.agree.{cost_form}")
    impact = ImpactConfig("sqrt" if cost_form == "power15" else "linear", 0.5)
    costs = CostConfig(impact=impact)
    worst = {"far_above": 0.0, "far_below": 0.0, "local_gain": 0.0, "second": 0.0, "viol": 0.0}
    checked = {"far": 0, "local": 0}
    for _ in range(50):
        n = int(g.integers(2, 11))
        book = "LS" if g.random() < 0.5 else "LO"
        w0 = np.full(n, 1.0 / n) if book == "LO" else np.zeros(n)
        st = random_state(g, n, w0=w0)
        cols = np.arange(n)
        fc = random_forecast(g, n, 5e-4)
        risk = RiskModel("lw").estimate(st, cols)
        opt = Optimizer(
            OptimizerConfig("mean_variance", gamma=float(g.uniform(1, 30)), cost_scale=1.0),
            book,
            costs,
            5,
            solver_opts=PR.CLARABEL_TIGHT,
        )
        sol = opt.solve(st, cols, fc, risk)
        assert sol.status == "optimal"
        v1 = sol.objective * SCALE
        prob = get_problem("mv", PR.ladder_size(n), book, opt.cost_form)
        run = _slsqp(prob, opt.cost_form, PR.BOOKS[book])
        scale = max(1.0, abs(v1))
        tol = 1e-6 * scale
        v_far, viol_far = run(1e-3)  # (a) start away from the CVXPY optimum
        v_loc, viol_loc = run(0.0)  # (b) start at the CVXPY optimum
        worst["viol"] = max(worst["viol"], viol_far, viol_loc)
        if viol_far <= SLSQP_FEASIBLE:
            checked["far"] += 1
            worst["far_above"] = max(worst["far_above"], (v_far - v1) / scale)
            worst["far_below"] = max(worst["far_below"], (v1 - v_far) / scale)
            assert v_far <= v1 + tol, ("feasible SLSQP point beats the optimum", v1, v_far)
        if viol_loc <= SLSQP_FEASIBLE:
            checked["local"] += 1
            worst["local_gain"] = max(worst["local_gain"], (v_loc - v1) / scale)
            assert v_loc <= v1 + tol, ("SLSQP improves on the CVXPY solution", v1, v_loc)
        if cost_form == "quadratic":
            raw = prob.solve(
                "OSQP", {"eps_abs": 1e-10, "eps_rel": 1e-10, "polish": True, "max_iter": 400000}
            )
        else:
            raw = prob.solve("SCS", {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iters": 400000})
        v2 = raw.value * SCALE
        worst["second"] = max(worst["second"], abs(v1 - v2) / scale)
        assert abs(v1 - v2) <= tol, (v1, v2, raw.solver)
    # the SLSQP checks must not become vacuous: (nearly) every SLSQP point is feasible to 1e-8
    assert checked["far"] >= 45 and checked["local"] >= 45, checked
    print("checked", checked, "worst", worst)


# ---------------------------------------------------------------- check 7: closed forms


def _ref_state(g, n):
    st = random_state(g, n)
    cols = np.arange(n)
    return st, cols, RiskModel("sample").estimate(st, cols)


def test_r1_mean_variance_budget_only():
    g = stream(4, "test.r1")
    for _ in range(20):
        n = int(g.integers(2, 25))
        st, cols, risk = _ref_state(g, n)
        gamma = float(g.uniform(1, 50))
        fc = Forecast(g.standard_normal(n) * 5e-4, np.zeros(n), 0.0, False, "t")
        opt = Optimizer(
            OptimizerConfig("mean_variance", gamma=gamma, cost_scale=0.0), "BUDGET", COSTS, 1
        )
        sol = opt.solve(st, cols, fc, risk)
        Si = np.linalg.inv(risk.sigma)
        one = np.ones(n)
        nu = (one @ Si @ fc.mu - gamma) / (one @ Si @ one)
        want = Si @ (fc.mu - nu * one) / gamma
        assert np.max(np.abs(sol.weights - want)) <= 1e-6 * max(1.0, np.max(np.abs(want)))


def test_r2_min_variance_budget_only():
    g = stream(5, "test.r2")
    for _ in range(20):
        n = int(g.integers(2, 25))
        st, cols, risk = _ref_state(g, n)
        opt = Optimizer(OptimizerConfig("min_variance"), "BUDGET", COSTS, 1)
        sol = opt.solve(st, cols, None, risk)
        Si = np.linalg.inv(risk.sigma)
        want = Si @ np.ones(n) / (np.ones(n) @ Si @ np.ones(n))
        assert np.max(np.abs(sol.weights - want)) <= 1e-6 * max(1.0, np.max(np.abs(want)))


def test_r3_quadratic_costs_unconstrained():
    g = stream(6, "test.r3")
    costs = CostConfig(half_spread_bps=0.0, commission_bps=0.0, impact=ImpactConfig("linear", 0.5))
    for _ in range(20):
        n = int(g.integers(2, 25))
        w0 = g.uniform(-0.1, 0.1, n)
        st = random_state(g, n, w0=w0)
        cols = np.arange(n)
        risk = RiskModel("sample").estimate(st, cols)
        gamma = float(g.uniform(1, 50))
        fc = Forecast(g.standard_normal(n) * 5e-4, np.zeros(n), 0.0, False, "t")
        opt = Optimizer(OptimizerConfig("mean_variance", gamma=gamma), "UNC", costs, 1)
        sol = opt.solve(st, cols, fc, risk)
        a, l, form = convex_coefficients(
            np.asarray(st.last_close),
            np.asarray(st.sigma_bar),
            np.asarray(st.adv),
            st.portfolio.equity,
            costs,
        )
        assert form == "quadratic" and np.all(a == 0)
        Lq = np.diag(l)
        want = np.linalg.solve(
            gamma * risk.sigma + Lq, fc.mu + Lq @ np.asarray(st.portfolio.weights)
        )
        assert np.max(np.abs(sol.weights - want)) <= 1e-6 * max(1.0, np.max(np.abs(want)))


# ---------------------------------------------------------------- check 8: CVaR, robust


def cvar_definition(losses, alpha):
    """Mean of the worst (1 - alpha) S losses with the fractional weight at the boundary."""
    S = len(losses)
    tail = (1.0 - alpha) * S
    srt = np.sort(losses)[::-1]
    k = int(math.floor(tail))
    total = srt[:k].sum() + (tail - k) * (srt[k] if k < S else 0.0)
    return total / tail


def cvar_ru(losses, alpha):
    """Rockafellar-Uryasev: min over zeta of zeta + E[(loss - zeta)+] / (1 - alpha); the minimum
    is attained at a loss value."""
    S = len(losses)
    return min(z + np.maximum(losses - z, 0).sum() / ((1 - alpha) * S) for z in losses)


def test_r4_cvar_value_equals_definition_and_monotone():
    g = stream(7, "test.r4")
    for _ in range(50):
        S = int(g.integers(5, 200))
        losses = g.standard_normal(S)
        prev = -np.inf
        for alpha in (0.5, 0.8, 0.9, 0.95, 0.975):
            a, b = cvar_ru(losses, alpha), cvar_definition(losses, alpha)
            assert abs(a - b) <= 1e-12 * max(1.0, abs(b))
            assert b >= prev - 1e-12
            prev = b


@pytest.mark.parametrize("n", [2, 3])
def test_r4_cvar_optimum_equals_grid_search(n):
    g = stream(8 + n, "test.r4grid")
    zero = CostConfig(half_spread_bps=0.0, commission_bps=0.0, impact=ImpactConfig("none"))
    for _ in range(3):
        st = random_state(g, n, L=120, w0=np.full(n, 1.0 / n))
        cols = np.arange(n)
        mu = g.standard_normal(n) * 2e-3
        fc = Forecast(mu, np.zeros(n), 0.0, False, "t")
        eta, alpha, k = 0.2, 0.9, 1
        opt = Optimizer(
            OptimizerConfig("cvar", eta_cvar=eta, alpha=alpha, cvar_window=120), "LO", zero, k
        )
        sol = opt.solve(st, cols, fc, None)
        sc = cvar_scenarios(np.asarray(st.returns)[-120:], k)
        hi = max(0.10, 2.0 / n)

        def val(w):
            return mu @ w - eta * cvar_definition(-(sc @ w), alpha)

        step = 0.002 if n == 2 else 0.01
        best = -np.inf
        grid = np.arange(0, hi + 1e-12, step)
        if n == 2:
            for a in grid:
                w = np.array([a, 1 - a])
                if w.min() >= -1e-12 and w.max() <= hi + 1e-12:
                    best = max(best, val(w))
        else:
            for a in grid:
                for b in grid:
                    w = np.array([a, b, 1 - a - b])
                    if w.min() >= -1e-12 and w.max() <= hi + 1e-12:
                        best = max(best, val(w))
        v_opt = val(sol.weights)
        assert v_opt >= best - 1e-9  # the optimizer is at least as good as the grid
        assert abs(sol.objective - v_opt) <= 1e-6 * max(1.0, abs(v_opt))
        assert v_opt - best <= 5e-3 * abs(best) + 1e-6  # and the grid comes close


def _robust_setup(g, n):
    st = random_state(g, n)
    cols = np.arange(n)
    risk = RiskModel("lw").estimate(st, cols)
    fc = random_forecast(g, n, 5e-4)
    return st, cols, risk, fc


def test_r5_kappa_zero_routes_to_mean_variance():
    g = stream(9, "test.r5route")
    st, cols, risk, fc = _robust_setup(g, 12)
    rob = Optimizer(OptimizerConfig("robust_box", gamma=5.0, kappa_rob=0.0), "LS", COSTS, 5)
    mv = Optimizer(OptimizerConfig("mean_variance", gamma=5.0), "LS", COSTS, 5)
    a, b = rob.solve(st, cols, fc, risk), mv.solve(st, cols, fc, risk)
    assert a.problem_key == b.problem_key and a.problem_key[0] == "mv"
    assert np.array_equal(a.weights, b.weights)
    # the robust problem object itself with kse = 0 agrees to the solver tolerance
    rob1 = Optimizer(OptimizerConfig("robust_box", gamma=5.0, kappa_rob=1e-300), "LS", COSTS, 5)
    c = rob1.solve(st, cols, fc, risk)
    assert c.problem_key[0] == "robust_box"
    assert np.max(np.abs(c.weights - b.weights)) <= 1e-4
    assert abs(c.objective - b.objective) * SCALE <= 1e-6 * max(1.0, abs(b.objective * SCALE))


def test_r5_robust_objective_and_worst_case_optimality():
    g = stream(10, "test.r5")
    for kind in ("robust_box", "robust_ell"):
        for _ in range(5):
            n = int(g.integers(4, 15))
            st, cols, risk, fc = _robust_setup(g, n)
            kap = float(g.uniform(0.5, 2.5))
            opt = Optimizer(OptimizerConfig(kind, gamma=5.0, kappa_rob=kap), "LS", COSTS, 5)
            sol = opt.solve(st, cols, fc, risk)
            prob = get_problem(kind, PR.ladder_size(n), "LS")
            P = prob.P
            mu, F, ca, cb, w0, kse = (
                P["mu"][0].value,
                P["F"].value,
                P["ca"].value,
                P["cb"].value,
                P["w0"].value,
                P["kse"].value,
            )

            def nominal(w):
                z = np.abs(w - w0)
                return mu @ w - np.sum((F @ w) ** 2) - ca @ z - cb @ z**1.5

            def pen(w):
                return kse @ np.abs(w) if kind == "robust_box" else np.linalg.norm(kse * w)

            w = np.zeros(prob.N)
            w[:n] = sol.weights
            v = sol.objective * SCALE
            assert abs(v - (nominal(w) - pen(w))) <= 1e-6 * max(1.0, abs(v))
            # no feasible weight vector has a better worst case (closed-form inner minimum)
            for _ in range(200):
                u = g.standard_normal(prob.N) * g.uniform(0.01, 0.1)
                u[n:] = 0
                u[:n] -= u[:n].mean()
                u = np.clip(u, -0.1, 0.1)
                if np.sum(np.abs(u)) > 2 or abs(u.sum()) > 0.02:
                    continue
                for lam in (0.1, 0.5, 1.0):
                    x = (1 - lam) * w + lam * u
                    assert nominal(x) - pen(x) <= v + 1e-6 * max(1.0, abs(v))


def test_robust_weighted_norm_nonincreasing_in_kappa():
    g = stream(11, "test.r5mono")
    for kind in ("robust_box", "robust_ell"):
        st, cols, risk, fc = _robust_setup(g, 15)
        se_h = np.asarray(fc.se) * (1 - fc.decay**5) / (1 - fc.decay)
        prev = np.inf
        for kap in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            sol = Optimizer(OptimizerConfig(kind, gamma=5.0, kappa_rob=kap), "LS", COSTS, 5).solve(
                st, cols, fc, risk
            )
            w = sol.weights
            nrm = np.sum(se_h * np.abs(w)) if kind == "robust_box" else np.linalg.norm(se_h * w)
            assert nrm <= prev + 1e-4 * max(1.0, prev if np.isfinite(prev) else 1.0)
            prev = nrm


# ---------------------------------------------------------------- check 15: A-B-A


@pytest.mark.parametrize("name", SOLVED)
@pytest.mark.parametrize("book", ["LO", "LS"])
def test_solve_a_b_a_identical_bits(name, book):
    g = stream(12, f"test.aba.{name}.{book}")
    n = 10
    w0 = np.full(n, 0.1) if book == "LO" else np.zeros(n)
    sa = random_state(g, n, w0=w0)
    sb = random_state(g, n, w0=w0)
    cols = np.arange(n)
    fa, fb = random_forecast(g, n), random_forecast(g, n)
    ra, rb = RiskModel("lw").estimate(sa, cols), RiskModel("lw").estimate(sb, cols)
    opt = Optimizer(OptimizerConfig(name, gamma=5.0, eta_cvar=0.02, H=3), book, COSTS, 5)
    a1 = opt.solve(sa, cols, fa, ra)
    opt.solve(sb, cols, fb, rb)
    a2 = opt.solve(sa, cols, fa, ra)
    assert a1.problem_key == a2.problem_key
    assert np.array_equal(a1.weights, a2.weights)
    assert a1.objective == a2.objective


# ---------------------------------------------------------------- check 3: optimizer-loop parity


def test_cost_parity_linear_impact():
    """The quadratic cost form (l = 2 y sigma E / (V P)) against the loop's linear impact."""
    g = stream(14, "test.parity.linear")
    costs = CostConfig(impact=ImpactConfig("linear", 0.5), commission_per_share=0.002)
    zv = cp.Variable(1)
    ca, cb = cp.Parameter(1, nonneg=True), cp.Parameter(1, nonneg=True)
    expr = cost_expression(zv, ca, cb, "quadratic")
    for _ in range(200):
        E = float(10 ** g.uniform(5, 9))
        P = float(g.uniform(5, 500))
        sig = float(g.uniform(0.005, 0.05))
        V = float(10 ** g.uniform(4, 7))
        z = float(g.uniform(-0.2, 0.2))
        a, l, form = convex_coefficients(np.array([P]), np.array([sig]), np.array([V]), E, costs)
        assert form == "quadratic"
        ca.value, cb.value = a, 0.5 * l
        zv.value = np.array([z])
        fc = fill_cost(z * E / P, P, sig, V, costs)
        loop_cost = fc.spread_cost + fc.impact_cost + fc.commission
        assert abs(E * float(expr.value) - loop_cost) <= 1e-10 * max(abs(loop_cost), 1e-300)


def test_cost_parity_optimizer_and_loop():
    """200 random trades, noise off, no cap, min_commission 0, no lots: the optimizer's E * c(z)
    (the CVXPY expression evaluated at z, cost_scale 1) equals the loop's fill cost at the
    reference price P to 1e-10 relative."""
    g = stream(13, "test.parity")
    costs = CostConfig(commission_per_share=0.003)
    zv = cp.Variable(1)
    ca, cb = cp.Parameter(1, nonneg=True), cp.Parameter(1, nonneg=True)
    expr = cost_expression(zv, ca, cb, "power15")
    for _ in range(200):
        E = float(10 ** g.uniform(5, 9))
        P = float(g.uniform(5, 500))
        sig = float(g.uniform(0.005, 0.05))
        V = float(10 ** g.uniform(4, 7))
        z = float(g.uniform(-0.2, 0.2))
        a, b, form = convex_coefficients(np.array([P]), np.array([sig]), np.array([V]), E, costs)
        ca.value, cb.value = a, b
        zv.value = np.array([z])
        opt_cost = E * float(expr.value)
        fc = fill_cost(z * E / P, P, sig, V, costs)
        loop_cost = fc.spread_cost + fc.impact_cost + fc.commission
        assert abs(opt_cost - loop_cost) <= 1e-10 * max(abs(loop_cost), 1e-300)
