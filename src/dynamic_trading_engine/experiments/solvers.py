"""Tier-2 solver comparison: the optimizer problems on Clarabel, OSQP, SCS, and HiGHS (where the
problem class allows it), against the exact references and a tight Clarabel reference: accuracy,
time per solve, failures. Gurobi is used only if it imports with a licence (skipped otherwise)."""

from __future__ import annotations

import math
import time

import numpy as np

from dynamic_trading_engine.contracts.costs import CostConfig, ImpactConfig, convex_coefficients
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization import problems as PR
from dynamic_trading_engine.optimization.optimizers import Optimizer, OptimizerConfig
from dynamic_trading_engine.optimization.testing import random_forecast, random_state
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.rng import stream

OPTS = {
    "CLARABEL": {"max_threads": 1},
    "OSQP": {"eps_abs": 1e-9, "eps_rel": 1e-9, "polish": True, "max_iter": 200000},
    "SCS": {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 200000},
    "HIGHS": {},
}
# which problem classes each solver accepts (power cones: only Clarabel and SCS); the QP and LP
# cases are built with the quadratic cost form so that their problem objects hold no cone
ACCEPTS = {
    "CLARABEL": {"qp", "power", "lp"},
    "OSQP": {"qp", "lp"},
    "SCS": {"qp", "power", "lp"},
    "HIGHS": {"qp", "lp"},
}


def _gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401

        gurobipy.Model().dispose()
        return True
    except Exception:
        return False


def _cases(g, n):
    """(name, class, optimizer, book, costs, state, cols, forecast, risk, exact weights or None)."""
    out = []
    st = random_state(g, n, w0=np.zeros(n))
    cols = np.arange(n)
    risk = RiskModel("sample").estimate(st, cols)
    gamma = float(g.uniform(2, 30))
    fc = Forecast(g.standard_normal(n) * 5e-4, np.zeros(n), 0.0, False, "t")
    Si = np.linalg.inv(risk.sigma)
    one = np.ones(n)
    nu = (one @ Si @ fc.mu - gamma) / (one @ Si @ one)
    out.append(
        (
            "R1 mean-variance, budget only",
            "qp",
            OptimizerConfig("mean_variance", gamma=gamma, cost_scale=0.0, cost_form="quadratic"),
            "BUDGET",
            CostConfig(),
            1,
            st,
            cols,
            fc,
            risk,
            Si @ (fc.mu - nu * one) / gamma,
        )
    )
    out.append(
        (
            "R2 minimum variance, budget only",
            "qp",
            OptimizerConfig("min_variance", cost_form="quadratic"),
            "BUDGET",
            CostConfig(),
            1,
            st,
            cols,
            None,
            risk,
            Si @ one / (one @ Si @ one),
        )
    )
    w0 = g.uniform(-0.1, 0.1, n)
    st3 = random_state(g, n, w0=w0)
    risk3 = RiskModel("sample").estimate(st3, cols)
    qc = CostConfig(half_spread_bps=0.0, commission_bps=0.0, impact=ImpactConfig("linear", 0.5))
    _a, l, _f = convex_coefficients(
        np.asarray(st3.last_close),
        np.asarray(st3.sigma_bar),
        np.asarray(st3.adv),
        st3.portfolio.equity,
        qc,
    )
    w3 = np.linalg.solve(gamma * risk3.sigma + np.diag(l), fc.mu + np.diag(l) @ w0)
    out.append(
        (
            "R3 quadratic costs, unconstrained",
            "qp",
            OptimizerConfig("mean_variance", gamma=gamma),
            "UNC",
            qc,
            1,
            st3,
            cols,
            fc,
            risk3,
            w3,
        )
    )
    fc2 = random_forecast(g, n, 5e-4)
    lin = CostConfig(impact=ImpactConfig("linear", 0.5))
    out.append(
        (
            "MV, LS book, quadratic cost form",
            "qp",
            OptimizerConfig("mean_variance", gamma=gamma),
            "LS",
            lin,
            5,
            st,
            cols,
            fc2,
            risk,
            None,
        )
    )
    out.append(
        (
            "MV, LS book, 1.5-power cost",
            "power",
            OptimizerConfig("mean_variance", gamma=gamma),
            "LS",
            CostConfig(),
            5,
            st,
            cols,
            fc2,
            risk,
            None,
        )
    )
    nolin = CostConfig(impact=ImpactConfig("none"))
    out.append(
        (
            "CVaR, LS book, proportional cost (LP)",
            "lp",
            OptimizerConfig("cvar", eta_cvar=0.01, cost_form="quadratic"),
            "LS",
            nolin,
            5,
            st,
            cols,
            fc2,
            risk,
            None,
        )
    )
    return out


def compare(
    n_problems: int, sizes: tuple[int, ...], seed: int = 1, reps: int = 3
) -> tuple[list, list]:
    g = stream(seed, "experiment.solvers")
    solvers = list(OPTS) + (["GUROBI"] if _gurobi_available() else [])
    rows = {}
    for n in sizes:
        for _ in range(n_problems):
            for name, klass, ocfg, book, costs, k, st, cols, fc, risk, exact in _cases(g, n):
                ref = Optimizer(ocfg, book, costs, k, solver_opts=PR.CLARABEL_TIGHT).solve(
                    st, cols, fc, risk
                )
                for solver in solvers:
                    if solver != "GUROBI" and klass not in ACCEPTS[solver]:
                        continue
                    opt = Optimizer(
                        ocfg, book, costs, k, solver=solver, solver_opts=OPTS.get(solver, {})
                    )
                    walls = []
                    sol = None
                    for _r in range(reps):
                        t0 = time.perf_counter()
                        sol = opt.solve(st, cols, fc, risk)
                        walls.append((time.perf_counter() - t0) * 1e3)
                    key = (name, n, solver)
                    r = rows.setdefault(
                        key,
                        {
                            "problem": name,
                            "n": n,
                            "solver": solver,
                            "solves": 0,
                            "failed": 0,
                            "inaccurate": 0,
                            "max_err_exact": 0.0,
                            "max_err_ref_w": 0.0,
                            "max_err_ref_obj": 0.0,
                            "wall_ms": [],
                        },
                    )
                    r["solves"] += 1
                    if not sol.success:
                        r["failed"] += 1
                        continue
                    r["inaccurate"] += sol.status == "optimal_inaccurate"
                    if exact is not None:
                        r["max_err_exact"] = max(
                            r["max_err_exact"], float(np.max(np.abs(sol.weights - exact)))
                        )
                    if ref.success:
                        r["max_err_ref_w"] = max(
                            r["max_err_ref_w"], float(np.max(np.abs(sol.weights - ref.weights)))
                        )
                        if ref.objective is not None and sol.objective is not None:
                            d = (
                                abs(sol.objective - ref.objective)
                                * PR.SCALE
                                / max(1.0, abs(ref.objective) * PR.SCALE)
                            )
                            r["max_err_ref_obj"] = max(r["max_err_ref_obj"], d)
                    r["wall_ms"].append(min(walls))
    out = []
    for key in sorted(rows):
        r = rows[key]
        w = r.pop("wall_ms")
        r["wall_ms_median"] = float(np.median(w)) if w else math.nan
        out.append(r)
    return out, solvers
