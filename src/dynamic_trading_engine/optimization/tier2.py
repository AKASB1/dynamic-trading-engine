"""Tier-2 optimizers on a decision state, reached from ``Optimizer.solve`` by name (no Tier-1 path
changes):

- ``wdro``: the Wasserstein-robust mean-CVaR of ``optimization/wdro.py`` on the scenarios of the
  ``cvar`` optimizer (holding-period sums of the known returns of the common window, demeaned)
  shifted by the holding-period forecast; ``eps_w`` the radius, ``eta_cvar`` and ``alpha`` as in
  ``cvar``, the contract's convex cost scaled by ``cost_scale``.
- ``gp_aim``: the multi-asset Garleanu-Pedersen rule of ``policies/gp_aim.py`` (trade the fraction
  ``a`` of the way to ``aim * (k Sigma)^-1 mu_h``) with the quadratic cost level
  ``lam_gp = lam_mult * calibrated_lam`` (matched to the contract's cost at a 2 percent trade),
  projected onto the book (the closest feasible weights in the Euclidean norm, solved on the cached
  mean-variance object with an identity factor and zero cost coefficients).
"""

from __future__ import annotations

import math

import numpy as np

from dynamic_trading_engine.contracts.costs import convex_coefficients
from dynamic_trading_engine.optimization.optimizers import (
    Solution,
    _solver_version,
    cvar_scenarios,
    holding_factor,
)
from dynamic_trading_engine.optimization.problems import (
    SCALE,
    VIOLATION_TOL,
    get_problem,
    ladder_size,
)
from dynamic_trading_engine.optimization.wdro import get_wdro_problem, wdro_values
from dynamic_trading_engine.policies.gp_aim import calibrated_lam, gp_target
from dynamic_trading_engine.state.view import common_window


def solve(opt, state, cols, fc, risk) -> Solution:
    cols = np.asarray(cols, dtype=int)
    if opt.cfg.name == "wdro":
        return _wdro(opt, state, cols, fc)
    return _gp_aim(opt, state, cols, fc, risk)


def _fail(opt, key, why="no_data"):
    return Solution(None, "failed", why, None, "CLARABEL", "", None, 0.0, math.inf, key)


def _finish(opt, raw, vals, n_u, wall, key, fallback="", aux=None) -> Solution:
    version = _solver_version("CLARABEL")
    if raw.w is None:
        return Solution(
            None,
            "failed",
            raw.status,
            None,
            raw.solver,
            version,
            None,
            wall,
            math.inf,
            key,
            fallback,
        )
    viol = opt.max_violation(raw.w, vals, aux or {}, opt.cfg.extras())
    if not np.all(np.isfinite(raw.w)):
        viol = math.inf
    status = raw.status if viol <= VIOLATION_TOL else "failed"
    w = np.array(raw.w[:n_u])
    return Solution(
        w if status != "failed" else None,
        status,
        raw.status,
        raw.value,
        raw.solver,
        version,
        raw.iters,
        wall,
        viol,
        key,
        fallback,
    )


def _wdro(opt, state, cols, fc) -> Solution:
    c = opt.cfg
    n_u = state.n
    N = ladder_size(n_u)
    key = ("wdro", N)
    base, _aux = opt._values(state, cols, fc, None, N, "cvar", c.extras())
    if base is None or len(cols) == 0:
        return _fail(opt, key)
    h = common_window(np.asarray(state.returns), cols, c.cvar_window)
    sc = (
        cvar_scenarios(np.asarray(state.returns)[-h:, cols], opt.k)
        if h >= 2
        else np.zeros((0, len(cols)))
    )
    if sc.shape[0] == 0:
        return _fail(opt, key)
    mu_h = base["mu"][0][cols] / SCALE
    wv = wdro_values(mu_h, sc, c.eps_w, c.eta_cvar, c.alpha, N, cols)
    vals = {k: base[k] for k in ("w0", "ca", "cb", "lo", "hi")}
    vals.update({k: v for k, v in wv.items() if k != "S_act"})
    prob = get_wdro_problem(N, opt.book_name, opt.cost_form)
    prob.set(vals)
    t0 = opt.clock()
    raw = prob.solve(opt.solver_opts)
    fallback = "step" if raw.retries and raw.w is not None else ""
    if raw.w is None and opt.cost_form == "power15":
        p2 = get_wdro_problem(N, opt.book_name, opt.cost_form, soc=True)
        p2.set(vals)
        raw = p2.solve(opt.solver_opts)
        fallback = "soc"
    wall = (opt.clock() - t0) * 1e3
    return _finish(opt, raw, base, n_u, wall, key, fallback)


def _gp_aim(opt, state, cols, fc, risk) -> Solution:
    c = opt.cfg
    n_u = state.n
    N = ladder_size(n_u)
    key = ("gp_aim", N)
    if risk is None or fc is None or len(cols) == 0:
        return _fail(opt, key)
    k = opt.k
    decay = float(fc.decay)
    mu_h = np.asarray(fc.mu)[cols] * holding_factor(decay, k)
    Sk = k * risk.sigma
    P = np.asarray(state.last_close)[cols]
    a, b, _f = convex_coefficients(
        P,
        np.asarray(state.sigma_bar)[cols],
        np.asarray(state.adv)[cols],
        state.portfolio.equity,
        opt.costs,
    )
    lam = c.lam_mult * calibrated_lam(
        np.where(np.isfinite(a), a, 0.0), np.where(np.isfinite(b), b, 0.0), Sk
    )
    x_prev = np.asarray(state.portfolio.weights)[cols]
    target, _a, _aim = gp_target(mu_h, Sk, x_prev, c.gamma, lam, decay**k, c.delta_gp)
    base, aux = opt._values(state, cols, fc, risk, N, "mv", c.extras())
    if base is None:
        return _fail(opt, key)
    F = np.zeros((N, N))
    F[cols, cols] = math.sqrt(SCALE)
    m0 = np.zeros(N)
    m0[cols] = 2.0 * SCALE * target
    vals = dict(base)
    vals.update({"F": F, "mu": [m0], "ca": np.zeros(N), "cb": np.zeros(N)})
    prob = get_problem("mv", N, opt.book_name, opt.cost_form, c.extras().signature())
    prob.set(vals)
    t0 = opt.clock()
    raw = prob.solve(opt.solver, opt.solver_opts)
    wall = (opt.clock() - t0) * 1e3
    return _finish(opt, raw, base, n_u, wall, key, "", aux)
