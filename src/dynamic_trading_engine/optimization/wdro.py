"""Distributionally robust mean-CVaR over a 1-Wasserstein ball (Tier 2).

Worst case of ``E[-w'xi] + eta_cvar * CVaR_alpha(-w'xi)`` over every distribution within
1-Wasserstein distance ``eps`` (ground norm l2 on R^n) of the empirical distribution of the
holding-period scenarios ``xi_i = mu_h + R_i`` (the scenarios of the ``cvar`` optimizer with the
forecast as their mean). The loss is the maximum of two affine functions of ``xi``
(Mohajerin Esfahani and Kuhn 2018, the mean-CVaR case of their piecewise-affine result):

    l_1(xi) = -w'xi + eta tau
    l_2(xi) = -(1 + eta / (1 - alpha)) w'xi + eta tau (1 - 1 / (1 - alpha))

and the worst-case expectation is the convex program

    min_{tau, lam >= 0, s}  lam * eps + (1/S) sum_i s_i
    s.t.  s_i >= l_k(xi_i) (k = 1, 2),   lam >= (1 + eta / (1 - alpha)) ||w||_2.

The optimizer maximizes ``-(that value) - cost_scale * c(z)`` under the book's constraints. With
``eps = 0`` it is the empirical mean-CVaR problem of ``cvar``; with ``eta = 0`` (linear loss) the
worst-case expectation is the empirical one plus ``eps ||w||_2``.

DPP: the scenario matrix enters twice as parameters (``Xi`` and ``c2 * Xi``); unused scenario rows
are switched off (``b1 = b2 = 0`` and a small positive weight, so ``s_i = 0`` there).
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from dynamic_trading_engine.optimization.problems import (
    BOOKS,
    CLARABEL_DEFAULT,
    CLARABEL_RETRIES,
    CVAR_ROWS,
    SCALE,
    RawSolution,
    cost_expression,
)

INACTIVE_WEIGHT = 1e-3


class WDROProblem:
    def __init__(
        self,
        N: int,
        book: str,
        cost_form: str = "power15",
        soc: bool = False,
        n_scen: int = CVAR_ROWS,
    ):
        bk = BOOKS[book]
        self.N, self.book = N, bk
        P = {
            "w0": cp.Parameter(N, name="w0"),
            "ca": cp.Parameter(N, nonneg=True, name="ca"),
            "cb": cp.Parameter(N, nonneg=True, name="cb"),
            "lo": cp.Parameter(N, name="lo"),
            "hi": cp.Parameter(N, name="hi"),
            "Xi": cp.Parameter((n_scen, N), name="Xi"),  # SCALE * (mu_h + R_i) on active rows
            "Xi2": cp.Parameter((n_scen, N), name="Xi2"),  # c2 * Xi
            "b1": cp.Parameter(n_scen, name="b1"),  # SCALE * eta on active rows
            "b2": cp.Parameter(n_scen, name="b2"),  # SCALE * eta (1 - 1/(1-alpha)) on active rows
            "p": cp.Parameter(n_scen, nonneg=True, name="p"),  # 1/S on active rows
            "c2": cp.Parameter(nonneg=True, name="c2"),  # SCALE * (1 + eta/(1-alpha))
            "eps": cp.Parameter(nonneg=True, name="eps"),
        }
        w = cp.Variable(N, name="w")
        z = cp.Variable(N, name="z")
        tau = cp.Variable(name="tau")
        lam = cp.Variable(nonneg=True, name="lam")
        s = cp.Variable(n_scen, name="s")
        cons = [z == w - P["w0"], w >= P["lo"], w <= P["hi"]]
        if bk.budget == "eq":
            cons.append(cp.sum(w) == 1)
        elif bk.budget == "le":
            cons.append(cp.sum(w) <= 1)
        if bk.gross is not None:
            cons.append(cp.sum(cp.abs(w)) <= bk.gross)
        if bk.net is not None:
            cons += [cp.sum(w) >= bk.net[0], cp.sum(w) <= bk.net[1]]
        cons += [
            s >= -(P["Xi"] @ w) + P["b1"] * tau,
            s >= -(P["Xi2"] @ w) + P["b2"] * tau,
            lam >= P["c2"] * cp.norm(w, 2),
        ]
        worst = P["eps"] * lam + P["p"] @ s
        cost = cost_expression(z, P["ca"], P["cb"], cost_form, soc)
        self.problem = cp.Problem(cp.Maximize(-worst - cost), cons)
        if not self.problem.is_dpp():
            raise AssertionError("WDRO problem is not DPP")
        self.P, self.w, self.lam, self.s, self.tau = P, w, lam, s, tau

    def set(self, values: dict) -> None:
        for k, v in values.items():
            self.P[k].value = v if np.isscalar(v) else np.asarray(v, float)

    def solve(self, opts: dict | None = None) -> RawSolution:
        base = dict(CLARABEL_DEFAULT) if opts is None else dict(opts)
        status = "solver_error"
        for n_try, o in enumerate([base] + [{**base, **r} for r in CLARABEL_RETRIES]):  # noqa: B007
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.problem.solve(solver="CLARABEL", warm_start=False, **o)
                status = self.problem.status
            except cp.error.SolverError:
                continue
            if status in ("optimal", "optimal_inaccurate") and self.w.value is not None:
                break
        else:
            return RawSolution(None, status, None, None, "CLARABEL")
        stats = self.problem.solver_stats
        return RawSolution(
            np.array(self.w.value, float),
            status,
            float(self.problem.value) / SCALE,
            getattr(stats, "num_iters", None),
            "CLARABEL",
            None,
            None,
            n_try,
        )


_CACHE: dict[tuple, WDROProblem] = {}


def get_wdro_problem(
    N: int, book: str, cost_form: str = "power15", soc: bool = False
) -> WDROProblem:
    key = (N, book, cost_form, soc)
    if key not in _CACHE:
        _CACHE[key] = WDROProblem(N, book, cost_form, soc)
    return _CACHE[key]


def wdro_values(
    mu_h: np.ndarray,
    scen: np.ndarray,
    eps: float,
    eta: float,
    alpha: float,
    N: int,
    cols: np.ndarray,
) -> dict:
    """Scenario parameters: ``scen`` (S x n_cols) demeaned holding-period scenario returns,
    ``mu_h`` the holding-period forecast of the optimized columns."""
    S_act = min(scen.shape[0], CVAR_ROWS)
    scen = scen[-S_act:] if S_act else scen
    Xi = np.zeros((CVAR_ROWS, N))
    if S_act:
        Xi[:S_act, cols] = SCALE * (scen + mu_h[None, :])
    c2 = 1.0 + eta / (1.0 - alpha)
    b1 = np.zeros(CVAR_ROWS)
    b2 = np.zeros(CVAR_ROWS)
    p = np.full(CVAR_ROWS, INACTIVE_WEIGHT)
    if S_act:
        b1[:S_act] = SCALE * eta
        b2[:S_act] = SCALE * eta * (1.0 - 1.0 / (1.0 - alpha))
        p[:S_act] = 1.0 / S_act
    return {
        "Xi": Xi,
        "Xi2": c2 * Xi,
        "b1": b1,
        "b2": b2,
        "p": p,
        "c2": SCALE * c2,
        "eps": float(eps),
        "S_act": S_act,
    }
