"""Cached, parametrized CVXPY problems (DPP) for the single-period optimizers and the controller.

One problem object per (kind, ladder size, book, cost form, optional constraints, horizon,
discount), built once per process and re-solved with new parameter values. Every solve passes
``warm_start=False`` (a cached problem then returns the same bits whatever it solved before).

Scaled units: the objective is multiplied by ``SCALE`` (1e4) so that its optimum is of order 1
(solver tolerances are absolute). Products of parameters are passed as one parameter:
``F`` is a factor of ``SCALE * (gamma / 2) * k * Sigma`` (so the risk term is
``sum_squares(F @ w)``), ``ca = SCALE * cost_scale * a``, ``cb = SCALE * cost_scale * b``,
``kse = SCALE * kappa_rob * se_h``, ``pe = SCALE * eta_cvar / ((1 - alpha) S)`` on active
scenario rows, ``eta_s = SCALE * eta_cvar``. The trade is a variable ``z`` with ``z == w - w0``.
The 1.5 power is ``cp.power(cp.abs(z), 1.5, approx=False)`` (exact power cone).

Slots: the instruments listed at t in instrument_id order, padded to the next size of a fixed
ladder; a slot that is not optimized gets bounds of 0 and zero data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

SCALE = 1e4
LADDER = (8, 16, 32, 64, 128, 256)
SECTOR_ROWS = 8
CVAR_ROWS = 250
UNBOUNDED = 1e3
VIOLATION_TOL = 1e-6

CLARABEL_DEFAULT = {"max_threads": 1}
# deterministic retries when Clarabel stops early (it can stall on power-cone instances with
# "insufficient progress"): the same problem with a shorter maximal step
CLARABEL_RETRIES = ({"max_step_fraction": 0.9}, {"max_step_fraction": 0.8})
CLARABEL_TIGHT = {"max_threads": 1, "tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}


def ladder_size(n: int) -> int:
    for s in LADDER:
        if n <= s:
            return s
    raise ValueError(f"{n} instruments exceed the largest ladder size")


@dataclass(frozen=True)
class BookSpec:
    name: str
    budget: str  # "eq": sum(w) == 1, "le": sum(w) <= 1, "none"
    long_only: bool
    pos_bound: float | None  # |w_i| <= pos_bound (LS); LO uses hi = max(0.10, 2 / n)
    gross: float | None  # sum |w_i| <= gross
    net: tuple[float, float] | None  # n_lo <= sum(w) <= n_hi
    gross_bound: float  # the book's gross bound (the loop's leverage rule uses 1.05 times it)


BOOKS = {
    "LO": BookSpec("LO", "eq", True, None, None, None, 1.0),
    "LS": BookSpec("LS", "none", False, 0.10, 2.0, (-0.02, 0.02), 2.0),
    "BUDGET": BookSpec("BUDGET", "eq", False, None, None, None, 1e6),
    "UNC": BookSpec("UNC", "none", False, None, None, None, 1e6),
}


@dataclass(frozen=True)
class Extras:
    """Optional constraints (None: absent)."""

    turnover: float | None = None  # sum |z| <= turnover
    sector_max: float | None = None  # |sum_{i in sector} w_i| <= sector_max
    beta_band: tuple[float, float] | None = None  # lo <= beta' w <= hi
    participation: float | None = None  # |z_i| E <= part V_i P_i

    def signature(self) -> tuple[bool, bool, bool, bool]:
        return (
            self.turnover is not None,
            self.sector_max is not None,
            self.beta_band is not None,
            self.participation is not None,
        )


@dataclass
class RawSolution:
    w: np.ndarray | None  # (N,) first-step weights in slot order
    status: str
    value: float | None  # optimal objective in original (unscaled) units
    iters: int | None
    solver: str
    duals: dict | None = None
    path: list[np.ndarray] | None = None  # mpc: all steps
    retries: int = 0


KINDS = ("mv", "cvar", "robust_box", "robust_ell", "mpc")


def cost_expression(z, ca, cb, cost_form: str, soc: bool = False):
    """The optimizer's trading cost in scaled units (the expression every problem uses).

    ``soc``: the 1.5 power through second-order cones (CVXPY's rational representation, exact
    for p = 3/2) instead of the power cone; used only as the fallback formulation when the
    power-cone solve fails (it stalls when many |z_i| sit at 0, a no-trade optimum)."""
    if cost_form == "power15":
        return ca @ cp.abs(z) + cb @ cp.power(cp.abs(z), 1.5, approx=soc)
    return ca @ cp.abs(z) + cb @ cp.square(z)


class PortfolioProblem:
    def __init__(
        self,
        kind: str,
        N: int,
        book: BookSpec,
        cost_form: str = "power15",
        extras_sig: tuple[bool, bool, bool, bool] = (False, False, False, False),
        H: int = 1,
        delta: float = 1.0,
        n_scen: int = CVAR_ROWS,
        soc: bool = False,
    ):
        if kind not in KINDS:
            raise ValueError(kind)
        if cost_form not in ("power15", "quadratic"):
            raise ValueError(cost_form)
        self.kind, self.N, self.book, self.cost_form = kind, N, book, cost_form
        self.extras_sig = extras_sig
        self.H = H if kind == "mpc" else 1
        self.delta = float(delta)
        P = {}
        P["w0"] = cp.Parameter(N, name="w0")
        P["F"] = cp.Parameter((N, N), name="F")
        P["ca"] = cp.Parameter(N, nonneg=True, name="ca")
        P["cb"] = cp.Parameter(N, nonneg=True, name="cb")
        P["lo"] = cp.Parameter(N, name="lo")
        P["hi"] = cp.Parameter(N, name="hi")
        H_ = self.H
        P["mu"] = [cp.Parameter(N, name=f"mu{j}") for j in range(H_)]
        has_turn, has_sec, has_beta, has_liq = extras_sig
        if has_turn:
            P["turn"] = cp.Parameter(nonneg=True, name="turn")
        if has_sec:
            P["G"] = cp.Parameter((SECTOR_ROWS, N), name="G")
            P["smax"] = cp.Parameter(nonneg=True, name="smax")
        if has_beta:
            P["beta"] = cp.Parameter(N, name="beta")
            P["blo"] = cp.Parameter(name="blo")
            P["bhi"] = cp.Parameter(name="bhi")
        if has_liq:
            P["zmax"] = cp.Parameter(N, nonneg=True, name="zmax")
        ws = [cp.Variable(N, name=f"w{j}") for j in range(H_)]
        zs = [cp.Variable(N, name=f"z{j}") for j in range(H_)]
        cons = []
        self.named: dict[str, cp.Constraint] = {}
        obj = 0
        prev = P["w0"]
        for j in range(H_):
            w, z = ws[j], zs[j]
            dj = self.delta**j
            c_trade = z == w - prev
            cons.append(c_trade)
            if book.budget == "eq":
                c = cp.sum(w) == 1
                cons.append(c)
                self.named.setdefault("budget", c)
            elif book.budget == "le":
                cons.append(cp.sum(w) <= 1)
            c_lo, c_hi = w >= P["lo"], w <= P["hi"]
            cons += [c_lo, c_hi]
            if j == 0:
                self.named["lo"], self.named["hi"] = c_lo, c_hi
            if book.gross is not None:
                cons.append(cp.sum(cp.abs(w)) <= book.gross)
            if book.net is not None:
                cons += [cp.sum(w) >= book.net[0], cp.sum(w) <= book.net[1]]
            if has_turn:
                cons.append(cp.sum(cp.abs(z)) <= P["turn"])
            if has_sec:
                cons.append(cp.abs(P["G"] @ w) <= P["smax"])
            if has_beta:
                cons += [P["beta"] @ w >= P["blo"], P["beta"] @ w <= P["bhi"]]
            if has_liq:
                cons.append(cp.abs(z) <= P["zmax"])
            cost = cost_expression(z, P["ca"], P["cb"], cost_form, soc)
            risk = cp.sum_squares(P["F"] @ w)
            term = P["mu"][j] @ w - risk - cost
            if kind in ("mv", "mpc"):
                obj = obj + (dj * term if j else term)
            elif kind == "cvar":
                P["R"] = cp.Parameter((n_scen, N), name="R")
                P["pe"] = cp.Parameter(n_scen, nonneg=True, name="pe")
                P["eta"] = cp.Parameter(nonneg=True, name="eta")
                P["act"] = cp.Parameter(n_scen, nonneg=True, name="act")
                zeta = cp.Variable(name="zeta")
                u = cp.Variable(n_scen, name="u")
                # an unused row is switched off (act 0: only u_s >= 0) and carries a small
                # positive weight, so that u_s = 0 there and the objective is unchanged (a
                # zero-weight free direction made the interior-point solver stall)
                cons += [u >= 0, u >= -(P["R"] @ w) - cp.multiply(P["act"], zeta)]
                obj = P["mu"][0] @ w - cost - P["eta"] * zeta - P["pe"] @ u
                self.zeta, self.u = zeta, u
            elif kind == "robust_box":
                P["kse"] = cp.Parameter(N, nonneg=True, name="kse")
                obj = term - P["kse"] @ cp.abs(w)
            else:  # robust_ell
                P["kse"] = cp.Parameter(N, nonneg=True, name="kse")
                obj = term - cp.norm(cp.multiply(P["kse"], w), 2)
            prev = w
        self.P, self.ws, self.zs = P, ws, zs
        self.problem = cp.Problem(cp.Maximize(obj), cons)
        if not self.problem.is_dpp():
            raise AssertionError("problem is not DPP")

    def set(self, values: dict) -> None:
        for k, v in values.items():
            p = self.P[k]
            if isinstance(p, list):
                for pj, vj in zip(p, v):
                    pj.value = np.asarray(vj, float)
            else:
                p.value = v if np.isscalar(v) else np.asarray(v, float)

    def solve(self, solver: str = "CLARABEL", opts: dict | None = None) -> RawSolution:
        base = (
            dict(CLARABEL_DEFAULT if solver == "CLARABEL" else {}) if opts is None else dict(opts)
        )
        attempts = [base]
        if solver == "CLARABEL":
            attempts += [{**base, **r} for r in CLARABEL_RETRIES]
        status = "solver_error"
        for n_try, o in enumerate(attempts):  # noqa: B007 - n_try is reported after the loop
            try:
                with warnings.catch_warnings():
                    # an inaccurate status is counted, not printed
                    warnings.simplefilter("ignore", UserWarning)
                    self.problem.solve(solver=solver, warm_start=False, **o)
                status = self.problem.status
            except cp.error.SolverError:
                status = "solver_error"
                continue
            if status in ("optimal", "optimal_inaccurate") and self.ws[0].value is not None:
                break
        else:
            return RawSolution(None, status, None, None, solver, retries=len(attempts) - 1)
        stats = self.problem.solver_stats
        iters = getattr(stats, "num_iters", None) if stats is not None else None
        duals = {k: c.dual_value for k, c in self.named.items()}
        return RawSolution(
            np.array(self.ws[0].value, float),
            status,
            float(self.problem.value) / SCALE,
            iters,
            solver,
            duals,
            [np.array(w.value, float) for w in self.ws],
            n_try,
        )


_CACHE: dict[tuple, PortfolioProblem] = {}


def get_problem(
    kind: str,
    N: int,
    book: str,
    cost_form: str = "power15",
    extras_sig=(False, False, False, False),
    H: int = 1,
    delta: float = 1.0,
    soc: bool = False,
) -> PortfolioProblem:
    if kind != "mpc":
        H, delta = 1, 1.0  # the horizon and the discount exist only for the controller
    key = (kind, N, book, cost_form, tuple(extras_sig), H, float(delta))
    if soc:
        key = key + ("soc",)
    p = _CACHE.get(key)
    if p is None:
        p = PortfolioProblem(kind, N, BOOKS[book], cost_form, tuple(extras_sig), H, delta, soc=soc)
        _CACHE[key] = p
    return p


def clear_cache() -> None:
    _CACHE.clear()
