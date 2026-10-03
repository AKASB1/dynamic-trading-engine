"""The optimizer family on a decision state: equal_weight, rank_ls, min_variance, mean_variance,
cvar, robust (box and ellipsoid), and the model-predictive controller mpc.

Per holding period of ``k`` bars: ``mu_h = sum_{j<k} decay^j mu``, covariance ``k * Sigma``,
the trade's cost charged once (convex form of QC 4 in units of equity). Objective (maximize):
``mu_h' w - (gamma / 2) w' (k Sigma) w - cost_scale * c(z)``; ``cvar`` replaces the variance by
``eta_cvar * CVaR_alpha`` of the holding-period loss on scenario returns; ``robust`` subtracts
``kappa_rob * sum_i se_h,i |w_i|`` (box) or ``kappa_rob * ||diag(se_h) w||_2`` (ellipsoid).

A solve succeeds if its status is ``optimal`` or ``optimal_inaccurate`` and the largest
constraint violation, measured here on the returned weights, is at most 1e-6; any other
outcome is a failure and the strategy holds (no orders). ``mpc`` with ``H = 1`` and ``robust``
with ``kappa_rob = 0`` are routed to the ``mean_variance`` problem object (bit-identical).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from dynamic_trading_engine.contracts.costs import CostConfig, convex_coefficients
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization.problems import (
    BOOKS,
    CVAR_ROWS,
    SCALE,
    SECTOR_ROWS,
    UNBOUNDED,
    VIOLATION_TOL,
    Extras,
    get_problem,
    ladder_size,
)
from dynamic_trading_engine.risk.models import RiskEstimate
from dynamic_trading_engine.state.view import DecisionState, common_window

OPTIMIZERS = (
    "cash",
    "equal_weight",
    "rank_ls",
    "min_variance",
    "mean_variance",
    "cvar",
    "robust_box",
    "robust_ell",
    "mpc",
    "wdro",  # Tier 2: Wasserstein-robust mean-CVaR (optimization/tier2.py)
    "gp_aim",  # Tier 2: multi-asset Garleanu-Pedersen rule (optimization/tier2.py)
)
INACTIVE_ROW_WEIGHT = 1e-3
NEEDS_RISK = ("min_variance", "mean_variance", "robust_box", "robust_ell", "mpc", "gp_aim")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "mean_variance"
    gamma: float = 10.0
    cost_scale: float = 1.0
    eta_cvar: float = 1.0
    alpha: float = 0.95
    kappa_rob: float = 1.0
    H: int = 5
    delta: float = 1.0
    p_rank: float = 1.0
    cost_form: str = "auto"  # auto: from the impact model (sqrt -> power15, linear -> quadratic)
    turnover: float | None = None
    sector_max: float | None = None
    beta_band: tuple[float, float] | None = None
    participation: float | None = None
    cvar_window: int = 250
    eps_w: float = 0.0  # wdro: radius of the Wasserstein ball (holding-period return units)
    lam_mult: float = 1.0  # gp_aim: multiplier of the calibrated quadratic trading-cost level
    delta_gp: float = 0.98  # gp_aim: discount per decision step

    def __post_init__(self):
        if self.name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.name!r}")

    def extras(self) -> Extras:
        return Extras(self.turnover, self.sector_max, self.beta_band, self.participation)


@dataclass
class Solution:
    weights: np.ndarray | None  # universe order (state.ids)
    status: str  # optimal, optimal_inaccurate, failed, none
    raw_status: str
    objective: float | None
    solver: str
    solver_version: str
    iterations: int | None
    wall_ms: float
    max_violation: float
    problem_key: tuple | None = None
    fallback: str = ""  # "", "step" (Clarabel retried with a shorter step), "soc"

    @property
    def success(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate", "none")


def _null_clock() -> float:
    return 0.0


def holding_factor(decay: float, k: int) -> float:
    """sum_{j=0}^{k-1} decay^j."""
    d = min(max(float(decay), 0.0), 1.0)
    return float(k) if d == 1.0 else (1.0 - d**k) / (1.0 - d)


def rank_weights(mu: np.ndarray, p: float, gross: float = 2.0, bound: float = 0.10) -> np.ndarray:
    """Cross-sectional rank of mu scaled to [-1, 1] (centred), w ~ sign(r)|r|^p scaled to the
    gross and clipped to the position bound (ties keep instrument order)."""
    n = len(mu)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(mu, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n)
    r = 2.0 * rank / (n - 1) - 1.0
    w = np.sign(r) * np.abs(r) ** p
    s = float(np.sum(np.abs(w)))
    if s == 0:
        return np.zeros(n)
    return np.clip(w * gross / s, -bound, bound)


def market_beta(R: np.ndarray) -> np.ndarray:
    """Regression of each column on the equal-weighted return over the same rows."""
    ew = R.mean(axis=1)
    ec = ew - ew.mean()
    v = float(ec @ ec)
    if v <= 0:
        return np.ones(R.shape[1])
    return (R - R.mean(axis=0)).T @ ec / v


def cvar_scenarios(R: np.ndarray, k: int) -> np.ndarray:
    """Sums of k consecutive bar returns (overlapping), column means removed."""
    h = R.shape[0]
    S = h - k + 1
    if S < 1:
        return np.zeros((0, R.shape[1]))
    c = np.vstack([np.zeros(R.shape[1]), np.cumsum(R, axis=0)])
    sc = c[k : k + S] - c[0:S]
    return sc - sc.mean(axis=0)


class Optimizer:
    def __init__(
        self,
        cfg: OptimizerConfig,
        book: str,
        costs: CostConfig,
        k: int,
        clock: Callable[[], float] | None = None,
        solver: str = "CLARABEL",
        solver_opts: dict | None = None,
    ):
        self.cfg = cfg
        self.book = BOOKS[book]
        self.book_name = book
        self.costs = costs
        self.k = int(k)
        self.clock = clock or _null_clock
        self.solver = solver
        self.solver_opts = solver_opts
        cf = cfg.cost_form
        if cf == "auto":
            cf = "quadratic" if costs.impact.model == "linear" else "power15"
        self.cost_form = cf

    # ------------------------------------------------------------------ helpers
    def _kind(self) -> str:
        c = self.cfg
        if c.name in ("mean_variance", "min_variance"):
            return "mv"
        if c.name == "mpc":
            return "mv" if c.H == 1 else "mpc"
        if c.name in ("robust_box", "robust_ell"):
            return "mv" if c.kappa_rob == 0 else c.name
        return "cvar"

    def bounds(self, n_opt: int) -> tuple[float, float]:
        b = self.book
        if b.long_only:
            return 0.0, max(0.10, 2.0 / max(n_opt, 1))
        if b.pos_bound is not None:
            return -b.pos_bound, b.pos_bound
        return -UNBOUNDED, UNBOUNDED

    # ------------------------------------------------------------------ main
    def solve(
        self,
        state: DecisionState,
        cols: np.ndarray,
        fc: Forecast | None,
        risk: RiskEstimate | None,
    ) -> Solution:
        c = self.cfg
        n_u = state.n
        cols = np.asarray(cols, dtype=int)
        n_c = len(cols)
        name = c.name
        if name in ("wdro", "gp_aim"):
            from dynamic_trading_engine.optimization import tier2

            return tier2.solve(self, state, cols, fc, risk)
        if name == "cash":  # the reference that never holds a position
            return Solution(np.zeros(n_u), "none", "none", None, "none", "", None, 0.0, 0.0)
        if name == "equal_weight":
            w = np.zeros(n_u)
            if n_c:
                w[cols] = 1.0 / n_c
            return Solution(w, "none", "none", None, "none", "", None, 0.0, 0.0)
        if name == "rank_ls":
            w = np.zeros(n_u)
            if n_c and fc is not None:
                lo, hi = self.bounds(n_c)
                gross = self.book.gross if self.book.gross is not None else 2.0
                w[cols] = rank_weights(np.asarray(fc.mu)[cols], c.p_rank, gross, hi)
            return Solution(w, "none", "none", None, "none", "", None, 0.0, 0.0)
        N = ladder_size(n_u)
        kind = self._kind()
        extras = c.extras()
        H = c.H if kind == "mpc" else 1
        prob = get_problem(kind, N, self.book_name, self.cost_form, extras.signature(), H, c.delta)
        vals, aux = self._values(state, cols, fc, risk, N, kind, extras)
        if vals is None:
            return self._fail(state, "no_data", (kind, N))
        prob.set(vals)
        t0 = self.clock()
        raw = prob.solve(self.solver, self.solver_opts)
        fallback = "step" if raw.retries and raw.w is not None else ""
        if raw.w is None and self.cost_form == "power15":
            prob2 = get_problem(
                kind, N, self.book_name, self.cost_form, extras.signature(), H, c.delta, soc=True
            )
            prob2.set(vals)
            raw = prob2.solve(self.solver, self.solver_opts)
            fallback = "soc"
        wall = (self.clock() - t0) * 1e3
        version = _solver_version(self.solver)
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
                (kind, N),
                fallback,
            )
        w_slots = raw.w
        viol = self.max_violation(w_slots, vals, aux, extras)
        if not np.all(np.isfinite(w_slots)):
            viol = math.inf  # NaN weights never pass the failure rule
        w = np.array(w_slots[:n_u])
        status = raw.status if viol <= VIOLATION_TOL else "failed"
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
            (kind, N),
            fallback,
        )

    def _fail(self, state, why, key):
        return Solution(None, "failed", why, None, self.solver, "", None, 0.0, math.inf, key)

    def _values(self, state, cols, fc, risk, N, kind, extras):
        c = self.cfg
        n_u, n_c = state.n, len(cols)
        name = c.name
        E = state.portfolio.equity
        k = self.k
        vals: dict = {}
        w0 = np.zeros(N)
        w0[:n_u] = state.portfolio.weights
        vals["w0"] = w0
        lo_b, hi_b = self.bounds(n_c)
        lo, hi = np.zeros(N), np.zeros(N)
        lo[cols], hi[cols] = lo_b, hi_b
        vals["lo"], vals["hi"] = lo, hi
        # costs (every universe slot with data; the optimizer may have to sell an excluded one)
        P = np.asarray(state.last_close, float)
        okp = np.isfinite(P) & (P > 0)
        a, b, _form = convex_coefficients(
            np.where(okp, P, 1.0), np.asarray(state.sigma_bar), np.asarray(state.adv), E, self.costs
        )
        a, b = np.where(okp, a, 0.0), np.where(okp, b, 0.0)
        cs = 0.0 if name == "min_variance" else c.cost_scale
        ca, cb = np.zeros(N), np.zeros(N)
        ca[:n_u] = SCALE * cs * a
        cb[:n_u] = SCALE * cs * (b if self.cost_form == "power15" else 0.5 * b)
        vals["ca"], vals["cb"] = ca, cb
        # forecast over the holding period
        mu_h = np.zeros(n_u)
        se_h = np.zeros(n_u)
        decay = 0.0
        if fc is not None and name != "min_variance":
            decay = float(fc.decay)
            hf = holding_factor(decay, k)
            mu_h[cols] = np.asarray(fc.mu)[cols] * hf
            se_h[cols] = np.asarray(fc.se)[cols] * hf
        # risk factor
        F = np.zeros((N, N))
        if kind != "cvar":
            if risk is None:
                return None, None
            g = 1.0 if name == "min_variance" else c.gamma / 2.0 * k
            F[:n_c, cols] = math.sqrt(SCALE * g) * risk.factor
        vals["F"] = F
        if kind == "mpc":
            mus = []
            for j in range(c.H):
                mj = np.zeros(N)
                mj[:n_u] = SCALE * decay ** (j * k) * mu_h
                mus.append(mj)
            vals["mu"] = mus
        else:
            m0 = np.zeros(N)
            m0[:n_u] = SCALE * mu_h
            vals["mu"] = [m0]
        if kind in ("robust_box", "robust_ell"):
            kse = np.zeros(N)
            kse[:n_u] = SCALE * c.kappa_rob * se_h
            vals["kse"] = kse
        aux: dict = {"n_u": n_u, "cols": cols}
        R_common = None
        if kind == "cvar" or extras.beta_band is not None:
            h = common_window(np.asarray(state.returns), cols, c.cvar_window)
            if h < 2:
                return None, None
            R_common = np.asarray(state.returns)[-h:, cols]
        if kind == "cvar":
            sc = cvar_scenarios(R_common, k)
            S_act = min(sc.shape[0], CVAR_ROWS)
            if S_act == 0:
                return None, None  # fewer common bars than the holding period: no scenario
            sc = sc[-S_act:] if S_act else sc
            Rp = np.zeros((CVAR_ROWS, N))
            if S_act:
                Rp[:S_act, cols] = sc
            pe = np.full(CVAR_ROWS, INACTIVE_ROW_WEIGHT)
            act = np.zeros(CVAR_ROWS)
            if S_act:
                pe[:S_act] = SCALE * c.eta_cvar / ((1.0 - c.alpha) * S_act)
                act[:S_act] = 1.0
            vals["R"], vals["pe"], vals["eta"], vals["act"] = Rp, pe, SCALE * c.eta_cvar, act
        if extras.turnover is not None:
            vals["turn"] = float(extras.turnover)
        if extras.sector_max is not None:
            secs = sorted({state.sector[i] for i in cols})
            G = np.zeros((SECTOR_ROWS, N))
            for r, s in enumerate(secs[:SECTOR_ROWS]):
                for i in cols:
                    if state.sector[i] == s:
                        G[r, i] = 1.0
            vals["G"], vals["smax"] = G, float(extras.sector_max)
            aux["G"] = G
        if extras.beta_band is not None:
            beta = np.zeros(N)
            beta[cols] = market_beta(R_common)
            vals["beta"], vals["blo"], vals["bhi"] = (
                beta,
                float(extras.beta_band[0]),
                float(extras.beta_band[1]),
            )
        if extras.participation is not None:
            zmax = np.zeros(N)
            V = np.asarray(state.adv)
            okv = np.isfinite(V) & okp
            zm = np.where(
                okv, extras.participation * np.where(okv, V, 0) * np.where(okp, P, 0) / E, 1e6
            )
            zmax[:n_u] = zm
            vals["zmax"] = zmax
        return vals, aux

    def max_violation(self, w: np.ndarray, vals: dict, aux: dict, extras: Extras) -> float:
        b = self.book
        v = 0.0
        if b.budget == "eq":
            v = max(v, abs(float(np.sum(w)) - 1.0))
        elif b.budget == "le":
            v = max(v, float(np.sum(w)) - 1.0)
        v = max(v, float(np.max(vals["lo"] - w)), float(np.max(w - vals["hi"])))
        if b.gross is not None:
            v = max(v, float(np.sum(np.abs(w))) - b.gross)
        if b.net is not None:
            s = float(np.sum(w))
            v = max(v, b.net[0] - s, s - b.net[1])
        z = w - vals["w0"]
        if extras.turnover is not None:
            v = max(v, float(np.sum(np.abs(z))) - extras.turnover)
        if extras.sector_max is not None:
            v = max(v, float(np.max(np.abs(aux["G"] @ w))) - extras.sector_max)
        if extras.beta_band is not None:
            bw = float(vals["beta"] @ w)
            v = max(v, extras.beta_band[0] - bw, bw - extras.beta_band[1])
        if extras.participation is not None:
            v = max(v, float(np.max(np.abs(z) - vals["zmax"])))
        return max(v, 0.0)


_VERSIONS: dict[str, str] = {}


def _solver_version(name: str) -> str:
    if name not in _VERSIONS:
        try:
            if name == "CLARABEL":
                import clarabel

                _VERSIONS[name] = clarabel.__version__
            elif name == "OSQP":
                import osqp

                _VERSIONS[name] = osqp.__version__
            elif name == "SCS":
                import scs

                _VERSIONS[name] = scs.__version__
            else:
                _VERSIONS[name] = ""
        except Exception:  # pragma: no cover - version lookup only
            _VERSIONS[name] = ""
    return _VERSIONS[name]


def solver_versions() -> dict:
    return {"cvxpy": cp.__version__, **{s: _solver_version(s) for s in ("CLARABEL", "OSQP", "SCS")}}
