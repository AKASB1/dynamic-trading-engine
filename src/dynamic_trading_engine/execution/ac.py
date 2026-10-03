"""Almgren-Chriss reference (R8): discrete time, zero fixed cost.

With ``N`` intervals, ``tau = T / N``, ``eta_tilde = eta_ac - gamma_ac tau / 2``,
``kappa_tilde^2 = lambda_ac sigma_ac^2 / eta_tilde`` and
``kappa_ac = (2 / tau) asinh(kappa_tilde tau / 2)`` (exact for tiny lambda; at kappa_ac = 0 the
schedule is TWAP), the optimal holdings after interval j are
``x_j = X sinh(kappa_ac (T - t_j)) / sinh(kappa_ac T)``, the trades ``n_j = x_{j-1} - x_j``,
``E_ac = gamma_ac X^2 / 2 + (eta_tilde / tau) sum n_j^2`` and
``V_ac = sigma_ac^2 tau sum_{j=1}^{N-1} x_j^2``; the schedule minimizes ``E_ac + lambda_ac V_ac``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class ACParams:
    X: float
    T: float
    N: int
    sigma: float
    gamma: float
    eta: float
    lam: float


@dataclass(frozen=True)
class ACSchedule:
    kappa: float
    x: np.ndarray  # holdings x_0 .. x_N
    n: np.ndarray  # trades n_1 .. n_N
    E: float
    V: float

    @property
    def objective_at(self):
        return lambda lam: self.E + lam * self.V


def expected_cost_variance(p: ACParams, x: np.ndarray) -> tuple[float, float]:
    tau = p.T / p.N
    n = x[:-1] - x[1:]
    eta_t = p.eta - p.gamma * tau / 2.0
    E = p.gamma * p.X**2 / 2.0 + eta_t / tau * float(np.sum(n * n))
    V = p.sigma**2 * tau * float(np.sum(x[1:-1] ** 2))
    return E, V


def ac_closed_form(p: ACParams) -> ACSchedule:
    tau = p.T / p.N
    eta_t = p.eta - p.gamma * tau / 2.0
    if eta_t <= 0:
        raise ValueError("eta_tilde must be positive")
    kt = math.sqrt(p.lam * p.sigma**2 / eta_t)
    kappa = (2.0 / tau) * math.asinh(kt * tau / 2.0)
    t = np.arange(p.N + 1) * tau
    if kappa == 0.0:
        x = p.X * (1.0 - t / p.T)
    else:
        x = p.X * np.sinh(kappa * (p.T - t)) / math.sinh(kappa * p.T)
    x[-1] = 0.0
    E, V = expected_cost_variance(p, x)
    return ACSchedule(kappa, x, x[:-1] - x[1:], E, V)


def twap_schedule(p: ACParams) -> np.ndarray:
    t = np.arange(p.N + 1) * (p.T / p.N)
    x = p.X * (1.0 - t / p.T)
    x[-1] = 0.0
    return x


def ac_linear_solve(p: ACParams) -> np.ndarray:
    """First-order conditions: (2 + lambda sigma^2 tau^2 / eta_tilde) x_j - x_{j-1} - x_{j+1} = 0,
    x_0 = X, x_N = 0 (a tridiagonal system)."""
    tau = p.T / p.N
    eta_t = p.eta - p.gamma * tau / 2.0
    m = p.N - 1
    x = np.zeros(p.N + 1)
    x[0] = p.X
    if m <= 0:
        return x
    d = 2.0 + p.lam * p.sigma**2 * tau**2 / eta_t
    A = np.diag(np.full(m, d)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)
    b = np.zeros(m)
    b[0] = p.X
    x[1:-1] = np.linalg.solve(A, b)
    return x


def ac_qp(p: ACParams, opts: dict | None = None) -> np.ndarray:
    """The quadratic program min E + lambda V over x_1..x_{N-1} in scaled units (holdings in
    units of X, objective divided by its TWAP value)."""
    tau = p.T / p.N
    eta_t = p.eta - p.gamma * tau / 2.0
    y = cp.Variable(p.N + 1)
    n = y[:-1] - y[1:]
    E = eta_t / tau * cp.sum_squares(n) * p.X**2
    V = p.sigma**2 * tau * cp.sum_squares(y[1:-1]) * p.X**2
    xt = twap_schedule(p)
    e0, v0 = expected_cost_variance(p, xt)
    scale = 1.0 / max(e0 - p.gamma * p.X**2 / 2.0 + p.lam * v0, 1e-300)
    prob = cp.Problem(cp.Minimize(scale * (E + p.lam * V)), [y[0] == 1.0, y[-1] == 0.0])
    o = {"max_threads": 1, "tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}
    o.update(opts or {})
    prob.solve(solver="CLARABEL", warm_start=False, **o)
    return np.asarray(y.value, float) * p.X
