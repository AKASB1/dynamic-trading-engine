"""Grid dynamic programs for one asset (NumPy only, independent of CVXPY).

State: the position held before the trade ``xp`` (grid) and the expected return ``mu`` (grid).
Action: the new position ``x`` on the same position grid. Reward of a period
``x mu - (g/2) x^2 - cost(x - xp)`` with ``cost`` quadratic (``(Lam/2) dx^2``, R6) or
proportional (``c_prop |dx|``, R7); ``mu' = (1 - phi) mu + sigma_eps eps`` with the expectation
over ``eps`` by Gauss-Hermite quadrature and linear interpolation (linear extrapolation beyond
the grid) along ``mu``. Modified policy iteration: a greedy step, then repeated evaluations of
the greedy policy, until the value changes by less than ``tol``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


@dataclass
class DPResult:
    x_grid: np.ndarray
    mu_grid: np.ndarray
    V: np.ndarray  # (n_x, n_mu): value at (xp, mu)
    policy_idx: np.ndarray  # (n_x, n_mu): index of the optimal x on x_grid
    iterations: int

    def policy(self) -> np.ndarray:
        return self.x_grid[self.policy_idx]

    def value_at(self, xp: float, mu: float) -> float:
        i = int(np.argmin(np.abs(self.x_grid - xp)))
        return float(_interp_rows(self.V[i : i + 1], self.mu_grid, np.array([mu]))[0, 0])


def _interp_rows(V: np.ndarray, grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Linear interpolation of each row of V (values on ``grid``) at ``pts``; linear
    extrapolation with the edge slopes."""
    h = grid[1] - grid[0]
    pos = (pts - grid[0]) / h
    j = np.clip(np.floor(pos).astype(int), 0, len(grid) - 2)
    t = pos - j
    return V[:, j] * (1.0 - t) + V[:, j + 1] * t


def solve_dp(
    g: float,
    phi: float,
    sigma_eps: float,
    delta: float,
    cost: str,
    cost_param: float,
    x_max: float,
    n_x: int,
    mu_max: float,
    n_mu: int,
    n_quad: int = 9,
    tol: float = 1e-11,
    max_outer: int = 500,
    eval_sweeps: int = 30,
) -> DPResult:
    x = np.linspace(-x_max, x_max, n_x)
    mu = np.linspace(-mu_max, mu_max, n_mu)
    z, wq = hermegauss(n_quad)
    wq = wq / wq.sum()
    nxt = (1.0 - phi) * mu[:, None] + sigma_eps * z[None, :]  # (n_mu, n_quad)
    dx = x[None, :] - x[:, None]  # (xp, x)
    trade_cost = 0.5 * cost_param * dx**2 if cost == "quadratic" else cost_param * np.abs(dx)
    gain = x[:, None] * mu[None, :] - 0.5 * g * x[:, None] ** 2  # (x, mu)
    V = np.zeros((n_x, n_mu))
    pol = np.zeros((n_x, n_mu), dtype=int)

    def continuation(V):
        # C(x, mu) = E V(x, mu') : (n_x, n_mu)
        vals = _interp_rows(V, mu, nxt.ravel()).reshape(n_x, n_mu, n_quad)
        return vals @ wq

    it = 0
    for it in range(1, max_outer + 1):  # noqa: B007 - the count is returned
        C = continuation(V)
        G = gain + delta * C  # (x, mu)
        Q = G[None, :, :] - trade_cost[:, :, None]  # (xp, x, mu)
        pol = np.argmax(Q, axis=1)
        V_new = np.take_along_axis(Q, pol[:, None, :], axis=1)[:, 0, :]
        # evaluate the greedy policy for a few sweeps
        r = np.take_along_axis(
            (gain[None, :, :] - trade_cost[:, :, None]), pol[:, None, :], axis=1
        )[:, 0, :]
        cols = np.arange(n_mu)[None, :]
        for _ in range(eval_sweeps):
            Cn = continuation(V_new)
            V_new = r + delta * Cn[pol, cols]
        change = float(np.max(np.abs(V_new - V)))
        V = V_new
        if change < tol * max(1.0, float(np.max(np.abs(V)))):
            break
    return DPResult(x, mu, V, pol, it)


def no_trade_band(res: DPResult, j_mu: int) -> tuple[float, float]:
    """The no-trade interval of the policy at ``mu_grid[j_mu]``: positions that are kept."""
    keep = res.policy_idx[:, j_mu] == np.arange(len(res.x_grid))
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        x = res.x_grid[res.policy_idx[0, j_mu]]
        return float(x), float(x)
    return float(res.x_grid[idx[0]]), float(res.x_grid[idx[-1]])
