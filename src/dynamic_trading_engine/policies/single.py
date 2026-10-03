"""The single-asset environment of R6 and R7 and the model-predictive controller on it.

The controller plans ``H`` steps with the expected path ``mu_j = (1 - phi)^(j-1) mu`` (certainty
equivalence) and takes the first action. Two implementations of the unconstrained quadratic
case: a linear-algebra one (a tridiagonal system; the first action is linear in ``(xp, mu)``,
``x_1 = alpha_H xp + beta_H mu``) used by the Monte Carlo of S6(a), and the CVXPY controller
(the cached ``mpc`` problem of the portfolio optimizers, one active slot, book ``UNC``).
"""

from __future__ import annotations

import numpy as np

from dynamic_trading_engine.optimization.problems import (
    CLARABEL_TIGHT,
    SCALE,
    UNBOUNDED,
    get_problem,
)

N_SLOTS = 8


def mpc_coefficients(
    g: float, cost: float, phi: float, delta: float, H: int
) -> tuple[float, float]:
    """(alpha_H, beta_H) of the unconstrained quadratic controller: maximize
    sum_{j=1}^H delta^(j-1) [mu_j x_j - (g/2) x_j^2 - (cost/2) (x_j - x_{j-1})^2]."""
    M = np.zeros((H, H))
    for j in range(H):
        dj = delta**j
        M[j, j] += dj * (g + cost)
        if j > 0:
            M[j, j - 1] -= dj * cost
        if j + 1 < H:
            dn = delta ** (j + 1)
            M[j, j] += dn * cost
            M[j, j + 1] -= dn * cost
    path = np.array([delta**j * (1.0 - phi) ** j for j in range(H)])
    b_xp = np.zeros(H)
    b_xp[0] = cost
    x_xp = np.linalg.solve(M, b_xp)
    x_mu = np.linalg.solve(M, path)
    return float(x_xp[0]), float(x_mu[0])


def mpc_first_action_la(xp, mu, g, cost, phi, delta, H):
    a, b = mpc_coefficients(g, cost, phi, delta, H)
    return a * np.asarray(xp) + b * np.asarray(mu)


def mpc_first_action_cvxpy(
    xp: float,
    mu: float,
    g: float,
    cost: float,
    phi: float,
    delta: float,
    H: int,
    cost_kind: str = "quadratic",
    opts: dict | None = None,
) -> float:
    """The CVXPY controller on one asset: quadratic cost ``(cost/2) dx^2`` or proportional
    ``cost |dx|``; risk ``(g/2) x^2``; no other constraint."""
    form = "quadratic"
    kind = "mpc" if H > 1 else "mv"
    prob = get_problem(kind, N_SLOTS, "UNC", form, (False,) * 4, H, delta)
    F = np.zeros((N_SLOTS, N_SLOTS))
    F[0, 0] = np.sqrt(SCALE * g / 2.0)
    ca, cb = np.zeros(N_SLOTS), np.zeros(N_SLOTS)
    if cost_kind == "quadratic":
        cb[0] = SCALE * cost / 2.0
    else:
        ca[0] = SCALE * cost
    lo, hi = np.zeros(N_SLOTS), np.zeros(N_SLOTS)
    lo[0], hi[0] = -UNBOUNDED, UNBOUNDED
    w0 = np.zeros(N_SLOTS)
    w0[0] = xp
    mus = []
    for j in range(prob.H):
        m = np.zeros(N_SLOTS)
        m[0] = SCALE * (1.0 - phi) ** j * mu
        mus.append(m)
    prob.set({"w0": w0, "F": F, "ca": ca, "cb": cb, "lo": lo, "hi": hi, "mu": mus})
    raw = prob.solve("CLARABEL", opts if opts is not None else CLARABEL_TIGHT)
    if raw.w is None:
        raise RuntimeError(f"controller solve failed: {raw.status}")
    return float(raw.w[0])


def simulate_mu(n_paths: int, T: int, phi: float, sigma_eps: float, rng) -> np.ndarray:
    """(T, n_paths) expected-return paths started from the stationary distribution."""
    r = 1.0 - phi
    sd = sigma_eps / np.sqrt(1.0 - r * r)
    eps = rng.standard_normal((T, n_paths))
    mu = np.empty((T, n_paths))
    mu[0] = sd * eps[0]
    for t in range(1, T):
        mu[t] = r * mu[t - 1] + sigma_eps * eps[t]
    return mu


def discounted_utility(x: np.ndarray, mu: np.ndarray, g, delta, cost, cost_kind="quadratic"):
    """sum_t delta^t [x_t mu_t - (g/2) x_t^2 - cost(x_t - x_{t-1})] per path, x_{-1} = 0."""
    T = x.shape[0]
    dx = np.diff(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    c = 0.5 * cost * dx**2 if cost_kind == "quadratic" else cost * np.abs(dx)
    r = x * mu - 0.5 * g * x**2 - c
    disc = delta ** np.arange(T)
    return disc @ r


def run_linear_policy(mu: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    x = np.empty_like(mu)
    prev = np.zeros(mu.shape[1])
    for t in range(mu.shape[0]):
        prev = alpha * prev + beta * mu[t]
        x[t] = prev
    return x
