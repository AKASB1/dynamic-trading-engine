"""The Garleanu-Pedersen policy in the multi-asset market (Tier 2, ``gp_aim``).

Per decision step of ``k`` bars, maximize
``E sum_t delta^t [x_t' mu_t - (gamma/2) x_t' (k Sigma) x_t - (lam_gp/2) dx_t' (k Sigma) dx_t]``
with the expected holding-period return ``mu_t`` decaying by ``d = decay^k`` per step. With the
trading-cost matrix proportional to the risk matrix, the coordinates ``y = (k Sigma)^(1/2) x``
decouple the problem into independent scalar problems of the reference R6 with ``g = gamma``,
``Lam = lam_gp``, ``phi_gp = 1 - d``, ``rho_gp = 1 - delta`` and the expected return
``(k Sigma)^(-1/2) mu``. The scalar policy ``y = (1 - a) y_prev + a * aim * mu~`` maps back to

    x = (1 - a) x_prev + a * aim * (k Sigma)^(-1) mu_h

the GP rule: trade the fraction ``a`` of the way to the aim portfolio. The target is projected
onto the book's constraints (closest point in the Euclidean norm).
"""

from __future__ import annotations

import numpy as np

from dynamic_trading_engine.policies.gp import GPParams, gp_closed_form


def gp_target(
    mu_h: np.ndarray,
    sigma_k: np.ndarray,
    x_prev: np.ndarray,
    gamma: float,
    lam_gp: float,
    decay_step: float,
    delta: float = 0.98,
) -> tuple[np.ndarray, float, float]:
    """(target weights, trade rate a, aim coefficient) for the holding-period forecast ``mu_h``,
    the holding-period covariance ``sigma_k`` (= k Sigma), and the current weights."""
    p = GPParams(g=gamma, Lam=lam_gp, rho_gp=1.0 - delta, phi_gp=1.0 - decay_step, sigma_eps=0.0)
    s = gp_closed_form(p)
    aim_port = s.aim * np.linalg.solve(sigma_k, mu_h)
    return (1.0 - s.a) * x_prev + s.a * aim_port, s.a, s.aim


def calibrated_lam(a: np.ndarray, b: np.ndarray, sigma_k: np.ndarray, z_typ: float = 0.02) -> float:
    """``lam_gp`` such that the quadratic cost ``(lam_gp/2) z' (k Sigma) z`` of a trade of size
    ``z_typ`` in one instrument matches, on average over instruments, the contract's convex cost
    ``a z + b z^1.5`` at that size."""
    target = 2.0 * (a * z_typ + b * z_typ**1.5) / z_typ**2  # l_i with (l_i/2) z^2 = c_i(z)
    return float(np.mean(target) / np.mean(np.diag(sigma_k)))
