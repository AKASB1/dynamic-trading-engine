"""Garleanu-Pedersen reference (R6) for one asset (or independent assets).

Maximize ``E sum_t delta^t [x_t mu_t - (g/2) x_t^2 - (Lam/2) (x_t - x_{t-1})^2]`` with
``delta = 1 - rho_gp`` and ``mu_{t+1} = (1 - phi_gp) mu_t + sigma_eps eps_{t+1}``.
The value function is ``V(xp, mu) = -A xp^2 / 2 + B xp mu + C mu^2 + D``; ``A`` is the positive
root of ``delta A^2 + (g + Lam (1 - delta)) A - Lam g = 0``,
``B = Lam / (g + Lam + delta A - Lam delta (1 - phi_gp))``; the policy is
``x = (1 - a) xp + a * aim * mu`` with ``a = (g + delta A) / (g + Lam + delta A)`` and
``aim = (1 + delta B (1 - phi_gp)) / (g + delta A)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GPParams:
    g: float = 0.08
    Lam: float = 0.4
    rho_gp: float = 0.02
    phi_gp: float = 0.3
    sigma_eps: float = 0.01

    @property
    def delta(self) -> float:
        return 1.0 - self.rho_gp


@dataclass(frozen=True)
class GPSolution:
    A: float
    B: float
    a: float
    aim: float
    residual: float


def gp_closed_form(p: GPParams) -> GPSolution:
    d, g, L, phi = p.delta, p.g, p.Lam, p.phi_gp
    b = g + L * (1.0 - d)
    # positive root, written to avoid cancellation: A = 2 c / (b + sqrt(b^2 + 4 d c)), c = L g
    c = L * g
    A = 2.0 * c / (b + math.sqrt(b * b + 4.0 * d * c))
    B = L / (g + L + d * A - L * d * (1.0 - phi))
    a = (g + d * A) / (g + L + d * A)
    aim = (1.0 + d * B * (1.0 - phi)) / (g + d * A)
    res = d * A * A + b * A - c
    return GPSolution(A, B, a, aim, res)


def stationary_sd(p: GPParams) -> float:
    r = 1.0 - p.phi_gp
    return p.sigma_eps / math.sqrt(1.0 - r * r)
