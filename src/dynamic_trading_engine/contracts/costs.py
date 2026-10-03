"""Cost model v1 of the contract (section 4): fills, accruals, and the optimizer's convex form."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ImpactConfig:
    model: str = "sqrt"  # sqrt, linear, none
    y: float = 0.5

    def __post_init__(self):
        if self.model not in ("sqrt", "linear", "none"):
            raise ValueError(f"unknown impact model {self.model!r}")


@dataclass(frozen=True)
class CostConfig:
    schema_version: int = 1
    commission_bps: float = 1.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    half_spread_bps: float = 2.0
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    borrow_bps_annual: float = 50.0
    financing_bps_annual: float = 100.0
    cash_rate_bps_annual: float = 0.0
    participation_cap: float = 0.1

    def scaled(self, factor: float) -> CostConfig:
        """Spread, impact, and commissions multiplied by ``factor`` (sensitivity runs)."""
        return CostConfig(
            schema_version=self.schema_version,
            commission_bps=self.commission_bps * factor,
            commission_per_share=self.commission_per_share * factor,
            min_commission=self.min_commission * factor,
            half_spread_bps=self.half_spread_bps * factor,
            impact=ImpactConfig(self.impact.model, self.impact.y * factor),
            borrow_bps_annual=self.borrow_bps_annual,
            financing_bps_annual=self.financing_bps_annual,
            cash_rate_bps_annual=self.cash_rate_bps_annual,
            participation_cap=self.participation_cap,
        )


def cost_config_from_dict(d: dict) -> CostConfig:
    d = dict(d)
    if "impact" in d:
        d["impact"] = ImpactConfig(**d["impact"])
    return CostConfig(**d)


@dataclass(frozen=True)
class FillCost:
    price: float
    spread_cost: float
    impact_cost: float
    commission: float
    impact_bps: float
    impact_unavailable: bool


def impact_bps(
    q: float, sigma: float | None, V: float | None, cfg: CostConfig
) -> tuple[float, bool]:
    if cfg.impact.model == "none":
        return 0.0, False
    if sigma is None or V is None or not math.isfinite(sigma) or not math.isfinite(V) or V <= 0:
        return 0.0, True
    a = abs(q)
    if cfg.impact.model == "sqrt":
        return 1e4 * cfg.impact.y * sigma * math.sqrt(a / V), False
    return 1e4 * cfg.impact.y * sigma * a / V, False


def fill_cost(
    q: float, m: float, sigma: float | None, V: float | None, cfg: CostConfig
) -> FillCost:
    """Costs of a fill of signed quantity ``q`` at reference price ``m`` (decision-time inputs)."""
    a = abs(q)
    ib, unavailable = impact_bps(q, sigma, V, cfg)
    spread = a * m * cfg.half_spread_bps / 1e4
    impact = a * m * ib / 1e4
    sgn = 1.0 if q > 0 else -1.0
    price = m * (1.0 + sgn * (cfg.half_spread_bps + ib) / 1e4)
    comm = 0.0
    if q != 0:
        comm = max(
            cfg.min_commission, a * m * cfg.commission_bps / 1e4 + a * cfg.commission_per_share
        )
    return FillCost(price, spread, impact, comm, ib, unavailable)


def accruals(
    q_prev: np.ndarray, p_prev: np.ndarray, cash_prev: float, dt_days: float, cfg: CostConfig
) -> tuple[float, float, float]:
    """(borrow, financing, interest) at a bar's close from the previous close's state."""
    short = q_prev < 0
    borrow = float(
        np.sum(np.abs(q_prev[short]) * p_prev[short])
        * cfg.borrow_bps_annual
        / 1e4
        * dt_days
        / 365.0
    )
    financing = max(0.0, -cash_prev) * cfg.financing_bps_annual / 1e4 * dt_days / 365.0
    interest = max(0.0, cash_prev) * cfg.cash_rate_bps_annual / 1e4 * dt_days / 365.0
    return borrow, financing, interest


def convex_coefficients(
    P: np.ndarray, sigma: np.ndarray, V: np.ndarray, E: float, cfg: CostConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    """Coefficients of the optimizer's cost in units of equity for a trade z (value / E):
    ``c(z) = sum a|z| + b|z|^1.5`` (impact ``sqrt``) or ``a|z| + 0.5 l z^2`` (``linear``: the
    returned ``b`` is then ``l``). Unknown liquidity gives zero impact, as in the simulator."""
    P = np.asarray(P, float)
    a = (cfg.half_spread_bps + cfg.commission_bps) / 1e4 + cfg.commission_per_share / P
    ok = np.isfinite(sigma) & np.isfinite(V) & (np.asarray(V, float) > 0)
    sig = np.where(ok, sigma, 0.0)
    Vs = np.where(ok, V, 1.0)
    if cfg.impact.model == "sqrt":
        b = np.where(ok, cfg.impact.y * sig * np.sqrt(E / (Vs * P)), 0.0)
        return a, b, "power15"
    if cfg.impact.model == "linear":
        l = np.where(ok, 2.0 * cfg.impact.y * sig * E / (Vs * P), 0.0)
        return a, l, "quadratic"
    return a, np.zeros_like(a), "power15"


def convex_cost(z: np.ndarray, a: np.ndarray, b: np.ndarray, form: str = "power15") -> float:
    z = np.abs(np.asarray(z, float))
    if form == "power15":
        return float(np.sum(a * z + b * z**1.5))
    return float(np.sum(a * z + 0.5 * b * z * z))
