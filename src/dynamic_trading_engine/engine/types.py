"""Interfaces between the rolling loop and the strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dynamic_trading_engine.state.view import DecisionState


@dataclass
class Decision:
    """Target weights aligned with ``state.ids`` (or explicit orders for scripted strategies)."""

    weights: np.ndarray | None
    orders: dict[str, float] | None = None
    status: str = "none"  # optimal, optimal_inaccurate, failed, none (no solve)
    objective: float | None = None
    hold: bool = False
    wall_solve_ms: float = 0.0
    max_violation: float = 0.0
    info: dict = field(default_factory=dict)


class Strategy(Protocol):
    strategy_id: str
    forecast_id: str
    oracle: bool
    risk_id: str
    gross_bound: float

    def decide(self, state: DecisionState) -> Decision: ...
