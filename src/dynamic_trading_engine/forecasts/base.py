"""The forecast interface: a provider maps a decision state to a ``Forecast``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dynamic_trading_engine.state.view import DecisionState


@dataclass(frozen=True)
class Forecast:
    """Per-bar expected simple return of the next bar (``mu``), the standard error the provider
    attaches to it (``se``, 0 where it claims none), the per-bar decay of the expected return
    along the following bars (``decay`` in [0, 1]), and the ``oracle`` label."""

    mu: np.ndarray
    se: np.ndarray
    decay: float
    oracle: bool
    provider_id: str


class ForecastProvider(Protocol):
    provider_id: str
    oracle: bool

    def forecast(self, state: DecisionState) -> Forecast: ...
