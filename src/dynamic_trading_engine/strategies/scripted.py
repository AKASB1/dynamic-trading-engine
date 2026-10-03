"""Scripted strategies for tests: explicit orders or explicit target weights at given instants."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dynamic_trading_engine.engine.types import Decision
from dynamic_trading_engine.state.view import DecisionState


class ScriptedOrders:
    """Emits ``orders[t]`` (instrument id -> signed quantity) at decision instant ``t``."""

    forecast_id = "none"
    oracle = False
    risk_id = "none"

    def __init__(
        self,
        orders: dict[int, dict[str, float]],
        strategy_id: str = "scripted",
        gross_bound: float = 1e9,
    ):
        self.orders = orders
        self.strategy_id = strategy_id
        self.gross_bound = gross_bound

    def decide(self, state: DecisionState) -> Decision:
        return Decision(weights=None, orders=dict(self.orders.get(state.t, {})))


class ScriptedWeights:
    """Target weights from a function of the decision state."""

    forecast_id = "none"
    oracle = False
    risk_id = "none"

    def __init__(
        self,
        fn: Callable[[DecisionState], np.ndarray],
        strategy_id: str = "scripted_w",
        gross_bound: float = 2.0,
    ):
        self.fn = fn
        self.strategy_id = strategy_id
        self.gross_bound = gross_bound

    def decide(self, state: DecisionState) -> Decision:
        return Decision(weights=np.asarray(self.fn(state), float))
