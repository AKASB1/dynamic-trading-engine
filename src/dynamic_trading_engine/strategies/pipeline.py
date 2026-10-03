"""The built-in strategy: forecast provider + risk model + optimizer (or controller) + book.

It never receives the truth: oracle providers are constructed by the factory
(``strategies.factory``) and handed in as an object.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from dynamic_trading_engine.contracts.costs import CostConfig
from dynamic_trading_engine.engine.types import Decision
from dynamic_trading_engine.forecasts.base import ForecastProvider
from dynamic_trading_engine.forecasts.simple import NoForecast, PlainForecast
from dynamic_trading_engine.optimization.optimizers import NEEDS_RISK, Optimizer, OptimizerConfig
from dynamic_trading_engine.optimization.problems import BOOKS
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.state.view import DecisionState

NEEDS_FORECAST = (
    "rank_ls",
    "mean_variance",
    "cvar",
    "robust_box",
    "robust_ell",
    "mpc",
    "wdro",
    "gp_aim",
)


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str = "MV"
    forecast: str = "plain"  # none, plain, noisy_oracle, oracle
    rho: float = 0.5
    shrink: float = 1.0
    fwindow: int = 250
    risk: str = "lw"
    window: int = 250
    hl: float = 60.0
    n_pc: int = 3
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    book: str = "LS"
    min_history: int = 60

    @property
    def oracle(self) -> bool:
        return self.forecast in ("oracle", "noisy_oracle")


def optimized_columns(state: DecisionState, min_history: int) -> np.ndarray:
    """Instruments in the optimization: at least ``min_history`` known bars, a known close, and
    a known return at the decision bar (others get weight 0)."""
    n_known = np.asarray(state.n_known)
    last = np.asarray(state.last_close)
    r_last = np.asarray(state.returns)[-1] if state.returns.shape[0] else np.full(state.n, np.nan)
    ok = (n_known >= min_history) & np.isfinite(last) & (last > 0) & np.isfinite(r_last)
    return np.nonzero(ok)[0]


class PipelineStrategy:
    def __init__(
        self,
        spec: StrategySpec,
        costs: CostConfig,
        k: int,
        provider: ForecastProvider | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.spec = spec
        self.strategy_id = spec.strategy_id
        if provider is None:
            if spec.forecast == "plain":
                provider = PlainForecast(spec.fwindow, spec.shrink)
            elif spec.forecast == "none":
                provider = NoForecast()
            else:
                raise ValueError("oracle providers are built by the strategy factory")
        self.provider = provider
        self.forecast_id = provider.provider_id
        self.oracle = bool(provider.oracle)
        self.risk_model = RiskModel(spec.risk, spec.window, spec.hl, spec.n_pc)
        self.risk_id = f"{spec.risk}" if spec.optimizer.name in NEEDS_RISK else "none"
        self.optimizer = Optimizer(spec.optimizer, spec.book, costs, k, clock)
        self.gross_bound = BOOKS[spec.book].gross_bound
        self.last_solution = None

    def decide(self, state: DecisionState) -> Decision:
        name = self.spec.optimizer.name
        cols = optimized_columns(state, self.spec.min_history)
        fc = self.provider.forecast(state) if name in NEEDS_FORECAST else None
        risk = None
        if name in NEEDS_RISK and len(cols) >= 1:
            risk = self.risk_model.estimate(state, cols)
        sol = self.optimizer.solve(state, cols, fc, risk)
        self.last_solution = sol
        hold = not sol.success
        return Decision(
            weights=None if hold else sol.weights,
            status=sol.status,
            objective=sol.objective,
            hold=hold,
            wall_solve_ms=sol.wall_ms,
            max_violation=0.0 if hold else sol.max_violation,
        )
