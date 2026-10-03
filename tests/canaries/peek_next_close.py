"""D1: a forecast that reads the next bar's close through a memoized market handle."""

import numpy as np

from canaries._market_handle import handle
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

CANARY = {"id": "D1", "name": "peek_next_close_forecast", "guard": "replay"}


class PeekNextClose:
    provider_id = "peek_next_close"
    oracle = False

    def forecast(self, state):
        m = handle()
        k = min(state.bar + 1, m.n_bars - 1)
        col = {iid: j for j, iid in enumerate(m.ids)}
        mu = np.array(
            [
                m.close[k, col[i]] / m.close[state.bar, col[i]] - 1.0 if i in col else 0.0
                for i in state.ids
            ]
        )
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return Forecast(mu, np.zeros_like(mu), 0.0, False, self.provider_id)


class Strategy(PipelineStrategy):
    def __init__(self, costs, k):
        spec = StrategySpec("D1", forecast="none", optimizer=OptimizerConfig("rank_ls"), book="LS")
        super().__init__(spec, costs, k, provider=PeekNextClose())
