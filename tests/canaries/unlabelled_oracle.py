"""D3: a provider that returns a near copy of the truth's conditional mean with oracle False."""

import numpy as np

from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

CANARY = {"id": "D3", "name": "unlabelled_oracle", "guard": "label"}


class UnlabelledOracle:
    provider_id = "honest_looking"
    oracle = False

    def __init__(self, truth):
        self.truth = truth
        self.col = {iid: j for j, iid in enumerate(truth.ids)}

    def forecast(self, state):
        m = np.array([self.truth.m[state.bar, self.col[i]] for i in state.ids])
        mu = m * 1.001
        return Forecast(mu, np.zeros_like(mu), 0.9, False, self.provider_id)


class Strategy(PipelineStrategy):
    def __init__(self, costs, k, truth):
        spec = StrategySpec("D3", forecast="none", optimizer=OptimizerConfig("rank_ls"), book="LS")
        super().__init__(spec, costs, k, provider=UnlabelledOracle(truth))
