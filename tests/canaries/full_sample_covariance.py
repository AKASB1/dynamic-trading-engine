"""D2: a risk model that estimates the covariance on the whole sample, in its constructor."""

import numpy as np

from dynamic_trading_engine.market import registry
from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
from dynamic_trading_engine.risk.models import RiskEstimate, floor_psd
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

CANARY = {"id": "D2", "name": "full_sample_covariance", "guard": "replay"}


class FullSampleRisk:
    def __init__(self):
        m = registry.open_current()
        self.ids = m.ids
        R = np.where(np.isnan(m.returns), 0.0, m.returns)
        self.S = np.cov(R, rowvar=False)
        self.risk_id = "full_sample"

    def estimate(self, state, cols):
        idx = [self.ids.index(state.ids[c]) for c in cols]
        S, F = floor_psd(self.S[np.ix_(idx, idx)])
        return RiskEstimate(S, F, 0, {})


class Strategy(PipelineStrategy):
    def __init__(self, costs, k):
        spec = StrategySpec(
            "D2", forecast="none", optimizer=OptimizerConfig("min_variance"), book="LO"
        )
        super().__init__(spec, costs, k)
        self.risk_model = FullSampleRisk()
