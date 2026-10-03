"""D1b: a forecast that opens the market's contract files by path and reads the next close."""

import csv
import os

import numpy as np

from dynamic_trading_engine.contracts.timeutil import parse_ts
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.market import registry
from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

CANARY = {"id": "D1b", "name": "peek_by_path", "guard": "replay"}


class PeekByPath:
    provider_id = "peek_by_path"
    oracle = False

    def __init__(self):
        self._closes = None

    def _load(self):
        path = os.path.join(registry.current_dir(), "bars_v1.csv")
        out = {}
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out.setdefault(row["instrument_id"], []).append(
                    (parse_ts(row["ts_event"]), float(row["close"]))
                )
        return out

    def forecast(self, state):
        if self._closes is None:
            self._closes = self._load()
        mu = []
        for iid in state.ids:
            rows = self._closes.get(iid, [])
            nxt = [c for ts, c in rows if ts > state.t]
            cur = [c for ts, c in rows if ts <= state.t]
            mu.append(nxt[0] / cur[-1] - 1.0 if nxt and cur else 0.0)
        mu = np.array(mu)
        return Forecast(mu, np.zeros_like(mu), 0.0, False, self.provider_id)


class Strategy(PipelineStrategy):
    def __init__(self, costs, k):
        spec = StrategySpec("D1b", forecast="none", optimizer=OptimizerConfig("rank_ls"), book="LS")
        super().__init__(spec, costs, k, provider=PeekByPath())
