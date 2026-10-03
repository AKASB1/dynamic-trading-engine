"""Oracle forecast providers (labelled bounds, never competitors). They read the truth.

``oracle``: ``mu = m``, ``se = 0``, ``decay = phi``.
``noisy_oracle(rho)``: ``mu = rho * m + sqrt(1 - rho^2) * rms(m) * psi`` with ``psi`` standard
normal from the stream ``forecast.noise.<instrument_id>`` (the draw for a bar is a function of
the seed, the instrument, and the bar index only), ``rms(m)`` over the universe at that bar;
``se = sqrt(1 - rho^2) * rms(m)``, ``decay = phi``. Its pooled uncentred correlation with ``m``
is ``rho`` in expectation.
"""

from __future__ import annotations

import math

import numpy as np

from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.market.truth import Truth
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.state.view import DecisionState


class OracleForecast:
    oracle = True

    def __init__(self, truth: Truth):
        self.truth = truth
        self.provider_id = "oracle"
        self._col = {iid: j for j, iid in enumerate(truth.ids)}

    def _m(self, state: DecisionState) -> np.ndarray:
        cols = [self._col[i] for i in state.ids]
        return np.asarray(self.truth.m[state.bar, cols], float)

    def forecast(self, state: DecisionState) -> Forecast:
        m = self._m(state)
        return Forecast(m, np.zeros_like(m), float(self.truth.phi), True, self.provider_id)


class NoisyOracleForecast(OracleForecast):
    def __init__(self, truth: Truth, rho: float, seed: int):
        super().__init__(truth)
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        self.rho = float(rho)
        self.seed = int(seed)
        self.provider_id = f"noisy_oracle_{rho:g}"
        self._noise: dict[str, np.ndarray] = {}

    def _psi(self, iid: str, bar: int) -> float:
        a = self._noise.get(iid)
        if a is None or len(a) <= bar:
            n = max(bar + 1, len(self.truth.m))
            a = stream(self.seed, f"forecast.noise.{iid}").standard_normal(n)
            self._noise[iid] = a
        return float(a[bar])

    def forecast(self, state: DecisionState) -> Forecast:
        m = self._m(state)
        rms = math.sqrt(float(np.mean(m * m))) if len(m) else 0.0
        psi = np.array([self._psi(i, state.bar) for i in state.ids])
        a = math.sqrt(1.0 - self.rho * self.rho)
        mu = self.rho * m + a * rms * psi
        return Forecast(mu, np.full_like(m, a * rms), float(self.truth.phi), True, self.provider_id)
