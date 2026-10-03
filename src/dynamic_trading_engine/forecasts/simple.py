"""Honest forecast providers: ``none`` and the ``plain`` predictor from ``signal_x``."""

from __future__ import annotations

import math

import numpy as np

from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.state.view import DecisionState


class NoForecast:
    provider_id = "none"
    oracle = False

    def forecast(self, state: DecisionState) -> Forecast:
        z = np.zeros(state.n)
        return Forecast(z, z.copy(), 0.0, False, self.provider_id)


class PlainForecast:
    """``mu_i = shrink * b_t * sigma_i * x_i``.

    ``x_i`` is the latest known ``signal_x``, ``sigma_i`` the latest ``sigma_bar``, and ``b_t`` the
    slope of the pooled regression through the origin of the standardized next-bar return
    ``r_{i,u+1} / sigma_{i,u}`` on ``x_{i,u}`` over the last ``W`` bars in which both are known
    at t. ``se_i = SE(b_t) * sigma_i * |x_i|`` with the standard error of the slope;
    ``decay`` is the uncentred lag-1 autocorrelation of ``x`` pooled over the instruments,
    floored at 0."""

    oracle = False

    def __init__(self, window: int = 250, shrink: float = 1.0):
        self.window = int(window)
        self.shrink = float(shrink)
        self.provider_id = "plain"

    def forecast(self, state: DecisionState) -> Forecast:
        x_hist = np.asarray(state.signal_x)
        r = np.asarray(state.returns)
        sg = np.asarray(state.sigma_hist)
        W = self.window
        L = len(r)
        npair = max(0, min(W, L - 1))
        X = x_hist[L - 1 - npair : L - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = r[L - npair : L] / sg[L - 1 - npair : L - 1]
        ok = np.isfinite(X) & np.isfinite(Y)
        xs, ys = X[ok], Y[ok]
        sxx = float(np.dot(xs, xs))
        if len(xs) >= 3 and sxx > 0:
            b = float(np.dot(xs, ys)) / sxx
            resid = ys - b * xs
            s2 = float(np.dot(resid, resid)) / (len(xs) - 1)
            se_b = math.sqrt(s2 / sxx)
        else:
            b, se_b = 0.0, 0.0
        x = x_hist[-1] if len(x_hist) else np.zeros(state.n)
        sig = np.asarray(state.sigma_bar)
        good = np.isfinite(x) & np.isfinite(sig)
        mu = np.where(good, self.shrink * b * sig * x, 0.0)
        se = np.where(good, se_b * sig * np.abs(x), 0.0)
        xw = x_hist[-W:]
        a, c = xw[1:], xw[:-1]
        okp = np.isfinite(a) & np.isfinite(c)
        den = float(np.dot(c[okp], c[okp]))
        decay = max(0.0, float(np.dot(a[okp], c[okp])) / den) if den > 0 else 0.0
        return Forecast(mu, se, min(decay, 1.0), False, self.provider_id)
