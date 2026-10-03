"""Random decision states and forecasts for the optimizer checks and the S1 experiment."""

import numpy as np

from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.state.view import DecisionState, PortfolioView


def random_state(g, n, L=260, equity=1e7, w0=None, sectors=3, missing=0):
    """A decision state with n instruments, L bars of random factor-structured returns, random
    prices and liquidity. ``missing`` instruments have short histories (excluded)."""
    k = 2
    B = g.normal(1.0, 0.3, (n, k))
    f = g.standard_normal((L, k)) * 0.01
    R = f @ B.T + g.standard_normal((L, n)) * g.uniform(0.008, 0.02, n)
    n_known = np.full(n, L)
    if missing:
        idx = g.choice(n, missing, replace=False)
        n_known[idx] = g.integers(5, 50, missing)
        for i in idx:
            R[: L - n_known[i], i] = np.nan
    P = g.uniform(20, 200, n)
    sig = g.uniform(0.01, 0.03, n)
    adv = g.uniform(5e5, 5e6, n)
    if w0 is None:
        w0 = np.zeros(n)
    x = g.standard_normal((L, n))
    sector = tuple(f"S{i % sectors}" for i in range(n))
    pf = PortfolioView(1e7 * 0.0, equity, w0 * equity / P, np.asarray(w0, float), np.zeros(n))

    def ro(a):
        a = np.array(a, copy=True)
        a.flags.writeable = False
        return a

    return DecisionState(
        t=0,
        bar=L - 1,
        ppy=252,
        ids=tuple(f"I{i:03d}" for i in range(n)),
        sector=sector,
        lot_size=ro(np.zeros(n)),
        n_known=ro(n_known),
        returns=ro(R),
        close=ro(np.tile(P, (L, 1))),
        adj_close=ro(np.tile(P, (L, 1))),
        tr_close=ro(np.tile(P, (L, 1))),
        volume=ro(np.tile(adv, (L, 1))),
        signal_x=ro(x),
        sigma_hist=ro(np.tile(sig, (L, 1))),
        last_close=ro(P),
        sigma_bar=ro(sig),
        adv=ro(adv),
        portfolio=pf,
    )


def random_forecast(g, n, scale=3e-4):
    mu = g.standard_normal(n) * scale
    se = np.abs(g.standard_normal(n)) * scale
    return Forecast(mu, se, float(g.uniform(0, 0.95)), False, "test")
