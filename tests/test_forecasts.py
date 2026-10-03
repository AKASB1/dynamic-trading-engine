"""Check 13: forecast providers (estimators named; standard errors from the spread over seeds)."""

import math

import numpy as np
import pytest

from dynamic_trading_engine.forecasts.oracle import NoisyOracleForecast, OracleForecast
from dynamic_trading_engine.forecasts.simple import NoForecast, PlainForecast
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.state.view import build_state

SEEDS = range(1, 21)
STEP = 3


def _states(m, start=260, step=STEP):
    for k in range(start, m.n_bars - 1, step):
        yield k, build_state(m, int(m.ts_event[k]), 1e6, np.zeros(m.n), history=300)


def test_oracle_equals_m():
    m, tr = generate(load_market_config("tiny"), 1)
    o = OracleForecast(tr)
    for k, st in _states(m, 100, 7):
        f = o.forecast(st)
        cols = [m.ids.index(i) for i in st.ids]
        assert np.max(np.abs(f.mu - tr.m[k, cols])) <= 1e-12
        assert f.oracle and f.decay == tr.phi


def test_none_is_zero():
    m, _ = generate(load_market_config("tiny"), 1)
    st = build_state(m, int(m.ts_event[150]), 1e6, np.zeros(m.n))
    f = NoForecast().forecast(st)
    assert not f.mu.any() and not f.oracle


@pytest.mark.parametrize("rho", [0.25, 0.5, 0.75])
def test_noisy_oracle_quality(rho):
    """Pooled uncentred correlation sum(mu m) / sqrt(sum mu^2 sum m^2) over instruments and bars,
    within 4 standard errors of rho (standard error from the spread across 20 seeds)."""
    cfg = load_market_config("tiny")
    est = []
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        p = NoisyOracleForecast(tr, rho, seed)
        num = s_mu = s_m = 0.0
        for k, st in _states(m, 100, 1):
            f = p.forecast(st)
            mm = tr.m[k, [m.ids.index(i) for i in st.ids]]
            num += float(f.mu @ mm)
            s_mu += float(f.mu @ f.mu)
            s_m += float(mm @ mm)
        est.append(num / math.sqrt(s_mu * s_m))
    se = np.std(est, ddof=1) / math.sqrt(len(est))
    assert abs(np.mean(est) - rho) <= 4 * se, (np.mean(est), se)


def _ic(m, provider):
    """Pooled Pearson correlation of mu at bar k with the next bar's standardized return
    r_{k+1} / sigma_bar_k, over the bars from 260 on in steps of 3."""
    xs, ys = [], []
    for k, st in _states(m):
        f = provider.forecast(st)
        cols = [m.ids.index(i) for i in st.ids]
        y = m.returns[k + 1, cols] / np.asarray(st.sigma_bar)
        ok = np.isfinite(y) & np.isfinite(f.mu)
        xs.append(f.mu[ok])
        ys.append(y[ok])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(np.corrcoef(x, y)[0, 1])


def test_plain_ic_null_base_and_below_oracle():
    ic_null, ic_base, ic_orc = [], [], []
    for seed in SEEDS:
        mn, _ = generate(load_market_config("null"), seed)
        ic_null.append(_ic(mn, PlainForecast()))
        mb, tb = generate(load_market_config("base"), seed)
        ic_base.append(_ic(mb, PlainForecast()))
        ic_orc.append(_ic(mb, OracleForecast(tb)))
    se_null = np.std(ic_null, ddof=1) / math.sqrt(len(ic_null))
    assert abs(np.mean(ic_null)) <= 4 * se_null
    assert np.mean(ic_base) > 0
    assert np.mean(ic_base) < np.mean(ic_orc)


def test_plain_decay_understates_persistence():
    m, tr = generate(load_market_config("base"), 2)
    st = build_state(m, int(m.ts_event[600]), 1e6, np.zeros(m.n))
    f = PlainForecast().forecast(st)
    assert 0 <= f.decay < tr.phi
    assert (f.se >= 0).all()
