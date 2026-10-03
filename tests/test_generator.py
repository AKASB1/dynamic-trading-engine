"""Check 14: the generator (statistics over seeds; each test names its estimator)."""

import dataclasses
import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.data import NO_TS
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import read_market, write_market

SEEDS = list(range(1, 21))


def _tree_hashes(d):
    out = {}
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), "rb") as fh:
            out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def test_byte_identical_stores_across_fresh_processes(tmp_path):
    code = (
        "import sys;from dynamic_trading_engine.market.config import load_market_config as L;"
        "from dynamic_trading_engine.market.generator import generate as G;"
        "from dynamic_trading_engine.market.store import write_market as W;"
        "W(G(L('tiny'), 5)[0], sys.argv[1])"
    )
    dirs = []
    for k in range(2):
        d = tmp_path / f"p{k}"
        env = dict(os.environ, PYTHONHASHSEED=str(k + 1))
        subprocess.run([sys.executable, "-c", code, str(d)], check=True, env=env, timeout=120)
        dirs.append(d)
    assert _tree_hashes(dirs[0]) == _tree_hashes(dirs[1])
    # and every file passes its loader (OHLC consistency, positive prices, sorting)
    m = read_market(str(dirs[0]))
    assert m.n == 8


def test_store_round_trip_is_bit_identical(tmp_path):
    m, _ = generate(load_market_config("tiny"), 3)
    write_market(m, str(tmp_path))
    m2 = read_market(str(tmp_path))
    for a in ("open", "high", "low", "close", "volume", "returns", "adv", "sigma_bar"):
        assert np.array_equal(getattr(m, a), getattr(m2, a), equal_nan=True), a
    assert m.corporate_actions == m2.corporate_actions
    assert np.array_equal(m.series_x.value[0], m2.series_x.value[0], equal_nan=True)


def test_different_seeds_differ_and_same_seed_repeats():
    cfg = load_market_config("tiny")
    a, _ = generate(cfg, 1)
    b, _ = generate(cfg, 1)
    c, _ = generate(cfg, 2)
    assert np.array_equal(a.close, b.close, equal_nan=True)
    assert not np.array_equal(a.close, c.close, equal_nan=True)


def test_returns_equal_drawn_returns_and_ohlc_valid():
    m, tr = generate(load_market_config("base"), 4)
    ok = ~np.isnan(m.returns)
    assert np.array_equal(ok, ~np.isnan(tr.r))
    assert np.max(np.abs(m.returns[ok] - tr.r[ok])) < 1e-12
    has = m.has_bar
    assert np.all(m.close[has] > 0) and np.all(m.open[has] > 0)
    assert np.all(m.high[has] >= np.maximum(m.open[has], m.close[has]))
    assert np.all(m.low[has] <= np.minimum(m.open[has], m.close[has]))


def _pooled_corr(a, b):
    """Pooled centred Pearson correlation over all finite pairs."""
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@pytest.mark.parametrize("name,ic", [("base", 0.02), ("null", 0.0)])
def test_signal_correlation_with_idiosyncratic_return(name, ic):
    cfg = load_market_config(name)
    est = []
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        sd = (tr.sig_pre[None, :] * tr.v[1 : cfg.n_bars, None]) * tr.vm[1 : cfg.n_bars, None]
        y = tr.e[1:] / sd
        x = tr.s[:-1]
        est.append(_pooled_corr(x, y))
    se = np.std(est, ddof=1) / math.sqrt(len(est))
    assert abs(np.mean(est) - ic) <= 4 * se, (np.mean(est), se)


def test_m_is_the_conditional_mean():
    cfg = load_market_config("base")
    m, tr = generate(cfg, 7)
    T = cfg.n_bars
    # formula: m[t] = ic * sigma * v[t+1] * s[t]
    want = cfg.ic * tr.sig_pre[None, :] * tr.v[1 : T + 1, None] * tr.s
    assert np.max(np.abs(tr.m - want)) < 1e-15
    # the residual e[t+1] - m[t] is uncorrelated with s[t] (pooled, 20 seeds, 4 standard errors)
    est = []
    for seed in SEEDS:
        _, tr = generate(cfg, seed)
        est.append(_pooled_corr(tr.s[:-1], tr.e[1:] - tr.m[:-1]))
    se = np.std(est, ddof=1) / math.sqrt(len(est))
    assert abs(np.mean(est)) <= 4 * se


def test_standardized_returns_variance_and_correlation():
    """(r - c_t - m_{t-1}) / true sd: variance 1 per instrument and the true correlation matrix
    (which differs by seed: the loadings are drawn), every entry within 5 standard errors of the
    estimator pooled over 20 seeds x 1259 bars (sum of per-seed deviations over the root of the
    sum of their squared standard errors; Pearson correlation per seed, ddof 1 variance)."""
    cfg = load_market_config("base")
    n = cfg.n_instruments
    dev = np.zeros((n, n))
    var_se = np.zeros((n, n))
    nobs = np.zeros((n, n))
    vdev = np.zeros(n)
    vse = np.zeros(n)
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        T = cfg.n_bars
        z = np.full((T, m.n), np.nan)
        for t in range(1, T):
            sd = np.sqrt(np.diag(tr.cov(t)))
            z[t] = (tr.r[t] - tr.cond_mean(t)) / sd
        z = z[1:]
        C = tr.cov(1)
        rho = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
        for i in range(n):
            zi = z[:, i][~np.isnan(z[:, i])]
            vdev[i] += np.var(zi, ddof=1) - 1.0
            vse[i] += 2.0 / len(zi)
            for j in range(i + 1, n):
                ok = ~np.isnan(z[:, i]) & ~np.isnan(z[:, j])
                if ok.sum() < 3:  # a late listing and an early delisting may not overlap
                    continue
                r = np.corrcoef(z[ok, i], z[ok, j])[0, 1]
                dev[i, j] += r - rho[i, j]
                var_se[i, j] += (1 - rho[i, j] ** 2) ** 2 / ok.sum()
                nobs[i, j] += ok.sum()
    assert np.all(np.abs(vdev) <= 5 * np.sqrt(vse))
    iu = np.triu_indices(n, 1)
    assert np.median(nobs[iu]) >= 20000
    assert np.all(np.abs(dev[iu]) <= 5 * np.sqrt(var_se[iu]))


def test_event_counts_match_rates_and_regime_occupancy():
    cfg = load_market_config("base")
    dl, sp, dv, stress = 0, 0, 0, []
    exp_dl = exp_sp = 0.0
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        listed_bars = m.has_bar.sum(axis=0)
        dl += int((m.ts_delist != NO_TS).sum())
        sp += sum(1 for r in m.corporate_actions if r[1] == "split")
        dv += sum(1 for r in m.corporate_actions if r[1] == "cash_dividend")
        exp_dl += (
            float(np.sum(np.maximum(listed_bars - 30, 0))) * cfg.delist_hazard_annual / cfg.ppy
        )
        exp_sp += float(np.sum(np.maximum(listed_bars - 15, 0))) * cfg.split_rate_annual / cfg.ppy
        stress.append(np.mean(tr.v[: cfg.n_bars] > 1))
    assert abs(dl - exp_dl) <= 4 * math.sqrt(exp_dl) + 1
    assert abs(sp - exp_sp) <= 4 * math.sqrt(exp_sp) + 1
    # quarterly dividends: about n * T / 63 per seed
    per_seed = dv / len(SEEDS)
    assert 0.8 * 30 * 1260 / 63 <= per_seed <= 1.05 * 30 * 1260 / 63
    want = cfg.p_calm_to_stress / (cfg.p_calm_to_stress + cfg.p_stress_to_calm)
    se = np.std(stress, ddof=1) / math.sqrt(len(stress))
    assert abs(np.mean(stress) - want) <= 4 * se + 0.01


def test_shift_market_identical_before_at_bar_and_moves_after():
    base, shift = load_market_config("base"), load_market_config("shift")
    at = shift.shift.at_bar
    vol_ratio, corr_ratio = [], []
    for seed in range(1, 11):
        mb, tb = generate(base, seed)
        ms, ts = generate(shift, seed)
        for a in ("open", "high", "low", "close", "volume", "returns", "adv", "sigma_bar"):
            assert np.array_equal(getattr(mb, a)[:at], getattr(ms, a)[:at], equal_nan=True), a
        assert np.array_equal(tb.s, ts.s) and np.array_equal(tb.m[: at - 1], ts.m[: at - 1])
        assert ts.ic_t[at] == pytest.approx(base.ic * shift.shift.ic_mult)
        rb, rs = mb.returns[at:], ms.returns[at:]
        vol_ratio.append(np.nanstd(rs) / np.nanstd(rb))
        Cb = (tb.B * tb.fvar) @ tb.B.T + np.diag(tb.sig_pre**2)
        Bs = ts.B * ts.lsc_post[:, None]
        Cs = (Bs * ts.fvar) @ Bs.T + np.diag(ts.sig_post**2)
        cb = Cb / np.sqrt(np.outer(np.diag(Cb), np.diag(Cb)))
        cs = Cs / np.sqrt(np.outer(np.diag(Cs), np.diag(Cs)))
        iu = np.triu_indices(mb.n, 1)
        corr_ratio.append(cs[iu].mean() / cb[iu].mean())
        # unconditional total variance (the ic term included) scales by vol_mult^2
        ub = (tb.B**2 * tb.fvar).sum(axis=1) + tb.sig_pre**2
        us = ts.lsc_post**2 * (ts.B**2 * ts.fvar).sum(axis=1) + ts.sig_post**2
        assert np.allclose(ts.vm[at] ** 2 * us, shift.shift.vol_mult**2 * ub, rtol=1e-12)
    assert np.allclose(corr_ratio, shift.shift.corr_mult, rtol=1e-9)
    assert abs(np.mean(vol_ratio) - shift.shift.vol_mult) < 0.1


def test_fundamental_law_on_clean():
    """Per-bar Sharpe of w = s / (sigma sqrt(N)) held close to close: ic sqrt(N), 4 SE over 20 seeds."""
    cfg = load_market_config("clean")
    srs = []
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        N = m.n
        w = tr.s[:-1] / (tr.sig_pre[None, :] * math.sqrt(N))
        R = np.nansum(w * tr.r[1:], axis=1)
        srs.append(R.mean() / R.std(ddof=1))
    se = np.std(srs, ddof=1) / math.sqrt(len(srs))
    assert abs(np.mean(srs) - cfg.ic * math.sqrt(cfg.n_instruments)) <= 4 * se


def test_signal_x_correlation_with_s():
    cfg = load_market_config("base")
    est = []
    for seed in SEEDS:
        m, tr = generate(cfg, seed)
        est.append(_pooled_corr(m.series_x.value[0], tr.s))
    se = np.std(est, ddof=1) / math.sqrt(len(est))
    want = 1 / math.sqrt(1 + cfg.observable_noise**2)
    assert abs(np.mean(est) - want) <= 4 * se


def test_market_files_pass_loaders(tmp_path):
    m, _ = generate(dataclasses.replace(load_market_config("tiny")), 11)
    write_market(m, str(tmp_path))
    for name, sch in (("bars_v1", S.BARS), ("liquidity_v1", S.LIQUIDITY), ("series_v1", S.SERIES)):
        with open(tmp_path / (name + ".csv"), "rb") as fh:
            assert len(S.load_bytes(sch, fh.read())) > 0
