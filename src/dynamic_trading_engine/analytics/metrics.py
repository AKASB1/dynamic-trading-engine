"""Metrics of the contract (section 6) and the declared criterion ``ce_ann``.

All per-bar quantities; ``ppy`` from the configuration. The criterion is the only quantity that
selects a configuration or orders a table of strategies:
``ce_ann = ppy * (mean(r) - (gamma_ce / 2) * var(r, ddof=1))`` with ``gamma_ce = 6``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

GAMMA_CE = 6.0
FLAT_GROSS = 1e-4  # a bar whose gross exposure at the close is below this holds no position
EULER_GAMMA = 0.5772156649015329


def ce_ann(mean: float, var: float, ppy: int, gamma_ce: float = GAMMA_CE) -> float:
    """The declared criterion from the mean and the ddof-1 variance of the net per-bar returns."""
    return ppy * (mean - 0.5 * gamma_ce * var)


def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, float)
    return e[1:] / e[:-1] - 1.0


def sharpe_bar(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return float(np.mean(r)) / sd if sd > 0 else 0.0


def skew_kurt(r: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(r))
    s = float(np.sqrt(np.mean((r - m) ** 2)))
    if s == 0:
        return 0.0, 3.0
    return float(np.mean((r - m) ** 3) / s**3), float(np.mean((r - m) ** 4) / s**4)


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    return float(np.max((peak - e) / peak))


def cvar95(r: np.ndarray) -> float:
    k = math.ceil(0.05 * len(r))
    return float(-np.mean(np.sort(r)[:k])) if k else 0.0


def psr(sr: float, sr_star: float, t_bars: int, skew: float, kurt: float) -> float:
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return float(norm.cdf((sr - sr_star) * math.sqrt(t_bars - 1) / math.sqrt(den)))


def expected_max_sr(V: float, N: int) -> float:
    if N < 2 or V <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))


def dsr(sr: float, t_bars: int, skew: float, kurt: float, V: float, N: int) -> float:
    return psr(sr, expected_max_sr(V, N), t_bars, skew, kurt)


def lo_eta(q: int, rho: np.ndarray, L: int = 10) -> float:
    """Lo (2002): eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q-k) rho_k), rho_k = 0 beyond L."""
    s = 0.0
    for k in range(1, q):
        if k > L or k > len(rho):
            break
        s += (q - k) * rho[k - 1]
    return q / math.sqrt(q + 2 * s)


def summarize(
    equity: np.ndarray,
    gross_pnl: np.ndarray,
    costs: np.ndarray,
    turnover: np.ndarray,
    gross_exp: np.ndarray,
    net_exp: np.ndarray,
    ppy: int,
    start: int,
) -> dict:
    """Metrics over the bars after bar ``start`` (the first decision): the returns of bars
    start+1 .. T-1 relative to the previous equity."""
    e = np.asarray(equity, float)[start:]
    r = returns_from_equity(e)
    T = len(r)
    prev = e[:-1]
    g = np.asarray(gross_pnl, float)[start + 1 :] / prev
    c = np.asarray(costs, float)[start + 1 :] / prev
    mean = float(np.mean(r)) if T else 0.0
    var = float(np.var(r, ddof=1)) if T > 1 else 0.0
    sk, ku = skew_kurt(r) if T else (0.0, 3.0)
    vol = math.sqrt(var * ppy)
    sd_g = float(np.std(g, ddof=1)) if T > 1 else 0.0
    return {
        "n_bars": T,
        "mean": mean,
        "var": var,
        "ce_ann": ce_ann(mean, var, ppy),
        "sharpe": mean / math.sqrt(var) * math.sqrt(ppy) if var > 0 else 0.0,
        "sharpe_gross": float(np.mean(g)) / sd_g * math.sqrt(ppy) if sd_g > 0 else 0.0,
        "sr_bar": mean / math.sqrt(var) if var > 0 else 0.0,
        "skew": sk,
        "kurt": ku,
        "ann_return": float((e[-1] / e[0]) ** (ppy / T) - 1.0) if T else 0.0,
        "ann_vol": vol,
        "max_drawdown": max_drawdown(e) if T else 0.0,
        "cvar95": cvar95(r) if T else 0.0,
        "turnover": float(np.mean(np.asarray(turnover, float)[start + 1 :])) if T else 0.0,
        "cost_drag": float(np.mean(c)) * ppy if T else 0.0,
        "gross_exposure": float(np.mean(np.asarray(gross_exp, float)[start + 1 :])) if T else 0.0,
        "net_exposure": float(np.mean(np.asarray(net_exp, float)[start + 1 :])) if T else 0.0,
        "flat_frac": float(np.mean(np.asarray(gross_exp, float)[start + 1 :] < FLAT_GROSS))
        if T
        else 0.0,
    }


def segment(equity: np.ndarray, start: int, lo: int, hi: int) -> tuple[int, float, float]:
    """(number of bars, mean, ddof-1 variance) of the per-bar returns of bars lo..hi-1
    (restricted to bars after ``start``)."""
    e = np.asarray(equity, float)
    lo = max(lo, start + 1)
    if hi - lo < 2:
        return max(hi - lo, 0), 0.0, 0.0
    r = e[lo:hi] / e[lo - 1 : hi - 1] - 1.0
    return len(r), float(np.mean(r)), float(np.var(r, ddof=1))
