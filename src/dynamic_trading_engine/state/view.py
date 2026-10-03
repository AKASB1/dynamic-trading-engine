"""Knowledge-bounded view of a market (QC 3.1-3.3) and the ``DecisionState``.

``build_state(market, t, ...)`` is the only way the decision pipeline sees market data: it cuts
every array at the knowledge boundary (rows with ``ts_avail <= t``), copies it, marks it
read-only, and keeps no reference to the market (or to any truth). Window estimators use the
last ``h`` bars in which every instrument they need has a known return (``common_window``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamic_trading_engine.market.data import NO_TS, Market

DEFAULT_HISTORY = 520


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class PortfolioView:
    cash: float
    equity: float
    quantities: np.ndarray  # (n_u,) shares, universe order
    weights: np.ndarray  # (n_u,) value / equity at the closes of t
    prev_target: np.ndarray  # (n_u,) the previous decision's target weights (0 where none)


@dataclass(frozen=True)
class DecisionState:
    t: int  # decision instant, int64 microseconds UTC
    bar: int  # calendar index of the last bar with ts_event <= t
    ppy: int
    ids: tuple[str, ...]  # universe at t, instrument_id order
    sector: tuple[str, ...]
    lot_size: np.ndarray
    n_known: np.ndarray  # number of known bars per instrument
    returns: np.ndarray  # (L, n_u) simple total returns known at t, oldest row first, NaN unknown
    close: np.ndarray  # (L, n_u) raw closes known at t
    adj_close: np.ndarray  # (L, n_u) split-adjusted closes as of t (QC 3.2)
    tr_close: np.ndarray  # (L, n_u) total-return series as of t (QC 3.2)
    volume: np.ndarray  # (L, n_u)
    signal_x: np.ndarray  # (L, n_u) signal_x as of t (largest vintage known at t)
    sigma_hist: np.ndarray  # (L, n_u) liquidity sigma_bar rows known at t
    last_close: np.ndarray  # (n_u,)
    sigma_bar: np.ndarray  # (n_u,) latest known liquidity inputs (NaN if none)
    adv: np.ndarray  # (n_u,)
    portfolio: PortfolioView

    @property
    def n(self) -> int:
        return len(self.ids)


def universe_mask(market: Market, t: int) -> np.ndarray:
    """QC 3.3: listed at or before t and not delisted at or before t."""
    return (market.ts_list <= t) & (market.ts_delist > t)


def instruments_asof(market: Market, t: int) -> list[dict]:
    """Instrument rows known at t: ``ts_delist`` and ``delist_return`` read as empty before
    ``ts_delist``; instruments not yet listed are unknown."""
    out = []
    for j, iid in enumerate(market.ids):
        if market.ts_list[j] > t:
            continue
        dl = int(market.ts_delist[j])
        known = dl != NO_TS and dl <= t
        out.append(
            {
                "instrument_id": iid,
                "ts_list": int(market.ts_list[j]),
                "ts_delist": dl if known else None,
                "delist_return": float(market.delist_return[j]) if known else None,
                "sector": market.sector[j],
            }
        )
    return out


def known_corporate_actions(market: Market, t: int) -> list[tuple]:
    return [r for r in market.corporate_actions if r[3] <= t]


def series_asof(rows: list[tuple], series_id: str, ts_event: int, t: int):
    """QC 2.4: the value of the largest vintage with ts_avail <= t, or None."""
    best = None
    for sid, te, ta, vin, val in rows:
        if sid == series_id and te == ts_event and ta <= t and (best is None or vin > best[0]):
            best = (vin, val)
    return None if best is None else best[1]


def split_adjustment(market: Market, t: int, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Factor by which the raw closes of ``rows`` x ``cols`` are divided (splits with
    ``ts_ex <= t`` only: each divides the closes of the bars before its ex-bar)."""
    fac = np.ones((len(rows), len(cols)))
    col_of = {int(c): k for k, c in enumerate(cols)}
    idx = {iid: j for j, iid in enumerate(market.ids)}
    for iid, action, ts_ex, _avail, value in market.corporate_actions:
        if action != "split" or ts_ex > t:
            continue
        j = idx[iid]
        if j not in col_of:
            continue
        has = market.bar_avail[:, j] != NO_TS
        ex = np.nonzero(has & (market.ts_open >= ts_ex))[0]
        e = int(ex[0]) if len(ex) else market.n_bars
        fac[rows < e, col_of[j]] *= value
    return fac


def build_state(
    market: Market,
    t: int,
    cash: float,
    quantities: np.ndarray,
    prev_target: np.ndarray | None = None,
    history: int = DEFAULT_HISTORY,
) -> DecisionState:
    """Decision state at instant ``t``. ``quantities`` and ``prev_target`` are aligned with
    ``market.ids``; the state aligns them with the universe at t."""
    k = market.bar_index(t)
    if k < 0:
        raise ValueError("no bar is known at t")
    cols = np.nonzero(universe_mask(market, t))[0]
    lo = max(0, k - history + 1)
    rows = np.arange(lo, k + 1)
    sl = slice(lo, k + 1)
    bar_known = market.bar_avail[sl][:, cols] <= t
    close = np.where(bar_known, market.close[sl][:, cols], np.nan)
    volume = np.where(bar_known, market.volume[sl][:, cols], np.nan)
    ret_known = market.ret_avail[sl][:, cols] <= t
    rets = np.where(ret_known, market.returns[sl][:, cols], np.nan)
    liq_known = market.liq_avail[sl][:, cols] <= t
    sig_hist = np.where(liq_known, market.sigma_bar[sl][:, cols], np.nan)
    adv_hist = np.where(liq_known, market.adv[sl][:, cols], np.nan)
    x = market.series_x.asof(t, sl, cols)
    fac = split_adjustment(market, t, cols, rows)
    adj = close / fac
    # total-return series: cumulative product of known returns scaled to the last adjusted close
    tr = np.full_like(adj, np.nan)
    for c in range(len(cols)):
        known = ~np.isnan(adj[:, c])
        if not known.any():
            continue
        last = np.nonzero(known)[0][-1]
        first = np.nonzero(known)[0][0]
        g = np.ones(last - first + 1)
        rr = rets[first + 1 : last + 1, c]
        g[1:] = np.cumprod(1.0 + np.where(np.isnan(rr), 0.0, rr))
        tr[first : last + 1, c] = g / g[-1] * adj[last, c]
    n_known = np.count_nonzero(market.bar_avail[: k + 1][:, cols] <= t, axis=0)

    def latest(a):
        out = np.full(a.shape[1], np.nan)
        for c in range(a.shape[1]):
            nz = np.nonzero(~np.isnan(a[:, c]))[0]
            if len(nz):
                out[c] = a[nz[-1], c]
        return out

    last_close = latest(close)
    q = np.asarray(quantities, dtype=float)
    qu = q[cols]
    held_value = float(np.sum(np.where(q != 0, q * _last_closes_all(market, t, q), 0.0)))
    equity = float(cash) + held_value
    w0 = np.where(qu != 0, qu * last_close, 0.0) / equity if equity != 0 else np.zeros(len(cols))
    pt = np.zeros(len(cols)) if prev_target is None else np.asarray(prev_target, float)[cols]
    pf = PortfolioView(float(cash), equity, _ro(qu), _ro(w0), _ro(pt))
    return DecisionState(
        t=int(t),
        bar=int(k),
        ppy=int(market.ppy),
        ids=tuple(market.ids[c] for c in cols),
        sector=tuple(market.sector[c] for c in cols),
        lot_size=_ro(market.lot_size[cols]),
        n_known=_ro(n_known),
        returns=_ro(rets),
        close=_ro(close),
        adj_close=_ro(adj),
        tr_close=_ro(tr),
        volume=_ro(volume),
        signal_x=_ro(x),
        sigma_hist=_ro(sig_hist),
        last_close=_ro(last_close),
        sigma_bar=_ro(latest(sig_hist)),
        adv=_ro(latest(adv_hist)),
        portfolio=pf,
    )


def _last_closes_all(market: Market, t: int, q: np.ndarray) -> np.ndarray:
    """Last known close of every instrument with a position (marks of QC 3.7)."""
    k = market.bar_index(t)
    out = np.zeros(market.n)
    for j in np.nonzero(q != 0)[0]:
        col = market.close[: k + 1, j]
        known = (market.bar_avail[: k + 1, j] <= t) & ~np.isnan(col)
        nz = np.nonzero(known)[0]
        out[j] = col[nz[-1]] if len(nz) else 0.0
    return out


def common_window(returns: np.ndarray, cols: np.ndarray, max_window: int) -> int:
    """Number of latest rows in which every column of ``cols`` has a known return (at most
    ``max_window``)."""
    if len(cols) == 0:
        return 0
    r = returns[:, cols]
    ok = ~np.isnan(r).any(axis=1)
    if not ok[-1]:
        return 0
    bad = np.nonzero(~ok)[0]
    h = len(ok) if len(bad) == 0 else len(ok) - 1 - int(bad[-1])
    return min(h, max_window)
