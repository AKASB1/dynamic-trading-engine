"""Small hand-built markets for the accounting tests."""

import numpy as np

from dynamic_trading_engine.market.calendar import equity_daily
from dynamic_trading_engine.market.data import NO_TS, Market, SeriesStore, derive


def hand_market(
    closes: dict,
    opens: dict | None = None,
    volumes: dict | None = None,
    ca: list | None = None,
    delist: dict | None = None,
    lot: float = 0.0,
    start: str = "2024-03-04",
    sigma: float = 0.02,
    adv: float = 1_000_000.0,
    first_bar: dict | None = None,
):
    """closes: iid -> list of closes (NaN = no bar). ca rows: (iid, action, bar, announce_bar, value).
    delist: iid -> (last bar, delist_return). Liquidity: constant sigma and adv on every bar."""
    ids = tuple(sorted(closes))
    T = len(next(iter(closes.values())))
    ts_open, ts_event = equity_daily(start, T)
    n = len(ids)
    C = np.array([closes[i] for i in ids], float).T
    O = C.copy() if opens is None else np.array([opens[i] for i in ids], float).T
    V = np.full((T, n), 1e6) if volumes is None else np.array([volumes[i] for i in ids], float).T
    has = ~np.isnan(C)
    V = np.where(has, V, np.nan)
    H = np.where(has, np.maximum(O, C) * 1.01, np.nan)
    L = np.where(has, np.minimum(O, C) * 0.99, np.nan)
    avail = np.where(has, ts_event[:, None], NO_TS).astype(np.int64)
    first = {i: int(np.nonzero(has[:, j])[0][0]) for j, i in enumerate(ids)}
    delist = delist or {}
    ts_delist = np.array(
        [int(ts_event[delist[i][0]]) if i in delist else NO_TS for i in ids], np.int64
    )
    dret = np.array([delist[i][1] if i in delist else np.nan for i in ids])
    rows = []
    for iid, act, bar, ann, val in ca or []:
        rows.append((iid, act, int(ts_open[bar]), int(ts_event[ann]), float(val)))
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    m = Market(
        name="hand",
        seed=None,
        generator={"name": "hand", "version": 1, "params": {}},
        ppy=252,
        calendar_start=start,
        ts_open=ts_open,
        ts_event=ts_event,
        ids=ids,
        ts_list=np.array([int(ts_event[first[i]]) for i in ids], np.int64),
        ts_delist=ts_delist,
        delist_return=dret,
        lot_size=np.full(n, lot),
        tick_size=np.full(n, 0.01),
        sector=tuple("S0" for _ in ids),
        symbol=ids,
        asset_class=("equity",) * n,
        currency=("USD",) * n,
        open=np.where(has, O, np.nan),
        high=H,
        low=L,
        close=C,
        volume=V,
        bar_avail=avail,
        corporate_actions=rows,
        series_x=SeriesStore("signal_x", [np.where(has, 0.0, np.nan)], [avail.copy()]),
    )
    liq = (np.where(has, adv, np.nan), np.where(has, sigma, np.nan), avail.copy())
    return derive(m, liquidity_from=liq)


def truncate_market(m, k):
    """A copy of the market that ends at bar k: what exists after it is removed (bars, series,
    liquidity, corporate actions announced later, delistings not yet known)."""
    import dataclasses

    t = int(m.ts_event[k])
    sl = slice(0, k + 1)
    kw = {}
    for f in dataclasses.fields(m):
        v = getattr(m, f.name)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == m.n_bars:
            v = v[sl].copy()
        kw[f.name] = v
    kw["ts_open"] = m.ts_open[sl].copy()
    kw["ts_event"] = m.ts_event[sl].copy()
    kw["ts_delist"] = np.where(m.ts_delist <= t, m.ts_delist, NO_TS)
    kw["corporate_actions"] = [r for r in m.corporate_actions if r[3] <= t]
    kw["series_x"] = SeriesStore(
        "signal_x",
        [v[sl].copy() for v in m.series_x.value],
        [a[sl].copy() for a in m.series_x.avail],
    )
    return Market(**kw)
