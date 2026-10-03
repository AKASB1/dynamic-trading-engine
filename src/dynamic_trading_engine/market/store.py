"""Write a market to the contract's files (with manifests) and read it back.

Files: instruments, bars, corporate_actions, series (``<instrument_id>.signal_x`` and any
other series), returns, and liquidity, each ``<name>_v1.csv`` with its manifest, plus
``market.json`` (name, seed, generator, ppy, calendar). The truth is never written.
A market read back gives bit-identical arrays (shortest round-trip floats).
"""

from __future__ import annotations

import os

import numpy as np

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import (
    read_dataset_bytes,
    read_json,
    write_dataset,
    write_json,
)
from dynamic_trading_engine.market.calendar import equity_daily
from dynamic_trading_engine.market.data import NO_TS, Market, SeriesStore, derive

X_SUFFIX = ".signal_x"


def _opt(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def instrument_rows(m: Market) -> list[tuple]:
    rows = []
    for j, iid in enumerate(m.ids):
        dl = int(m.ts_delist[j])
        rows.append(
            (
                iid,
                m.symbol[j],
                m.asset_class[j],
                m.currency[j],
                int(m.ts_list[j]),
                None if dl == NO_TS else dl,
                None if dl == NO_TS else float(m.delist_return[j]),
                float(m.lot_size[j]),
                float(m.tick_size[j]),
                m.sector[j],
            )
        )
    return rows


def bar_rows(m: Market) -> list[tuple]:
    rows = []
    for j, iid in enumerate(m.ids):
        for t in np.nonzero(m.bar_avail[:, j] != NO_TS)[0]:
            rows.append(
                (
                    iid,
                    int(m.ts_open[t]),
                    int(m.ts_event[t]),
                    int(m.bar_avail[t, j]),
                    float(m.open[t, j]),
                    float(m.high[t, j]),
                    float(m.low[t, j]),
                    float(m.close[t, j]),
                    float(m.volume[t, j]),
                )
            )
    return rows


def series_rows(m: Market) -> list[tuple]:
    rows = list(m.other_series)
    sx = m.series_x
    for j, iid in enumerate(m.ids):
        sid = iid + X_SUFFIX
        for t in range(m.n_bars):
            for v, (val, av) in enumerate(zip(sx.value, sx.avail)):
                if av[t, j] != NO_TS:
                    rows.append((sid, int(m.ts_event[t]), int(av[t, j]), v, float(val[t, j])))
    rows.sort(key=lambda r: (r[0], r[1], r[3]))
    return rows


def return_rows(m: Market) -> list[tuple]:
    rows = []
    for t in range(m.n_bars):
        for j, iid in enumerate(m.ids):
            if m.ret_avail[t, j] != NO_TS:
                rows.append(
                    (int(m.ts_event[t]), int(m.ret_avail[t, j]), iid, float(m.returns[t, j]))
                )
    return rows


def liquidity_rows(m: Market) -> list[tuple]:
    rows = []
    for t in range(m.n_bars):
        for j, iid in enumerate(m.ids):
            if m.liq_avail[t, j] != NO_TS:
                rows.append(
                    (
                        int(m.ts_event[t]),
                        int(m.liq_avail[t, j]),
                        iid,
                        float(m.adv[t, j]),
                        float(m.sigma_bar[t, j]),
                    )
                )
    return rows


DATASETS = (
    ("instruments_v1", S.INSTRUMENTS, instrument_rows),
    ("bars_v1", S.BARS, bar_rows),
    ("corporate_actions_v1", S.CORPORATE_ACTIONS, lambda m: list(m.corporate_actions)),
    ("series_v1", S.SERIES, series_rows),
    ("returns_v1", S.RETURNS, return_rows),
    ("liquidity_v1", S.LIQUIDITY, liquidity_rows),
)


def write_market(m: Market, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for name, schema, rows_of in DATASETS:
        rows = rows_of(m)
        data = S.dump_bytes(schema, rows)
        write_dataset(os.path.join(directory, name + ".csv"), data, len(rows), m.generator, m.seed)
    write_json(
        os.path.join(directory, "market.json"),
        {
            "calendar": "equity_daily",
            "calendar_start": m.calendar_start,
            "generator": m.generator,
            "n_bars": m.n_bars,
            "name": m.name,
            "ppy": m.ppy,
            "seed": m.seed,
        },
    )


def _load(directory: str, name: str, schema, check: bool = True) -> S.Table:
    path = os.path.join(directory, name + ".csv")
    return S.load_bytes(schema, read_dataset_bytes(path, check_manifest=check))


def read_market(directory: str) -> Market:
    """Read a market directory. Without ``market.json`` (plain contract files such as the
    golden fixtures) the calendar is the set of bar instants and manifests are not required."""
    meta_path = os.path.join(directory, "market.json")
    check = os.path.exists(meta_path)
    if check:
        meta = read_json(meta_path)
        ts_open, ts_event = equity_daily(meta["calendar_start"], int(meta["n_bars"]))
    else:
        pairs = sorted({(r[2], r[1]) for r in _load(directory, "bars_v1", S.BARS, False).rows})
        ts_event = np.array([p[0] for p in pairs], dtype=np.int64)
        ts_open = np.array([p[1] for p in pairs], dtype=np.int64)
        meta = {
            "name": os.path.basename(os.path.normpath(directory)),
            "seed": None,
            "generator": {"name": "files", "version": 1, "params": {}},
            "ppy": 252,
            "calendar_start": "",
        }
    inst = _load(directory, "instruments_v1", S.INSTRUMENTS, check).rows
    ids = tuple(r[0] for r in inst)
    col = {iid: j for j, iid in enumerate(ids)}
    T, n = len(ts_event), len(ids)
    row_of = {int(t): k for k, t in enumerate(ts_event)}
    shape = (T, n)
    arrs = {k: np.full(shape, np.nan) for k in ("open", "high", "low", "close", "volume")}
    bar_avail = np.full(shape, NO_TS, dtype=np.int64)
    for iid, _o, te, ta, o, h, lo, c, v in _load(directory, "bars_v1", S.BARS, check).rows:
        t, j = row_of[te], col[iid]
        arrs["open"][t, j], arrs["high"][t, j], arrs["low"][t, j] = o, h, lo
        arrs["close"][t, j], arrs["volume"][t, j] = c, v
        bar_avail[t, j] = ta
    ca = list(_load(directory, "corporate_actions_v1", S.CORPORATE_ACTIONS, check).rows)
    other, xval, xav = [], [], []
    for sid, te, ta, vin, val in _load(directory, "series_v1", S.SERIES, check).rows:
        if sid.endswith(X_SUFFIX) and sid[: -len(X_SUFFIX)] in col and te in row_of:
            while len(xval) <= vin:
                xval.append(np.full(shape, np.nan))
                xav.append(np.full(shape, NO_TS, dtype=np.int64))
            t, j = row_of[te], col[sid[: -len(X_SUFFIX)]]
            xval[vin][t, j] = val
            xav[vin][t, j] = ta
        else:
            other.append((sid, te, ta, vin, val))
    if not xval:
        xval, xav = [np.full(shape, np.nan)], [np.full(shape, NO_TS, dtype=np.int64)]
    ret = np.full(shape, np.nan)
    ret_av = np.full(shape, NO_TS, dtype=np.int64)
    for te, ta, iid, rv in _load(directory, "returns_v1", S.RETURNS, check).rows:
        t, j = row_of[te], col[iid]
        ret[t, j], ret_av[t, j] = rv, ta
    adv = np.full(shape, np.nan)
    sig = np.full(shape, np.nan)
    liq_av = np.full(shape, NO_TS, dtype=np.int64)
    for te, ta, iid, a, sg in _load(directory, "liquidity_v1", S.LIQUIDITY, check).rows:
        t, j = row_of[te], col[iid]
        adv[t, j], sig[t, j], liq_av[t, j] = a, sg, ta
    m = Market(
        name=meta["name"],
        seed=meta["seed"],
        generator=meta["generator"],
        ppy=int(meta["ppy"]),
        calendar_start=meta["calendar_start"],
        ts_open=ts_open,
        ts_event=ts_event,
        ids=ids,
        ts_list=np.array([r[4] for r in inst], dtype=np.int64),
        ts_delist=np.array([NO_TS if r[5] is None else r[5] for r in inst], dtype=np.int64),
        delist_return=np.array([np.nan if r[6] is None else r[6] for r in inst]),
        lot_size=np.array([r[7] for r in inst]),
        tick_size=np.array([r[8] for r in inst]),
        sector=tuple(r[9] or "" for r in inst),
        symbol=tuple(r[1] for r in inst),
        asset_class=tuple(r[2] for r in inst),
        currency=tuple(r[3] for r in inst),
        bar_avail=bar_avail,
        corporate_actions=ca,
        series_x=SeriesStore("signal_x", xval, xav),
        other_series=other,
        **arrs,
    )
    return derive(m, liquidity_from=(adv, sig, liq_av), returns_from=(ret, ret_av))
