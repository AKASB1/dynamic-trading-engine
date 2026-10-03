"""The consumer side of the contract's export format (version 1; Tier 2).

An export directory holds ``manifest.json`` and ``instruments.csv``, ``bars.csv``,
``corporate_actions.csv``, ``returns.csv``, ``signals.csv``, ``forecasts.csv``, ``liquidity.csv``
(each a v1 schema with its own ``.manifest.json``; ``series.csv`` optional). The reader validates
every listed file against its schema and against the hash in ``manifest.json`` and in its own
manifest, and checks the knowledge rule (no row has ``ts_avail`` before ``ts_event``). The
market built from it feeds the rolling loop; ``signals.csv`` rows named ``signal_x`` become the
``signal_x`` series, and ``forecasts.csv`` (horizon 1) is available to the ``export`` forecast
provider. ``write_export`` produces such a directory from a market of this repository (for the
tests; the files carry this generator's name).
"""

from __future__ import annotations

import os

import numpy as np

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import (
    read_json,
    sha256_bytes,
    write_dataset,
    write_json,
)
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.market.data import NO_TS, Market, SeriesStore, derive
from dynamic_trading_engine.market.store import (
    bar_rows,
    instrument_rows,
    liquidity_rows,
    return_rows,
)

FILES = {
    "instruments": S.INSTRUMENTS,
    "bars": S.BARS,
    "corporate_actions": S.CORPORATE_ACTIONS,
    "returns": S.RETURNS,
    "signals": S.SIGNALS,
    "forecasts": S.FORECASTS,
    "liquidity": S.LIQUIDITY,
}
OPTIONAL = {"series": S.SERIES}


class ExportError(ValueError):
    pass


def write_export(
    m: Market, directory: str, forecasts: list[tuple] | None = None, producer_commit: str = ""
) -> None:
    os.makedirs(directory, exist_ok=True)
    sig = []
    for t in range(m.n_bars):
        for j, iid in enumerate(m.ids):
            av = m.series_x.avail[0][t, j]
            if av != NO_TS:
                sig.append(
                    (int(m.ts_event[t]), int(av), iid, "signal_x", float(m.series_x.value[0][t, j]))
                )
    sig.sort(key=lambda r: (r[0], r[3], r[2]))
    rows = {
        "instruments": instrument_rows(m),
        "bars": bar_rows(m),
        "corporate_actions": list(m.corporate_actions),
        "returns": return_rows(m),
        "signals": sig,
        "forecasts": sorted(forecasts or [], key=lambda r: (r[0], r[2], r[3])),
        "liquidity": liquidity_rows(m),
    }
    listing = {}
    for name, schema in FILES.items():
        data = S.dump_bytes(schema, rows[name])
        write_dataset(
            os.path.join(directory, name + ".csv"), data, len(rows[name]), m.generator, m.seed
        )
        listing[name + ".csv"] = sha256_bytes(data)
    write_json(
        os.path.join(directory, "manifest.json"),
        {
            "export_version": 1,
            "files": listing,
            "calendar": "equity_daily",
            "calendar_start": m.calendar_start,
            "n_bars": m.n_bars,
            "ppy": m.ppy,
            "seed": m.seed,
            "generator": m.generator,
            "producer_commit": producer_commit,
            "derived_only": False,
            "data": "synthetic",
        },
    )


def read_export(directory: str) -> tuple[Market, dict, dict]:
    """(market, manifest, forecasts by (ts_event, instrument_id) -> (ts_avail, mu)); raises
    ExportError on any schema, hash, or knowledge violation."""
    man = read_json(os.path.join(directory, "manifest.json"))
    if man.get("export_version") != 1:
        raise ExportError("unsupported export_version")
    tables = {}
    for name, schema in {**FILES, **OPTIONAL}.items():
        fn = name + ".csv"
        path = os.path.join(directory, fn)
        if name in OPTIONAL and not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        want = man["files"].get(fn)
        if want is None or want != sha256_bytes(data):
            raise ExportError(f"{fn}: content hash does not match manifest.json")
        own = read_json(path[:-4] + ".manifest.json")
        if own.get("content_sha256") != sha256_bytes(data) or own.get("qc_version") != 1:
            raise ExportError(f"{fn}: its own manifest does not match")
        try:
            tables[name] = S.load_bytes(schema, data).dicts()
        except S.ContractError as e:
            raise ExportError(str(e)) from e
        for r in tables[name]:
            if "ts_avail" in r and "ts_event" in r and r["ts_avail"] < r["ts_event"]:
                raise ExportError(f"{fn}: a row is available before its event")
    if man.get("derived_only") and tables["bars"]:
        raise ExportError("derived_only exports carry header-only bars")
    from dynamic_trading_engine.market.calendar import equity_daily

    ts_open, ts_event = equity_daily(man["calendar_start"], int(man["n_bars"]))
    inst = tables["instruments"]
    ids = tuple(r["instrument_id"] for r in inst)
    col = {iid: j for j, iid in enumerate(ids)}
    row_of = {int(t): k for k, t in enumerate(ts_event)}
    T, n = len(ts_event), len(ids)
    arr = {k: np.full((T, n), np.nan) for k in ("open", "high", "low", "close", "volume")}
    bar_avail = np.full((T, n), NO_TS, dtype=np.int64)
    for r in tables["bars"]:
        t, j = row_of[r["ts_event"]], col[r["instrument_id"]]
        for k in arr:
            arr[k][t, j] = r[k]
        bar_avail[t, j] = r["ts_avail"]
    xv = np.full((T, n), np.nan)
    xa = np.full((T, n), NO_TS, dtype=np.int64)
    for r in tables["signals"]:
        if r["name"] == "signal_x" and r["ts_event"] in row_of:
            t, j = row_of[r["ts_event"]], col[r["instrument_id"]]
            xv[t, j], xa[t, j] = r["value"], r["ts_avail"]
    ret = np.full((T, n), np.nan)
    rav = np.full((T, n), NO_TS, dtype=np.int64)
    for r in tables["returns"]:
        t, j = row_of[r["ts_event"]], col[r["instrument_id"]]
        ret[t, j], rav[t, j] = r["ret"], r["ts_avail"]
    adv = np.full((T, n), np.nan)
    sg = np.full((T, n), np.nan)
    lav = np.full((T, n), NO_TS, dtype=np.int64)
    for r in tables["liquidity"]:
        t, j = row_of[r["ts_event"]], col[r["instrument_id"]]
        adv[t, j], sg[t, j], lav[t, j] = r["adv_shares"], r["sigma_bar"], r["ts_avail"]
    m = Market(
        name="export",
        seed=man.get("seed"),
        generator=man.get("generator", {}),
        ppy=int(man["ppy"]),
        calendar_start=man["calendar_start"],
        ts_open=ts_open,
        ts_event=ts_event,
        ids=ids,
        ts_list=np.array([r["ts_list"] for r in inst], dtype=np.int64),
        ts_delist=np.array(
            [NO_TS if r["ts_delist"] is None else r["ts_delist"] for r in inst], dtype=np.int64
        ),
        delist_return=np.array(
            [np.nan if r["delist_return"] is None else r["delist_return"] for r in inst]
        ),
        lot_size=np.array([r["lot_size"] for r in inst]),
        tick_size=np.array([r["tick_size"] for r in inst]),
        sector=tuple(r["sector"] or "" for r in inst),
        symbol=tuple(r["symbol"] for r in inst),
        asset_class=tuple(r["asset_class"] for r in inst),
        currency=tuple(r["currency"] for r in inst),
        bar_avail=bar_avail,
        corporate_actions=[tuple(r.values()) for r in tables["corporate_actions"]],
        series_x=SeriesStore("signal_x", [xv], [xa]),
        **arr,
    )
    derive(m, liquidity_from=(adv, sg, lav), returns_from=(ret, rav))
    fcs = {}
    for r in tables["forecasts"]:
        if r["horizon_bars"] == 1:
            fcs[(r["ts_event"], r["instrument_id"])] = (r["ts_avail"], r["mu"])
    return m, man, fcs


class ExportForecast:
    """The export's own forecasts (horizon 1) as known at the decision instant: for each
    instrument, the latest forecast row with ts_avail <= t; 0 where none."""

    oracle = False
    provider_id = "export"

    def __init__(self, forecasts: dict):
        self.by_inst = {}
        for (te, iid), (ta, mu) in sorted(forecasts.items()):
            self.by_inst.setdefault(iid, []).append((te, ta, mu))

    def forecast(self, state) -> Forecast:
        mu = np.zeros(state.n)
        for c, iid in enumerate(state.ids):
            best = None
            for te, ta, v in self.by_inst.get(iid, []):
                if ta <= state.t and (best is None or te >= best[0]):
                    best = (te, v)
            if best is not None:
                mu[c] = best[1]
        return Forecast(mu, np.zeros(state.n), 0.0, False, self.provider_id)
