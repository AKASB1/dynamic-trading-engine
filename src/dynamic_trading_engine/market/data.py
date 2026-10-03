"""The market as the contract's datasets, held as dense (bar x instrument) arrays.

All instruments share one calendar. Missing bars are NaN (prices, volume) and ``NO_TS``
(availability). This module never imports the truth module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamic_trading_engine.market.liquidity import liquidity

NO_TS = np.iinfo(np.int64).max


@dataclass
class SeriesStore:
    """A per-instrument series family (``<instrument_id>.<name>``) with vintages.

    ``value[v]`` and ``avail[v]`` are (bar x instrument) arrays for vintage ``v``; ``avail`` is
    ``NO_TS`` where that vintage does not exist. The value of a period as of ``t`` is the one of
    its largest vintage with ``avail <= t`` (QC 2.4)."""

    name: str
    value: list[np.ndarray]
    avail: list[np.ndarray]

    def asof(self, t: int, rows: slice, cols: np.ndarray | slice = slice(None)) -> np.ndarray:
        out = np.full(self.value[0][rows][:, cols].shape, np.nan)
        for val, av in zip(self.value, self.avail):
            known = av[rows][:, cols] <= t
            out = np.where(known, val[rows][:, cols], out)
        return out

    def copy(self) -> SeriesStore:
        return SeriesStore(
            self.name, [v.copy() for v in self.value], [a.copy() for a in self.avail]
        )


@dataclass
class Market:
    name: str
    seed: int | None
    generator: dict
    ppy: int
    calendar_start: str
    ts_open: np.ndarray  # (T,) int64 us
    ts_event: np.ndarray  # (T,)
    ids: tuple[str, ...]
    ts_list: np.ndarray  # (n,) int64
    ts_delist: np.ndarray  # (n,) int64, NO_TS when none
    delist_return: np.ndarray  # (n,) float, NaN when none
    lot_size: np.ndarray  # (n,)
    tick_size: np.ndarray  # (n,)
    sector: tuple[str, ...]
    symbol: tuple[str, ...]
    asset_class: tuple[str, ...]
    currency: tuple[str, ...]
    open: np.ndarray  # (T, n) NaN where no bar
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    bar_avail: np.ndarray  # (T, n) int64, NO_TS where no bar
    corporate_actions: list[tuple]  # rows of corporate_actions_v1, sorted
    series_x: SeriesStore  # signal_x
    other_series: list[tuple] = field(default_factory=list)  # other series_v1 rows
    # derived (filled by ``derive``)
    split_ratio: np.ndarray | None = None  # (T, n), 1 where none
    dividend: np.ndarray | None = None  # (T, n), 0 where none
    returns: np.ndarray | None = None  # (T, n) NaN where none
    ret_avail: np.ndarray | None = None
    adv: np.ndarray | None = None  # liquidity inputs (T, n), NaN where no row
    sigma_bar: np.ndarray | None = None
    liq_avail: np.ndarray | None = None

    @property
    def n_bars(self) -> int:
        return len(self.ts_event)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def has_bar(self) -> np.ndarray:
        return self.bar_avail != NO_TS

    def bar_index(self, t: int) -> int:
        """Index of the last calendar bar with ``ts_event <= t`` (-1 if none)."""
        return int(np.searchsorted(self.ts_event, t, side="right")) - 1


def ex_bar_index(market: Market, i: int, ts_ex: int) -> int | None:
    """The first bar of instrument ``i`` with ``ts_open >= ts_ex`` (None if it has no such bar)."""
    has = market.bar_avail[:, i] != NO_TS
    cand = np.nonzero(has & (market.ts_open >= ts_ex))[0]
    return int(cand[0]) if len(cand) else None


def corporate_action_arrays(market: Market) -> tuple[np.ndarray, np.ndarray]:
    T, n = market.n_bars, market.n
    ratio = np.ones((T, n))
    div = np.zeros((T, n))
    col = {iid: j for j, iid in enumerate(market.ids)}
    for iid, action, ts_ex, _avail, value in market.corporate_actions:
        j = col[iid]
        k = ex_bar_index(market, j, ts_ex)
        if k is None:
            continue
        if action == "split":
            ratio[k, j] = value
        else:
            div[k, j] = value
    return ratio, div


def compute_returns(market: Market) -> tuple[np.ndarray, np.ndarray]:
    """Simple total returns of QC 2.5 for every bar after an instrument's first bar."""
    ratio, div = market.split_ratio, market.dividend
    c = market.close
    prev = np.full_like(c, np.nan)
    prev[1:] = c[:-1]
    has = market.has_bar
    has_prev = np.zeros_like(has)
    has_prev[1:] = has[:-1]
    ok = has & has_prev
    with np.errstate(invalid="ignore"):
        r = (c * ratio + div) / prev - 1.0
    r = np.where(ok, r, np.nan)
    avail = np.where(ok, market.bar_avail, NO_TS)
    return r, avail


def derive(market: Market, liquidity_from: tuple | None = None, returns_from: tuple | None = None):
    """Fill the derived arrays. ``liquidity_from`` / ``returns_from`` are (values..., avail)
    read from files; otherwise they are computed from the bars and corporate actions."""
    market.split_ratio, market.dividend = corporate_action_arrays(market)
    if returns_from is not None:
        market.returns, market.ret_avail = returns_from
    else:
        market.returns, market.ret_avail = compute_returns(market)
    if liquidity_from is not None:
        market.adv, market.sigma_bar, market.liq_avail = liquidity_from
    else:
        market.adv, market.sigma_bar, market.liq_avail = liquidity(
            market.volume, market.bar_avail, market.returns, market.split_ratio, NO_TS
        )
    return market
