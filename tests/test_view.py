"""Check 2 (view and point-in-time parts): knowledge boundary, read-only state, no reference to
the market or the truth, PIT adjustment, universe, the A.eps hand case and the fixture answers."""

import gc

import numpy as np
import pytest
from conftest import GOLDEN

from dynamic_trading_engine.contracts.timeutil import parse_ts
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.data import NO_TS, Market
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import read_market
from dynamic_trading_engine.market.truth import Truth
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.state.view import (
    DecisionState,
    build_state,
    instruments_asof,
    series_asof,
    split_adjustment,
    universe_mask,
)


def _state(m, t, history=520):
    return build_state(m, t, 1e6, np.zeros(m.n), history=history)


def _reachable(root, limit=200000):
    seen, stack, out = set(), [root], []
    while stack and len(seen) < limit:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        out.append(o)
        if isinstance(o, (str, bytes, int, float, type)):
            continue
        stack.extend(gc.get_referents(o))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_state_respects_the_knowledge_boundary(seed):
    m, tr = generate(load_market_config("tiny"), seed)
    g = stream(seed, "test.instants")
    lo, hi = int(m.ts_event[0]), int(m.ts_event[-1])
    instants = [int(x) for x in g.integers(lo, hi, 25)] + [lo, hi, hi + 10**12]
    for t in instants:
        st = _state(m, t)
        k = m.bar_index(t)
        assert st.bar == k
        # nothing known after t: rows of the state end at bar k and every value is known at t
        cols = [m.ids.index(i) for i in st.ids]
        L = st.returns.shape[0]
        rows = np.arange(k - L + 1, k + 1)
        assert np.all(np.isnan(st.returns) | (m.ret_avail[rows][:, cols] <= t))
        assert np.all(np.isnan(st.close) | (m.bar_avail[rows][:, cols] <= t))
        assert np.all(np.isnan(st.signal_x) | (m.series_x.avail[0][rows][:, cols] <= t))
        assert np.all(np.isnan(st.sigma_hist) | (m.liq_avail[rows][:, cols] <= t))
        for a in ("returns", "close", "adj_close", "signal_x", "sigma_bar", "lot_size"):
            assert not getattr(st, a).flags.writeable
        objs = _reachable(st)
        assert not any(isinstance(o, (Market, Truth)) for o in objs)
    with pytest.raises(ValueError):
        _state(m, lo - 1)


def test_universe_matches_brute_force_on_random_universes():
    g = stream(9, "test.universe")
    for seed in range(1, 6):
        m, _ = generate(load_market_config("base"), seed)
        # randomize listings and delistings
        m.ts_list = m.ts_event[g.integers(0, m.n_bars // 2, m.n)].astype(np.int64)
        dl = g.integers(m.n_bars // 2, m.n_bars + 200, m.n)
        m.ts_delist = np.array(
            [int(m.ts_event[d]) if d < m.n_bars else NO_TS for d in dl], dtype=np.int64
        )
        for t in g.integers(int(m.ts_event[0]), int(m.ts_event[-1]), 30):
            t = int(t)
            want = [
                i
                for i in range(m.n)
                if m.ts_list[i] <= t and (m.ts_delist[i] == NO_TS or m.ts_delist[i] > t)
            ]
            assert list(np.nonzero(universe_mask(m, t))[0]) == want
            for row in instruments_asof(m, t):
                j = m.ids.index(row["instrument_id"])
                if m.ts_delist[j] != NO_TS and m.ts_delist[j] > t:
                    assert row["ts_delist"] is None and row["delist_return"] is None


def test_pit_adjustment_matches_brute_force_and_changes_with_time():
    m, _ = generate(load_market_config("tiny"), 4)
    splits = [r for r in m.corporate_actions if r[1] == "split"]
    assert splits, "tiny seed 4 should have a split"
    cols = np.arange(m.n)
    rows = np.arange(m.n_bars)
    for iid, _a, ts_ex, _av, _ratio in splits:
        j = m.ids.index(iid)
        ex = int(np.nonzero(m.ts_open >= ts_ex)[0][0])
        before, after = int(m.ts_event[ex - 1]), int(m.ts_event[ex])
        f_before = split_adjustment(m, before, cols, rows)
        f_after = split_adjustment(m, after, cols, rows)
        # brute force: product of the ratios of splits with ts_ex <= t whose ex-bar is later
        for t, f in ((before, f_before), (after, f_after)):
            want = np.ones(m.n_bars)
            for iid2, act, tex, _v, rr in m.corporate_actions:
                if iid2 == iid and act == "split" and tex <= t:
                    e = int(np.nonzero(m.ts_open >= tex)[0][0])
                    want[:e] *= rr
            assert np.array_equal(f[:, j], want)
        assert not np.array_equal(f_before[:, j], f_after[:, j])


def test_fixture_pit_answers():
    m = read_market(GOLDEN)
    t6 = parse_ts("2024-03-06T21:00:00Z")
    t5 = parse_ts("2024-03-05T21:00:00Z")
    t4 = parse_ts("2024-03-04T21:00:00Z")
    st6 = _state(m, t6)
    b = st6.ids.index("B")
    assert list(st6.adj_close[:, b]) == [20.2, 20.15, 20.0]
    st4 = _state(m, t4)
    assert list(st4.adj_close[:, st4.ids.index("B")]) == [40.4]
    a = st6.ids.index("A")
    want = [100.50490196078431, 101.5, 101.5]
    assert np.allclose(st6.tr_close[:, a], want, rtol=1e-12, atol=0)
    st5 = _state(m, t5)
    assert np.allclose(st5.tr_close[:, st5.ids.index("A")], [101.0, 102.0], rtol=1e-12)
    # returns of 2.5
    assert st6.returns[1, b] == pytest.approx(20.15 * 2.0 / 40.4 - 1, rel=1e-12)
    assert st6.returns[2, a] == 0.0


def test_a_eps_hand_case():
    m = read_market(GOLDEN)
    te = parse_ts("2023-12-31T21:00:00Z")
    assert series_asof(m.other_series, "A.eps", te, parse_ts("2024-03-04T21:00:00Z")) == 1.1
    assert series_asof(m.other_series, "A.eps", te, parse_ts("2024-03-05T21:00:00Z")) == 1.05
    assert series_asof(m.other_series, "A.eps", te, parse_ts("2024-02-15T20:59:59Z")) is None


def test_shortcut_of_dividend_adjustment_fails_hand_case():
    # QC App. A: the shortcut of reducing the previous close does not reproduce the 2.5 return
    prev, div, close = 102.0, 0.5, 101.7
    total = (close + div) / prev - 1
    shortcut = close / (prev - div) - 1
    assert total == pytest.approx(0.0019607843137254832, rel=1e-12)
    assert shortcut == pytest.approx(0.0019704433497538254, rel=1e-12)
    assert abs(total - shortcut) > 1e-6


def test_signal_revision_as_of():
    """A later vintage appended to a period is invisible before its ts_avail."""
    m, _ = generate(load_market_config("tiny"), 2)
    k = 150
    t = int(m.ts_event[k])
    v1 = np.full_like(m.series_x.value[0], np.nan)
    a1 = np.full_like(m.series_x.avail[0], NO_TS)
    v1[k, 0], a1[k, 0] = 99.0, int(m.ts_event[k + 3])
    m.series_x.value.append(v1)
    m.series_x.avail.append(a1)
    assert _state(m, t).signal_x[-1, 0] != 99.0
    st = _state(m, int(m.ts_event[k + 3]))
    assert st.signal_x[-4, 0] == 99.0


def test_state_is_a_frozen_dataclass():
    m, _ = generate(load_market_config("tiny"), 1)
    st = _state(m, int(m.ts_event[150]))
    assert isinstance(st, DecisionState)
    with pytest.raises(AttributeError):
        st.t = 0
