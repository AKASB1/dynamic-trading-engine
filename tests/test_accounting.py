"""Check 4 (accounting and the loop, exact) and check 3 (the contract's cost model, hand cases)."""

import math

import numpy as np
import pytest
from helpers import hand_market

from dynamic_trading_engine.contracts.costs import (
    CostConfig,
    ImpactConfig,
    accruals,
    fill_cost,
)
from dynamic_trading_engine.engine.logs import write_run_logs
from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
from dynamic_trading_engine.engine.validator import validate_run
from dynamic_trading_engine.market.config import MarketConfig
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import write_market
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.strategies.scripted import ScriptedOrders

NAN = float("nan")


def run(m, orders_by_bar, costs=None, cash=100000.0, tmp=None, lev=-1.0, gross_bound=1e9):
    costs = costs or CostConfig()
    orders = {int(m.ts_event[b]): o for b, o in orders_by_bar.items()}
    strat = ScriptedOrders(orders, gross_bound=gross_bound)
    cfg = LoopConfig(
        initial_cash=cash, decision_bars=tuple(sorted(orders_by_bar)), max_leverage=lev
    )
    res = run_loop(m, strat, costs, cfg, write_logs=True)
    if tmp is not None:
        md, rd = tmp / "market", tmp / "run"
        write_market(m, str(md))
        meta = {"costs": _cost_dict(costs), "initial_cash": cash}
        write_run_logs(res.logs, str(rd), meta, {"name": "t", "version": 1, "params": {}}, None)
        rep = validate_run(str(rd), str(md))
        assert rep["max_identity_residual"] <= 1e-9
    return res


def _cost_dict(c: CostConfig):
    return {
        "commission_bps": c.commission_bps,
        "commission_per_share": c.commission_per_share,
        "min_commission": c.min_commission,
        "half_spread_bps": c.half_spread_bps,
        "impact": {"model": c.impact.model, "y": c.impact.y},
        "borrow_bps_annual": c.borrow_bps_annual,
        "financing_bps_annual": c.financing_bps_annual,
        "cash_rate_bps_annual": c.cash_rate_bps_annual,
        "participation_cap": c.participation_cap,
    }


ZERO = CostConfig(
    commission_bps=0.0,
    half_spread_bps=0.0,
    impact=ImpactConfig("none"),
    borrow_bps_annual=0.0,
    financing_bps_annual=0.0,
)


def test_round_trip(tmp_path):
    m = hand_market(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, opens={"A": [100, 100.5, 101.5, 102.5, 103.5]}
    )
    res = run(m, {0: {"A": 10.0}, 2: {"A": -10.0}}, tmp=tmp_path)
    fl = res.logs["fills"]
    assert len(fl) == 2 and fl[0][4] == 10.0 and fl[1][4] == -10.0
    # bought at 100.5 (bar 1 open), sold at 102.5 (bar 3 open): gross 20 before costs
    gross = sum(res.flows["hold_pnl"]) + sum(res.flows["trade_pnl"])
    assert gross == pytest.approx(20.0, abs=1e-9)
    assert not res.logs["positions"][-1:] or res.logs["positions"][-1][0] < int(m.ts_event[3])


def test_short_with_borrow(tmp_path):
    m = hand_market({"A": [50.0] * 6})
    res = run(m, {0: {"A": -100.0}}, tmp=tmp_path)
    # borrow accrues from the close after the fill: |q| P * 50 bp / 1e4 * days / 365
    b = res.flows["borrow"]
    assert b[1] == 0.0
    assert b[2] == pytest.approx(100 * 50.0 * 50 / 1e4 * 1 / 365, rel=1e-12)
    assert b[5] == pytest.approx(100 * 50.0 * 50 / 1e4 * 3 / 365, rel=1e-12)  # Fri -> Mon is 3 days


def test_split_in_holding_leaves_equity_unchanged(tmp_path):
    m = hand_market(
        {"A": [100.0, 100.0, 100.0, 50.0, 50.0]},
        ca=[("A", "split", 3, 1, 2.0)],
    )
    res = run(m, {0: {"A": 10.0}}, costs=ZERO, tmp=tmp_path)
    assert res.equity[3] == pytest.approx(res.equity[2], abs=1e-9)
    pos = [r for r in res.logs["positions"] if r[0] == int(m.ts_event[3])]
    assert pos[0][2] == 20.0


def test_delisting_with_negative_return(tmp_path):
    m = hand_market({"A": [100.0, 100.0, 90.0, 80.0]}, delist={"A": (3, -0.5)})
    res = run(m, {0: {"A": 10.0}}, costs=ZERO, tmp=tmp_path)
    delist = [f for f in res.logs["fills"] if f[1] == "DELIST"]
    assert len(delist) == 1 and delist[0][4] == -10.0 and delist[0][5] == pytest.approx(40.0)
    # equity: 100000 + 10 * (40 - 100)
    assert res.equity[-1] == pytest.approx(100000.0 - 600.0, abs=1e-9)
    assert res.flows["trade_pnl"][3] == pytest.approx(-10 * (80 - 40), abs=1e-9)


def test_dividend_long_and_short(tmp_path):
    m = hand_market(
        {"A": [100.0, 100.0, 99.0, 99.0], "B": [50.0, 50.0, 49.0, 49.0]},
        ca=[("A", "cash_dividend", 2, 0, 1.0), ("B", "cash_dividend", 2, 0, 1.0)],
    )
    res = run(m, {0: {"A": 10.0, "B": -20.0}}, costs=ZERO, tmp=tmp_path)
    assert res.flows["income"][2] == pytest.approx(10 * 1.0 - 20 * 1.0)
    # the price drop and the dividend cancel: equity unchanged
    assert res.equity[2] == pytest.approx(res.equity[1], abs=1e-9)


def test_participation_cap_binds(tmp_path):
    m = hand_market({"A": [10.0] * 4}, volumes={"A": [1000.0, 500.0, 1000.0, 1000.0]})
    res = run(m, {0: {"A": 120.0}}, tmp=tmp_path)
    fills = res.logs["fills"]
    assert len(fills) == 1 and fills[0][4] == 50.0  # 0.1 * 500, the rest cancelled
    assert res.n_capped == 1


def test_zero_volume_bar_does_not_fill(tmp_path):
    m = hand_market({"A": [10.0] * 4}, volumes={"A": [1000.0, 0.0, 1000.0, 1000.0]})
    res = run(m, {0: {"A": 10.0}}, tmp=tmp_path)
    assert res.logs["fills"] == []


def test_lot_rounding(tmp_path):
    m = hand_market({"A": [10.0] * 4}, lot=100.0)
    from dynamic_trading_engine.strategies.scripted import ScriptedWeights

    strat = ScriptedWeights(lambda s: np.array([0.5]), gross_bound=1.0)
    cfg = LoopConfig(initial_cash=1999.0 * 10, decision_bars=(0,))
    res = run_loop(m, strat, ZERO, cfg, write_logs=True)
    # target 0.5 * 19990 / 10 = 999.5 shares -> 900 (toward zero to the lot)
    assert res.logs["orders"][0][3] == 900.0


def test_leverage_scaling_is_logged(tmp_path):
    # order 1.5x equity in A: gross above max_leverage 1.05 is scaled down to it
    m = hand_market({"A": [100.0] * 4})
    res = run(m, {0: {"A": 1500.0}}, costs=ZERO, lev=1.05, tmp=tmp_path)
    assert len(res.leverage_events) == 1
    k, s, g_before, g_after = res.leverage_events[0]
    assert g_before == pytest.approx(1.5) and g_after == pytest.approx(1.05, abs=1e-9)
    assert res.logs["fills"][0][4] == pytest.approx(1050.0, abs=1e-6)


def test_attribution_sums_to_totals(tmp_path):
    m, _ = generate(
        MarketConfig(
            name="t",
            n_instruments=4,
            n_bars=80,
            warmup_bars=30,
            delist_hazard_annual=2.0,
            split_rate_annual=3.0,
        ),
        3,
    )
    g = stream(1, "test.attr")
    orders = {b: {i: float(g.integers(-50, 50)) for i in m.ids} for b in range(30, 78, 4)}
    orders = {b: {i: q for i, q in o.items() if q} for b, o in orders.items()}
    res = run(m, orders, cash=1e6, tmp=tmp_path)
    a = res.attribution["by_instrument"]
    inst = (
        a["hold"]
        + a["trade"]
        - a["spread"]
        - a["impact"]
        - a["commission"]
        - a["borrow"]
        + a["dividend"]
    )
    f = res.flows
    total = sum(
        f["hold_pnl"]
        + f["trade_pnl"]
        - f["spread_cost"]
        - f["impact_cost"]
        - f["commission"]
        - f["borrow"]
        - f["financing"]
        + f["income"]
    )
    book = res.attribution["by_book"]
    assert math.fsum(inst) == pytest.approx(total + sum(f["financing"]), abs=1e-6)
    assert book["long"] + book["short"] + book["cash"] == pytest.approx(total, abs=1e-6)
    assert res.equity[-1] - res.equity[0] == pytest.approx(total, abs=1e-6)


@pytest.mark.parametrize("block", range(10))
def test_random_scenarios_fill_rules(tmp_path, block):
    """100 random scenarios (10 blocks of 10): random scripted orders on random small markets
    with events; the validator checks fill bar = decision bar + 1, nothing earlier, the cap, the
    costs, and the identity."""
    for s in range(10):
        seed = 1 + block * 10 + s
        g = stream(seed, "test.scenario")
        cfg = MarketConfig(
            name="t",
            n_instruments=int(g.integers(2, 5)),
            n_bars=int(g.integers(40, 70)),
            warmup_bars=25,
            delist_hazard_annual=3.0,
            split_rate_annual=4.0,
            late_listing_share=0.3,
            adv_lo=2000.0,
            adv_hi=20000.0,
        )
        m, _ = generate(cfg, seed)
        orders = {}
        for b in range(25, cfg.n_bars - 1, int(g.integers(1, 4))):
            orders[b] = {i: float(g.integers(-400, 400)) for i in m.ids if g.random() < 0.6}
        res = run(m, orders, cash=1e6, tmp=tmp_path / f"s{seed}")
        for _t_dec, subs in res.orders_by_decision:
            assert all(q != 0 for _, q in subs)


# ---------------------------------------------------------------- check 3: cost model cases


def test_cost_hand_cases():
    c = CostConfig()
    # min_commission binds
    fc = fill_cost(1.0, 10.0, 0.02, 1e6, CostConfig(min_commission=1.0))
    assert fc.commission == 1.0
    # commission per share
    fc = fill_cost(-200.0, 50.0, 0.02, 1e6, CostConfig(commission_per_share=0.005))
    assert fc.commission == pytest.approx(200 * 50 * 1 / 1e4 + 200 * 0.005)
    assert fc.price < 50.0  # a sell receives less than the reference
    # linear impact
    fc = fill_cost(1000.0, 20.0, 0.03, 1e5, CostConfig(impact=ImpactConfig("linear", 0.5)))
    assert fc.impact_bps == pytest.approx(1e4 * 0.5 * 0.03 * 1000 / 1e5)
    # unknown liquidity: zero impact, counted
    fc = fill_cost(1000.0, 20.0, None, None, c)
    assert fc.impact_cost == 0.0 and fc.impact_unavailable
    # short borrow accrual
    b, f, i = accruals(np.array([-100.0]), np.array([20.0]), -500.0, 3.0, c)
    assert b == pytest.approx(100 * 20 * 50 / 1e4 * 3 / 365)
    assert f == pytest.approx(500 * 100 / 1e4 * 3 / 365)
    assert i == 0.0


def test_cost_monotonicity():
    c = CostConfig()
    qs = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    tot = [sum(_tc(fill_cost(q, 30.0, 0.02, 1e5, c))) for q in qs]
    assert all(a <= b for a, b in zip(tot, tot[1:]))
    sig = [sum(_tc(fill_cost(500.0, 30.0, s, 1e5, c))) for s in (0.0, 0.01, 0.02, 0.05)]
    assert all(a <= b for a, b in zip(sig, sig[1:]))
    vol = [sum(_tc(fill_cost(500.0, 30.0, 0.02, V, c))) for V in (1e3, 1e4, 1e5, 1e6)]
    assert all(a >= b for a, b in zip(vol, vol[1:]))


def _tc(fc):
    return fc.spread_cost, fc.impact_cost, fc.commission


def test_impact_ignores_fill_bar_volume_unless_cap_binds():
    a = hand_market({"A": [10.0] * 4}, volumes={"A": [1e6, 1e6, 1e6, 1e6]})
    b = hand_market({"A": [10.0] * 4}, volumes={"A": [1e6, 2e5, 1e6, 1e6]})
    ra = run(a, {0: {"A": 1000.0}})
    rb = run(b, {0: {"A": 1000.0}})
    assert ra.logs["fills"][0][8] == rb.logs["fills"][0][8]


def test_ex_split_fill_bar_rescales_the_decision_time_adv(tmp_path):
    """An order decided before a split and filled in the ex-split bar: the quantity and the
    decision-time ADV are both in post-split shares, so impact is the contract's formula on the
    rescaled pair (and the validator agrees)."""
    m = hand_market({"A": [100.0, 50.0, 50.0]}, ca=[("A", "split", 1, 0, 2.0)], adv=1e5)
    res = run(m, {0: {"A": 1000.0}}, tmp=tmp_path)
    f = res.logs["fills"][0]
    assert f[4] == 2000.0
    want = 2000.0 * 50.0 * 0.5 * 0.02 * math.sqrt(2000.0 / 2e5)
    assert f[8] == pytest.approx(want, rel=1e-12)


def test_retuned_rows_need_their_marker():
    from dynamic_trading_engine.experiments.tables import TableError, check_rows

    with pytest.raises(TableError):
        check_rows([{"strategy": "MV_RETUNED", "label": ""}])
    check_rows([{"strategy": "MV_RETUNED", "label": "RETUNED"}])
