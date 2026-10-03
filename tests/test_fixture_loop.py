"""Check 1 (loop part): the explicit order of QC Appendix A, run through the loop on the fixture
market, reproduces the fill, positions, and equity files of the appendix to 1e-9."""

import numpy as np
from conftest import GOLDEN

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.costs import CostConfig, fill_cost
from dynamic_trading_engine.contracts.timeutil import parse_ts
from dynamic_trading_engine.engine.logs import write_run_logs
from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
from dynamic_trading_engine.engine.validator import validate_run
from dynamic_trading_engine.market.store import read_market
from dynamic_trading_engine.strategies.scripted import ScriptedOrders


def _golden(name):
    with open(f"{GOLDEN}/{name}.csv", "rb") as fh:
        return S.load_bytes(S.SCHEMAS[name], fh.read()).rows


def _run():
    m = read_market(GOLDEN)
    t = parse_ts("2024-03-04T21:00:00Z")
    strat = ScriptedOrders({t: {"A": 100.0}}, strategy_id="fixture")
    cfg = LoopConfig(initial_cash=100000.0, decision_bars=(0,))
    return m, run_loop(m, strat, CostConfig(), cfg, write_logs=True)


def _same(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if isinstance(y, float):
                assert abs(x - y) <= 1e-9 * max(1.0, abs(y)), (ra, rb)
            else:
                assert x == y, (ra, rb)


def test_fixture_order_reproduces_appendix():
    m, res = _run()
    logs = res.logs
    want_fills = _golden("fills_v1")
    got_fills = logs["fills"]
    # ids are formatted by the engine; everything else must match
    _same([r[2:] for r in got_fills], [r[2:] for r in want_fills])
    _same(logs["positions"], _golden("positions_v1"))
    _same(logs["equity"], _golden("equity_v1"))
    o = logs["orders"][0]
    w = _golden("orders_v1")[0]
    assert o[1:] == w[1:]


def test_cost_known_answer():
    fc = fill_cost(100.0, 101.5, 0.02, 1_000_000.0, CostConfig())
    assert abs(fc.impact_bps - 1.0) < 1e-12
    assert abs(fc.impact_cost - 1.015) < 1e-12
    assert abs(fc.spread_cost - 2.03) < 1e-12
    assert abs(fc.commission - 1.015) < 1e-12
    assert abs(fc.price - 101.53045) < 1e-9


def test_fixture_run_passes_validator(tmp_path):
    m, res = _run()
    write_run_logs(
        res.logs,
        str(tmp_path),
        {"costs": {}, "initial_cash": 100000.0},
        {"name": "fixture", "version": 1, "params": {}},
        None,
    )
    rep = validate_run(str(tmp_path), GOLDEN)
    assert rep["bars"] == 3 and rep["fills"] == 1
    assert np.isclose(res.equity[-1], 100045.94)
