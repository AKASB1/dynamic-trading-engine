"""Independent validator of a run's logs (check 4).

It reads only contract files: the run's orders, fills, positions, and equity logs, the run's
``run.json`` (cost configuration and initial cash), and the market's instruments, bars,
corporate actions, and liquidity. From those it recomputes, bar by bar and without any code of
the loop, the positions, ``cash``, ``hold_pnl``, ``trade_pnl``, every cost and accrual, and the
identity of QC 5, and checks the fill rules (the fill bar is the decision bar plus one, the
reference price is that bar's open, the participation cap holds, the costs follow QC 4).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import read_dataset_bytes, read_json

TOL = 1e-9


class ValidationError(AssertionError):
    pass


def _load(directory: str, name: str, schema, check: bool = True) -> list[dict]:
    path = os.path.join(directory, name + ".csv")
    check = check and os.path.exists(path[:-4] + ".manifest.json")
    return S.load_bytes(schema, read_dataset_bytes(path, check_manifest=check)).dicts()


def _close(a: float, b: float, scale: float, what: str, where) -> None:
    if abs(a - b) > TOL * max(1.0, abs(scale)):
        raise ValidationError(f"{what} at {where}: log {a!r} != recomputed {b!r}")


def validate_run(run_dir: str, market_dir: str, liquidity_dir: str | None = None) -> dict:
    meta = read_json(os.path.join(run_dir, "run.json"))
    costs = meta["costs"]
    hs = costs.get("half_spread_bps", 2.0)
    cb = costs.get("commission_bps", 1.0)
    cps = costs.get("commission_per_share", 0.0)
    minc = costs.get("min_commission", 0.0)
    imp = costs.get("impact", {})
    model, y = imp.get("model", "sqrt"), imp.get("y", 0.5)
    borrow_bps = costs.get("borrow_bps_annual", 50.0)
    fin_bps = costs.get("financing_bps_annual", 100.0)
    cash_bps = costs.get("cash_rate_bps_annual", 0.0)
    cap = costs.get("participation_cap", 0.1)

    inst = {r["instrument_id"]: r for r in _load(market_dir, "instruments_v1", S.INSTRUMENTS)}
    bars = defaultdict(dict)  # iid -> ts_event -> row
    bar_by_open = {}
    for r in _load(market_dir, "bars_v1", S.BARS):
        bars[r["instrument_id"]][r["ts_event"]] = r
        bar_by_open[(r["instrument_id"], r["ts_open"])] = r
    first_open_after = {}
    for iid, by in bars.items():
        first_open_after[iid] = sorted((r["ts_open"], te) for te, r in by.items())
    splits, divs = {}, {}
    for r in _load(market_dir, "corporate_actions_v1", S.CORPORATE_ACTIONS):
        ex = next(
            (te for to, te in first_open_after.get(r["instrument_id"], []) if to >= r["ts_ex"]),
            None,
        )
        if ex is None:
            continue
        (splits if r["action"] == "split" else divs)[(r["instrument_id"], ex)] = r["value"]
    liq = defaultdict(list)
    for r in _load(liquidity_dir or market_dir, "liquidity_v1", S.LIQUIDITY):
        liq[r["instrument_id"]].append(r)

    def liquidity_at(iid, t):
        best = None
        for r in liq[iid]:
            if r["ts_avail"] <= t and (best is None or r["ts_event"] >= best["ts_event"]):
                best = r
        return best

    orders = {r["order_id"]: r for r in _load(run_dir, "orders_v1", S.ORDERS)}
    fills = _load(run_dir, "fills_v1", S.FILLS)
    positions = defaultdict(dict)
    for r in _load(run_dir, "positions_v1", S.POSITIONS):
        positions[r["ts_event"]][r["instrument_id"]] = r["quantity"]
    eq = _load(run_dir, "equity_v1", S.EQUITY)
    if not eq:
        raise ValidationError("empty equity log")
    calendar = [r["ts_event"] for r in eq]
    fills_at = defaultdict(list)
    cal_open = {}
    for te in calendar:
        for by in bars.values():
            if te in by:
                cal_open[te] = by[te]["ts_open"]
                break
    open_to_event = {v: k for k, v in cal_open.items()}
    for f in fills:
        te = f["ts_fill"] if f["order_id"] == "DELIST" else open_to_event.get(f["ts_fill"])
        if te is None:
            raise ValidationError(f"fill {f['fill_id']} at an instant that is no bar open")
        fills_at[te].append(f)

    r0 = eq[0]
    init = meta["initial_cash"]
    _close(r0["cash"], init, init, "opening cash", calendar[0])
    for k in S.EQUITY_FLOWS:
        if r0[k] != 0.0:
            raise ValidationError("opening row has nonzero flows")
    if fills_at.get(calendar[0]):
        raise ValidationError("a fill in the first bar")
    q: dict[str, float] = {}
    mark: dict[str, float] = {}
    for iid, by in bars.items():
        if calendar[0] in by:
            mark[iid] = by[calendar[0]]["close"]
    cash = init
    max_res = 0.0
    cum_change = 0.0
    n_checked = 0
    for k in range(1, len(eq)):
        te, prev_te = calendar[k], calendar[k - 1]
        row = eq[k]
        q_open = dict(q)
        p_prev = dict(mark)
        q_adj, p_adj = dict(q_open), dict(p_prev)
        dividends = 0.0
        for iid in sorted(set(q_open) | set(mark)):
            if te in bars.get(iid, {}):
                ratio = splits.get((iid, te), 1.0)
                if iid in q_adj:
                    q_adj[iid] = q_open[iid] * ratio
                if iid in p_adj:
                    p_adj[iid] = p_prev[iid] / ratio
                dividends += q_open.get(iid, 0.0) * divs.get((iid, te), 0.0)
        qn = dict(q_adj)
        trade = spread = impact = comm = spent = 0.0
        for f in fills_at.get(te, []):
            iid = f["instrument_id"]
            bar = bars[iid][te]
            qf = f["quantity"]
            if f["order_id"] == "DELIST":
                ins = inst[iid]
                if ins["ts_delist"] != te:
                    raise ValidationError(f"DELIST fill of {iid} outside its last bar")
                ref = bar["close"] * (1.0 + ins["delist_return"])
                _close(f["ref_price"], ref, ref, "delist reference price", (te, iid))
                if abs(qn.get(iid, 0.0) + qf) > TOL * max(1.0, abs(qf)):
                    raise ValidationError(f"DELIST fill does not close the position of {iid}")
                trade += qf * (bar["close"] - ref)
                spent += qf * ref
                qn[iid] = 0.0
                continue
            o = orders.get(f["order_id"])
            if o is None or o["instrument_id"] != iid:
                raise ValidationError(f"fill {f['fill_id']} has no matching order")
            if o["ts_submit"] != prev_te:
                raise ValidationError(f"fill {f['fill_id']} is not in the bar after its decision")
            if f["ts_fill"] != bar["ts_open"] or f["ref_price"] != bar["open"]:
                raise ValidationError(f"fill {f['fill_id']} is not at the next open")
            ratio = splits.get((iid, te), 1.0)
            if qf * o["quantity"] <= 0 or abs(qf) > abs(o["quantity"] * ratio) * (1 + 1e-12):
                raise ValidationError(f"fill {f['fill_id']} exceeds its order")
            if abs(qf) > cap * bar["volume"] * (1 + 1e-12) or bar["volume"] <= 0:
                raise ValidationError(f"fill {f['fill_id']} exceeds the participation cap")
            m = bar["open"]
            lr = liquidity_at(iid, o["ts_submit"])
            a = abs(qf)
            # the decision-time ADV in shares of the fill bar (an ex-split fill bar rescales it)
            adv = lr["adv_shares"] * ratio if lr is not None else None
            if model == "none" or lr is None:
                ib = 0.0
            elif model == "sqrt":
                ib = 1e4 * y * lr["sigma_bar"] * math.sqrt(a / adv)
            else:
                ib = 1e4 * y * lr["sigma_bar"] * a / adv
            sc, ic = a * m * hs / 1e4, a * m * ib / 1e4
            cc = max(minc, a * m * cb / 1e4 + a * cps)
            price = m * (1 + math.copysign(1.0, qf) * (hs + ib) / 1e4)
            for what, got, want in (
                ("spread_cost", f["spread_cost"], sc),
                ("impact_cost", f["impact_cost"], ic),
                ("commission", f["commission"], cc),
                ("price", f["price"], price),
            ):
                _close(got, want, want, what, f["fill_id"])
            spread += sc
            impact += ic
            comm += cc
            spent += qf * price + cc
            trade += qf * (bar["close"] - m)
            qn[iid] = qn.get(iid, 0.0) + qf
        dt = (te - prev_te) / 86_400_000_000
        borrow = (
            sum(abs(v) * p_prev[i] for i, v in q_open.items() if v < 0)
            * borrow_bps
            / 1e4
            * dt
            / 365
        )
        financing = max(0.0, -cash) * fin_bps / 1e4 * dt / 365
        interest = max(0.0, cash) * cash_bps / 1e4 * dt / 365
        hold = 0.0
        new_mark = dict(p_adj)
        for iid, by in bars.items():
            if te in by:
                new_mark[iid] = by[te]["close"]
        for iid, v in q_open.items():
            if v != 0:
                hold += q_adj[iid] * (new_mark[iid] - p_adj[iid])
        cash = cash - spent + dividends + interest - borrow - financing
        q = {i: v for i, v in qn.items() if v != 0}
        mark = new_mark
        pv = sum(v * mark[i] for i, v in q.items())
        e_prev = eq[k - 1]["equity"]
        scale = max(abs(e_prev), abs(row["equity"]))
        for what, got, want in (
            ("hold_pnl", row["hold_pnl"], hold),
            ("trade_pnl", row["trade_pnl"], trade),
            ("spread_cost", row["spread_cost"], spread),
            ("impact_cost", row["impact_cost"], impact),
            ("commission", row["commission"], comm),
            ("borrow", row["borrow"], borrow),
            ("financing", row["financing"], financing),
            ("income", row["income"], dividends + interest),
            ("cash", row["cash"], cash),
            ("position_value", row["position_value"], pv),
            ("equity", row["equity"], cash + pv),
        ):
            _close(got, want, scale, what, te)
        change = S.equity_change(row)
        res = abs((row["equity"] - e_prev) - change) / max(1.0, abs(e_prev))
        max_res = max(max_res, res)
        if res > TOL:
            raise ValidationError(f"identity violated at {te}: {res:.3e}")
        cum_change += change
        logged = positions.get(te, {})
        if set(logged) != set(q) or any(
            abs(logged[i] - q[i]) > TOL * max(1.0, abs(q[i])) for i in q
        ):
            raise ValidationError(f"positions at {te} do not reconcile with the fills")
        n_checked += 1
    cum_res = abs(eq[-1]["equity"] - eq[0]["equity"] - cum_change) / max(1.0, abs(eq[-1]["equity"]))
    if cum_res > TOL:
        raise ValidationError(f"cumulative identity violated: {cum_res:.3e}")
    return {
        "bars": n_checked + 1,
        "fills": len(fills),
        "max_identity_residual": max_res,
        "cumulative_residual": cum_res,
    }
