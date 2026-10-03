"""The rolling loop: decisions on a calendar, next-open fills with the contract's costs,
corporate actions, delistings, accruals, marks, and the exact accounting identity (QC 3, 4, 5).

Per bar, in the order of QC 3.7: splits (which rescale the opening quantity, the previous
close, and pending orders) and dividends, the fills of the orders submitted at the previous
decision, the forced exit of an instrument in its last bar, accruals, the mark to the close;
then, on a decision bar, the pipeline (decision state, strategy, orders). The identity of QC 5
is asserted on every bar. The loop never reads the wall clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dynamic_trading_engine.contracts.costs import CostConfig, accruals, fill_cost
from dynamic_trading_engine.contracts.timeutil import days_between
from dynamic_trading_engine.engine.types import Decision, Strategy
from dynamic_trading_engine.market.data import NO_TS, Market
from dynamic_trading_engine.state.view import build_state

IDENTITY_TOL = 1e-9
LEVERAGE_EPS = 1e-6


class IdentityError(AssertionError):
    pass


@dataclass(frozen=True)
class LoopConfig:
    initial_cash: float = 10_000_000.0
    rebalance_every: int = 5
    warmup_bars: int = 260
    max_leverage: float = -1.0  # -1: 1.05 x the strategy's gross bound
    history: int = 520
    decision_bars: tuple[int, ...] = ()  # explicit schedule (tests); empty: the regular one

    def schedule(self, n_bars: int) -> list[int]:
        if self.decision_bars:
            return sorted(b for b in self.decision_bars if 0 <= b < n_bars - 1)
        return list(range(self.warmup_bars, n_bars - 1, self.rebalance_every))


@dataclass
class RunResult:
    ts_event: np.ndarray
    equity: np.ndarray
    cash: np.ndarray
    position_value: np.ndarray
    flows: dict[str, np.ndarray]
    gross_exposure: np.ndarray
    net_exposure: np.ndarray
    turnover: np.ndarray
    decision_bars: list[int]
    n_decisions: int = 0
    n_holds: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    leverage_events: list[tuple] = field(default_factory=list)
    impact_unavailable: int = 0
    n_fills: int = 0
    n_capped: int = 0
    max_identity_residual: float = 0.0
    max_violation: float = 0.0
    orders_by_decision: list[tuple] = field(default_factory=list)
    attribution: dict = field(default_factory=dict)
    logs: dict | None = None
    wall_solve_ms: list[float] = field(default_factory=list)


FLOW_NAMES = (
    "hold_pnl",
    "trade_pnl",
    "spread_cost",
    "impact_cost",
    "commission",
    "borrow",
    "financing",
    "income",
)


def _round_lot(x: np.ndarray, lot: np.ndarray) -> np.ndarray:
    """Round toward zero to a multiple of ``lot`` (no rounding where lot is 0)."""
    out = np.array(x, dtype=float)
    pos = lot > 0
    out[pos] = np.trunc(out[pos] / lot[pos]) * lot[pos]
    return out


def _fmt_id(prefix: str, k: int) -> str:
    return f"{prefix}{k:07d}"


def run_loop(
    market: Market,
    strategy: Strategy,
    costs: CostConfig,
    cfg: LoopConfig,
    write_logs: bool = False,
    stop_after_bar: int | None = None,
) -> RunResult:
    T, n = market.n_bars, market.n
    ids = market.ids
    lot = market.lot_size
    sched = cfg.schedule(T)
    sched_set = set(sched)
    max_lev = cfg.max_leverage
    if max_lev < 0:
        max_lev = 1.05 * float(getattr(strategy, "gross_bound", 1.0))

    q = np.zeros(n)
    cash = float(cfg.initial_cash)
    mark = np.full(n, np.nan)
    prev_target = np.zeros(n)
    pending: list[dict] = []
    flows = {k: np.zeros(T) for k in FLOW_NAMES}
    equity = np.zeros(T)
    cash_arr = np.zeros(T)
    posval = np.zeros(T)
    gross = np.zeros(T)
    net = np.zeros(T)
    turnover = np.zeros(T)
    res = RunResult(
        ts_event=market.ts_event.copy(),
        equity=equity,
        cash=cash_arr,
        position_value=posval,
        flows=flows,
        gross_exposure=gross,
        net_exposure=net,
        turnover=turnover,
        decision_bars=sched,
        status_counts={"optimal": 0, "optimal_inaccurate": 0, "failed": 0, "none": 0},
    )
    attr = {
        k: np.zeros(n)
        for k in ("hold", "trade", "spread", "impact", "commission", "borrow", "dividend")
    }
    book = {"long": 0.0, "short": 0.0, "cash": 0.0}
    logs = (
        {"orders": [], "fills": [], "positions": [], "equity": [], "decisions": []}
        if write_logs
        else None
    )
    order_no = 0
    fill_no = 0
    has_all = market.bar_avail != NO_TS

    for k in range(T):
        has = has_all[k]
        t_close = int(market.ts_event[k])
        if k == 0:
            mark = np.where(has, market.close[0], mark)
            equity[0] = cash
            cash_arr[0] = cash
            if logs is not None:
                logs["equity"].append((t_close, cash, 0.0, cash) + (0.0,) * 8)
        else:
            q_open = q.copy()
            p_prev = mark.copy()
            cash_prev = cash
            e_prev = equity[k - 1]
            ratio = np.where(has, market.split_ratio[k], 1.0)
            div = np.where(has, market.dividend[k], 0.0)
            # (1) splits
            q_adj = q_open * ratio
            p_prev_adj = p_prev / ratio
            for o in pending:
                o["qty"] *= ratio[o["j"]]
                if o["adv"] is not None:
                    o["adv"] *= ratio[o["j"]]  # decision-time ADV in post-split shares
            # (2) dividends on the opening quantity before the split, credited at the close
            div_inst = q_open * div
            income_div = float(np.sum(div_inst))
            # (3) fills of the orders submitted at the previous decision
            m_ref = np.where(has, market.open[k], p_prev_adj)
            m_ref = np.where(np.isfinite(m_ref), m_ref, 0.0)
            fill_q = np.zeros(n)
            cost_s = np.zeros(n)
            cost_i = np.zeros(n)
            cost_c = np.zeros(n)
            spent = 0.0
            fill_rows = []
            if pending:
                want = np.zeros(n)
                meta = {}
                for o in pending:
                    j = o["j"]
                    if not has[j] or not (market.volume[k, j] > 0):
                        continue  # no bar or no trading: a day order is cancelled
                    cap = costs.participation_cap * market.volume[k, j]
                    qty = o["qty"]
                    if abs(qty) > cap:
                        qty = math.copysign(cap, qty)
                        res.n_capped += 1
                    want[j] = qty
                    meta[j] = o
                if meta:
                    # leverage rule at the fill's reference prices
                    e_ref = cash_prev + float(np.sum(q_adj * np.where(q_adj != 0, m_ref, 0.0)))
                    if e_ref > 0:

                        def gross_at(s, want=want, q_adj=q_adj, m_ref=m_ref, e_ref=e_ref):
                            return float(np.sum(np.abs(q_adj + s * want) * m_ref)) / e_ref

                        g1 = gross_at(1.0)
                        if g1 - max_lev > LEVERAGE_EPS:
                            g0 = gross_at(0.0)
                            target = max(max_lev, g0)
                            lo_s, hi_s = 0.0, 1.0
                            for _ in range(60):
                                mid = 0.5 * (lo_s + hi_s)
                                if gross_at(mid) <= target:
                                    lo_s = mid
                                else:
                                    hi_s = mid
                            want = want * lo_s
                            res.leverage_events.append((k, lo_s, g1, gross_at(lo_s)))
                    want = _round_lot(want, lot)
                    for j in sorted(meta):
                        qty = float(want[j])
                        if qty == 0.0:
                            continue
                        o = meta[j]
                        m = float(m_ref[j])
                        fc = fill_cost(qty, m, o["sigma"], o["adv"], costs)
                        if fc.impact_unavailable:
                            res.impact_unavailable += 1
                        fill_q[j] = qty
                        cost_s[j], cost_i[j], cost_c[j] = (
                            fc.spread_cost,
                            fc.impact_cost,
                            fc.commission,
                        )
                        spent += qty * fc.price + fc.commission
                        fill_no += 1
                        res.n_fills += 1
                        if logs is not None:
                            fill_rows.append(
                                (
                                    _fmt_id("f", fill_no),
                                    o["order_id"],
                                    int(market.ts_open[k]),
                                    ids[j],
                                    qty,
                                    m,
                                    fc.price,
                                    fc.spread_cost,
                                    fc.impact_cost,
                                    fc.commission,
                                )
                            )
            pending = []
            q = q_adj + fill_q
            close_k = np.where(has, market.close[k], p_prev_adj)
            trade_inst = np.where(fill_q != 0, fill_q * (close_k - m_ref), 0.0)
            # (4) forced exit in the last bar of a delisting instrument
            delist_now = has & (market.ts_delist == t_close) & (q != 0)
            for j in np.nonzero(delist_now)[0]:
                ref = float(market.close[k, j] * (1.0 + market.delist_return[j]))
                qf = -float(q[j])
                spent += qf * ref
                trade_inst[j] += qf * (market.close[k, j] - ref)
                turnover[k] += abs(qf) * ref
                q[j] = 0.0
                fill_no += 1
                if logs is not None:
                    fill_rows.append(
                        (
                            _fmt_id("f", fill_no),
                            "DELIST",
                            t_close,
                            ids[j],
                            qf,
                            ref,
                            ref,
                            0.0,
                            0.0,
                            0.0,
                        )
                    )
            # (5) accruals on the previous close's state
            dt = days_between(int(market.ts_event[k - 1]), t_close)
            borrow, financing, interest = accruals(q_open, p_prev, cash_prev, dt, costs)
            short = q_open < 0
            borrow_inst = np.where(
                short, np.abs(q_open) * p_prev * costs.borrow_bps_annual / 1e4 * dt / 365.0, 0.0
            )
            # (6) mark to the close
            mark = close_k
            held = q_open != 0
            hold_inst = np.where(held, q_adj * (mark - p_prev_adj), 0.0)
            cash = cash_prev - spent + income_div + interest - borrow - financing
            pv = float(np.sum(np.where(q != 0, q * mark, 0.0)))
            e_now = cash + pv
            f = flows
            f["hold_pnl"][k] = float(np.sum(hold_inst))
            f["trade_pnl"][k] = float(np.sum(trade_inst))
            f["spread_cost"][k] = float(np.sum(cost_s))
            f["impact_cost"][k] = float(np.sum(cost_i))
            f["commission"][k] = float(np.sum(cost_c))
            f["borrow"][k] = borrow
            f["financing"][k] = financing
            f["income"][k] = income_div + interest
            change = (
                f["hold_pnl"][k]
                + f["trade_pnl"][k]
                - f["spread_cost"][k]
                - f["impact_cost"][k]
                - f["commission"][k]
                - borrow
                - financing
                + f["income"][k]
            )
            resid = abs((e_now - e_prev) - change) / max(1.0, abs(e_prev))
            res.max_identity_residual = max(res.max_identity_residual, resid)
            if resid > IDENTITY_TOL:
                raise IdentityError(
                    f"accounting identity violated at bar {k}: residual {resid:.3e}"
                )
            equity[k] = e_now
            cash_arr[k] = cash
            posval[k] = pv
            turnover[k] = (turnover[k] + float(np.sum(np.abs(fill_q) * m_ref))) / e_prev
            gross[k] = float(np.sum(np.abs(q * mark)[q != 0])) / e_now if e_now else 0.0
            net[k] = float(np.sum((q * mark)[q != 0])) / e_now if e_now else 0.0
            # attribution
            attr["hold"] += hold_inst
            attr["trade"] += trade_inst
            attr["spread"] += cost_s
            attr["impact"] += cost_i
            attr["commission"] += cost_c
            attr["borrow"] += borrow_inst
            attr["dividend"] += div_inst
            inst_total = hold_inst + trade_inst - cost_s - cost_i - cost_c - borrow_inst + div_inst
            side = np.where(
                q_open != 0, np.sign(q_open), np.where(q != 0, np.sign(q), np.sign(fill_q))
            )
            book["long"] += float(np.sum(inst_total[side > 0]))
            book["short"] += float(np.sum(inst_total[side < 0]))
            book["cash"] += interest - financing + float(np.sum(inst_total[side == 0]))
            if logs is not None:
                logs["fills"].extend(fill_rows)
                for j in np.nonzero(q)[0]:
                    logs["positions"].append(
                        (t_close, ids[j], float(q[j]), float(mark[j]), float(q[j] * mark[j]))
                    )
                logs["equity"].append(
                    (t_close, cash, pv, e_now) + tuple(float(f[nm][k]) for nm in FLOW_NAMES)
                )
        # ---- decision
        if k in sched_set:
            state = build_state(market, t_close, cash, q, prev_target, cfg.history)
            dec: Decision = strategy.decide(state)
            res.n_decisions += 1
            res.status_counts[dec.status] = res.status_counts.get(dec.status, 0) + 1
            res.max_violation = max(res.max_violation, dec.max_violation)
            res.wall_solve_ms.append(dec.wall_solve_ms)
            target_q = q.copy()
            col = {iid: j for j, iid in enumerate(ids)}
            if dec.orders is not None:
                for iid, qty in sorted(dec.orders.items()):
                    target_q[col[iid]] = q[col[iid]] + qty
            elif dec.hold or dec.weights is None:
                res.n_holds += 1 if dec.hold else 0
            else:
                w = np.zeros(n)
                for c, iid in enumerate(state.ids):
                    w[col[iid]] = dec.weights[c]
                e_k = state.portfolio.equity
                known = has & (market.bar_avail[k] <= t_close)
                price = np.where(known, market.close[k], np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tq = np.where(np.isfinite(price) & (w != 0), w * e_k / price, 0.0)
                target_q = np.where(np.isfinite(price), _round_lot(tq, lot), q)
                prev_target = w
            orders = target_q - q
            sig_of = dict(zip(state.ids, state.sigma_bar))
            adv_of = dict(zip(state.ids, state.adv))
            submitted = []
            traded = 0.0
            for j in np.nonzero(orders)[0]:
                qty = float(orders[j])
                order_no += 1
                oid = _fmt_id("o", order_no)
                sg = sig_of.get(ids[j], np.nan)
                av = adv_of.get(ids[j], np.nan)
                pending.append(
                    {
                        "j": int(j),
                        "qty": qty,
                        "order_id": oid,
                        "sigma": None if not np.isfinite(sg) else float(sg),
                        "adv": None if not np.isfinite(av) else float(av),
                    }
                )
                submitted.append((ids[j], qty))
                traded += abs(qty) * float(market.close[k, j]) if has[j] else 0.0
                if logs is not None:
                    logs["orders"].append(
                        (oid, t_close, ids[j], qty, "market", None, "day", strategy.strategy_id)
                    )
            res.orders_by_decision.append((t_close, tuple(submitted)))
            if logs is not None:
                logs["decisions"].append(
                    (
                        t_close,
                        strategy.strategy_id,
                        strategy.forecast_id,
                        bool(strategy.oracle),
                        strategy.risk_id,
                        dec.status,
                        dec.objective,
                        len(submitted),
                        traded / equity[k] if equity[k] else 0.0,
                        dec.wall_solve_ms,
                        bool(dec.hold),
                    )
                )
        if stop_after_bar is not None and k >= stop_after_bar:
            res.equity = equity[: k + 1]
            break
    res.attribution = {"by_instrument": {k: v.copy() for k, v in attr.items()}, "by_book": book}
    res.logs = logs
    return res
