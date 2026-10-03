"""One run of the rolling loop as a picklable job, executed in a worker process.

The worker generates the market of (configuration, seed), builds the strategy with the factory
(oracle providers get the truth), runs the loop with an injected wall clock, asserts the
accounting identity (inside the loop), and returns one summary row. A job that asks for logs
also writes the market's contract files and the run's logs to ``experiments/outputs/`` and has
them checked by the independent validator.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field

import numpy as np

from dynamic_trading_engine.analytics.metrics import segment, summarize
from dynamic_trading_engine.contracts.costs import CostConfig, cost_config_from_dict
from dynamic_trading_engine.engine.logs import write_run_logs
from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
from dynamic_trading_engine.engine.validator import validate_run
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import write_market
from dynamic_trading_engine.strategies.factory import build_strategy


@dataclass(frozen=True)
class Job:
    experiment: str
    variant: str
    spec: dict
    market: str
    seed: int
    loop: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    label: str = ""  # QUICK, ORACLE, RETUNED ... (joined by "+")
    segment_at: int = 0  # >0: also report the segments before and from this bar
    log_dir: str = ""  # non-empty: write logs there and validate them
    config_hash: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.experiment, self.variant, self.seed)


@functools.lru_cache(maxsize=2)
def _market(name: str, seed: int):
    cfg = load_market_config(name)
    m, tr = generate(cfg, seed)
    return cfg, m, tr


def cost_total(f: dict) -> np.ndarray:
    return f["spread_cost"] + f["impact_cost"] + f["commission"] + f["borrow"] + f["financing"]


class RecordingProvider:
    """Wraps a provider and records its forecasts (for the measured forecast quality)."""

    def __init__(self, inner):
        self.inner = inner
        self.provider_id = inner.provider_id
        self.oracle = inner.oracle
        self.records = []

    def forecast(self, state):
        f = self.inner.forecast(state)
        self.records.append((state.bar, state.ids, np.array(f.mu)))
        return f


def forecast_quality(records, truth) -> float:
    """Pooled uncentred correlation sum(mu m) / sqrt(sum mu^2 sum m^2) of the recorded forecasts
    with the truth's conditional mean m over instruments and decisions."""
    col = {iid: j for j, iid in enumerate(truth.ids)}
    num = a = b = 0.0
    for bar, ids, mu in records:
        m = truth.m[bar, [col[i] for i in ids]]
        num += float(mu @ m)
        a += float(mu @ mu)
        b += float(m @ m)
    return num / np.sqrt(a * b) if a > 0 and b > 0 else 0.0


class RiskProbe:
    """Wraps a pipeline strategy's decide() and records the risk estimate and weights (S4)."""

    def __init__(self, strategy):
        self.s = strategy
        self.records = []

    def __getattr__(self, name):
        return getattr(self.s, name)

    def decide(self, state):
        from dynamic_trading_engine.strategies.pipeline import optimized_columns

        d = self.s.decide(state)
        cols = optimized_columns(state, self.s.spec.min_history)
        risk = self.s.risk_model.estimate(state, cols) if len(cols) else None
        if risk is not None and d.weights is not None:
            R = np.asarray(state.returns)[-risk.h :, cols]
            self.records.append(
                (state.bar, [state.ids[c] for c in cols], risk.sigma, d.weights[cols], R)
            )
        return d


def s4_quantities(records, truth, market, k_hold: int) -> dict:
    from dynamic_trading_engine.risk.models import sample_cov

    col = {iid: j for j, iid in enumerate(truth.ids)}
    fro, fro_s, real2, pred2 = [], [], [], []
    for bar, ids, S, w, R in records:
        idx = [col[i] for i in ids]
        if bar + 1 >= market.n_bars:
            continue
        St = truth.cov(bar + 1, np.array(idx))
        nt = np.linalg.norm(St)
        fro.append(np.linalg.norm(S - St) / nt)
        if R.shape[0] > 1:
            fro_s.append(np.linalg.norm(sample_cov(R) - St) / nt)
        hi = min(market.n_bars, bar + 1 + k_hold)
        rr = market.returns[bar + 1 : hi][:, idx]
        if rr.shape[0] == 0:
            continue
        growth = np.nanprod(1.0 + np.where(np.isnan(rr), 0.0, rr), axis=0) - 1.0
        real2.append(float(w @ growth) ** 2)
        pred2.append(float(rr.shape[0] * (w @ S @ w)))
    out = {
        "frobenius_rel": float(np.mean(fro)) if fro else float("nan"),
        "calibration": float(np.sqrt(np.mean(real2) / np.mean(pred2))) if pred2 else float("nan"),
    }
    if fro_s:
        out["share_below_sample"] = float(np.mean(np.array(fro) < np.array(fro_s)))
    return out


S8_K = 13
S8_PATHS = 20


def s8_recost(logs: dict, market, costs: CostConfig, seed: int) -> dict:
    """S8: every parent order of the run executed in the execution simulator with market,
    twap, and vwap over the fill bar (13 intraday intervals, U-shaped expected volume, volume
    noise 0.3, the contract's square-root impact, spread, and commission, the cap per interval),
    against the single-print cost the loop charged. Does not feed back into the loop."""
    from dynamic_trading_engine.execution import simulator as X
    from dynamic_trading_engine.rng import stream

    col = {iid: j for j, iid in enumerate(market.ids)}
    t_index = {int(t): k for k, t in enumerate(market.ts_event)}
    fills = {f[1]: f for f in logs["fills"] if f[1] != "DELIST"}
    cfg = X.ExecConfig(
        impact=costs.impact.model if costs.impact.model in ("sqrt", "linear") else "sqrt",
        y=costs.impact.y,
        half_spread_bps=costs.half_spread_bps,
        commission_bps=costs.commission_bps,
        commission_per_share=costs.commission_per_share,
        participation_cap=costs.participation_cap,
        volume_noise_sd=0.3,
    )
    tau = tuple([1.0 / S8_K] * S8_K)
    g = stream(seed, "experiment.S8")
    loop_bps, sim = [], {"market": [], "twap": [], "vwap": []}
    loop_capped = sim_capped = n = 0
    for oid, ts, iid, qty, *_ in logs["orders"]:
        k = t_index[ts]
        j = col[iid]
        sig, adv = market.sigma_bar[k, j], market.adv[k, j]
        f = fills.get(oid)
        if f is None or not (sig > 0 and adv > 0):
            continue
        q, m = abs(f[4]), f[5]
        loop_bps.append((f[7] + f[8] + f[9]) / (q * m) * 1e4)
        loop_capped += abs(f[4]) < abs(qty) - 1e-9
        Q = abs(qty)
        order = X.ParentOrder(
            1 if qty > 0 else -1,
            Q,
            m,
            float(sig),
            float(adv),
            tau,
            X.u_profile(S8_K, float(adv), 1.0),
        )
        xi = g.standard_normal((S8_PATHS, S8_K))
        zv = g.standard_normal((S8_PATHS, S8_K))
        for name, pol in (
            ("market", X.market_policy(order)),
            ("twap", X.twap_policy(order)),
            ("vwap", X.vwap_policy(order)),
        ):
            out = X.simulate(order, pol, cfg, xi, zv)
            exec_cost = out["spread"] + out["temporary"] + out["commission"]
            filled = out["completion"] * Q
            ok = filled > 0
            sim[name].append(
                float(np.mean(exec_cost[ok] / (filled[ok] * m))) * 1e4 if ok.any() else np.nan
            )
            if name == "market":
                sim_capped += float(np.mean(out["completion"] < 1 - 1e-12))
        n += 1
    out = {
        "s8_orders": n,
        "s8_loop_cost_bps": float(np.mean(loop_bps)) if loop_bps else np.nan,
        "s8_loop_capped_share": loop_capped / n if n else np.nan,
        "s8_market_capped_share": sim_capped / n if n else np.nan,
    }
    for name, v in sim.items():
        out[f"s8_{name}_cost_bps"] = float(np.nanmean(v)) if v else np.nan
    return out


def run_job(job: Job) -> dict:
    mcfg, market, truth = _market(job.market, job.seed)
    costs = cost_config_from_dict(job.costs) if job.costs else CostConfig()
    loop_kw = {"warmup_bars": mcfg.warmup_bars, **job.loop}
    lcfg = LoopConfig(**loop_kw)
    clock = time.perf_counter
    strategy = build_strategy(
        job.spec, costs, lcfg.rebalance_every, truth=truth, seed=job.seed, clock=clock
    )
    rec = None
    if job.extra.get("measure_quality") and hasattr(strategy, "provider"):
        rec = RecordingProvider(strategy.provider)
        strategy.provider = rec
    probe = None
    if job.extra.get("s4_probe"):
        probe = RiskProbe(strategy)
        strategy = probe
    w0, c0 = time.perf_counter(), time.process_time()
    want_logs = bool(job.log_dir) or bool(job.extra.get("s8_recost"))
    res = run_loop(market, strategy, costs, lcfg, write_logs=want_logs)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    f = res.flows
    start = res.decision_bars[0] if res.decision_bars else 0
    s = summarize(
        res.equity,
        f["hold_pnl"] + f["trade_pnl"],
        cost_total(f),
        res.turnover,
        res.gross_exposure,
        res.net_exposure,
        market.ppy,
        start,
    )
    row = {
        "experiment": job.experiment,
        "variant": job.variant,
        "seed": job.seed,
        "market": job.market,
        "config_hash": job.config_hash,
        "label": job.label,
        "oracle": bool(strategy.oracle),
        "forecast": strategy.forecast_id,
        "risk_model": strategy.risk_id,
        **s,
        "n_decisions": res.n_decisions,
        "n_hold": res.n_holds,
        "hold_frac": res.n_holds / res.n_decisions if res.n_decisions else 0.0,
        "n_optimal": res.status_counts.get("optimal", 0),
        "n_inaccurate": res.status_counts.get("optimal_inaccurate", 0),
        "n_failed": res.status_counts.get("failed", 0),
        "max_violation": res.max_violation,
        "max_identity_residual": res.max_identity_residual,
        "n_leverage_scaled": len(res.leverage_events),
        "impact_unavailable": res.impact_unavailable,
        "n_fills": res.n_fills,
        "n_capped": res.n_capped,
        "wall_run_s": wall,
        "wall_cpu_s": cpu,
        "wall_solve_ms_mean": float(np.mean(res.wall_solve_ms)) if res.wall_solve_ms else 0.0,
    }
    if job.segment_at:
        a = job.segment_at
        n1, m1, v1 = segment(res.equity, start, start + 1, a)
        n2, m2, v2 = segment(res.equity, start, a, len(res.equity))
        row.update(
            {
                "pre_n": n1,
                "pre_mean": m1,
                "pre_var": v1,
                "post_n": n2,
                "post_mean": m2,
                "post_var": v2,
            }
        )
    if rec is not None:
        row["forecast_quality"] = forecast_quality(rec.records, truth)
    if probe is not None:
        row.update(s4_quantities(probe.records, truth, market, lcfg.rebalance_every))
    if job.extra.get("s8_recost"):
        row.update(s8_recost(res.logs, market, costs, job.seed))
    for k, v in job.extra.items():
        if k not in ("measure_quality", "s4_probe", "s8_recost"):
            row[k] = v
    if job.log_dir:
        mdir = os.path.join(job.log_dir, "market")
        rdir = os.path.join(job.log_dir, "run")
        write_market(market, mdir)
        meta = {
            "costs": _cost_dict(costs),
            "initial_cash": lcfg.initial_cash,
            "strategy_id": strategy.strategy_id,
            "oracle": bool(strategy.oracle),
            "label": job.label,
            "market": job.market,
            "seed": job.seed,
        }
        write_run_logs(res.logs, rdir, meta, market.generator, market.seed)
        rep = validate_run(rdir, mdir)
        row["validated"] = True
        row["validator_max_residual"] = rep["max_identity_residual"]
    return row


def _cost_dict(c: CostConfig) -> dict:
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
