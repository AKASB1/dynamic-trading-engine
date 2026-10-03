"""Microbenchmarks: solves per second by problem type and size, loop bars per second, simulator
paths per second. Wall and CPU time, minimum of five repetitions; machine-dependent values that
never enter a comparison."""

from __future__ import annotations

import os
import time

import numpy as np

from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.output import write_csv
from dynamic_trading_engine.experiments.runner import manifest


def _min_time(fn, reps):
    walls, cpus = [], []
    for _ in range(reps):
        w, c = time.perf_counter(), time.process_time()
        fn()
        walls.append(time.perf_counter() - w)
        cpus.append(time.process_time() - c)
    return min(walls), min(cpus)


def run_bench(out: str | None, quick: bool) -> str:
    from dynamic_trading_engine.contracts.costs import CostConfig
    from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
    from dynamic_trading_engine.execution import simulator as X
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.generator import generate
    from dynamic_trading_engine.optimization.optimizers import Optimizer, OptimizerConfig
    from dynamic_trading_engine.optimization.testing import random_forecast, random_state
    from dynamic_trading_engine.risk.models import RiskModel
    from dynamic_trading_engine.rng import stream
    from dynamic_trading_engine.strategies.factory import build_strategy

    reps = 5
    rows = []
    g = stream(1, "bench")
    costs = CostConfig()
    for n in (10, 30) if quick else (10, 30, 100):
        st = random_state(g, n)
        cols = np.arange(n)
        risk = RiskModel("lw").estimate(st, cols)
        fc = random_forecast(g, n)
        for name in ("min_variance", "mean_variance", "cvar", "robust_box", "mpc"):
            opt = Optimizer(OptimizerConfig(name, gamma=5.0, eta_cvar=0.03, H=5), "LS", costs, 5)
            opt.solve(st, cols, fc, risk)
            k = 3 if quick else 10
            w, c = _min_time(
                lambda opt=opt, st=st, cols=cols, fc=fc, risk=risk, k=k: [
                    opt.solve(st, cols, fc, risk) for _ in range(k)
                ],
                reps,
            )
            rows.append(
                {
                    "benchmark": f"solve:{name}",
                    "n": n,
                    "per_second_wall": k / w,
                    "wall_s": w / k,
                    "wall_cpu_s": c / k,
                }
            )
    market = "tiny" if quick else "base"
    cfg = load_market_config(market)
    m, tr = generate(cfg, 1)
    reg = R.load_registry()
    for name in ("EW", "MV"):
        spec = R.base_spec(reg, name)
        st = build_strategy(spec, costs, 5, truth=tr, seed=1)
        run_loop(m, st, costs, LoopConfig(warmup_bars=cfg.warmup_bars))
        w, c = _min_time(
            lambda st=st: run_loop(m, st, costs, LoopConfig(warmup_bars=cfg.warmup_bars)), reps
        )
        rows.append(
            {
                "benchmark": f"loop:{name}:{market}",
                "n": m.n,
                "per_second_wall": m.n_bars / w,
                "wall_s": w,
                "wall_cpu_s": c,
            }
        )
    K, P = 13, 2000
    order = X.ParentOrder(1, 2e4, 50.0, 0.02, 1e6, tuple([1 / K] * K), X.u_profile(K, 1e6, 1.0))
    xi, zv = X.draws(1, P, K)
    pol = X.twap_policy(order)
    w, c = _min_time(lambda: X.simulate(order, pol, X.ExecConfig(), xi, zv), reps)
    rows.append(
        {
            "benchmark": "simulator:twap",
            "n": K,
            "per_second_wall": P / w,
            "wall_s": w,
            "wall_cpu_s": c,
        }
    )
    out = out or (R.path(R.OUTPUTS_DIR, "quick", "bench") if quick else R.path(R.RESULTS_DIR))
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "bench_quick.csv" if quick else "bench.csv")
    write_csv(p, rows, ["benchmark", "n"])
    from dynamic_trading_engine.contracts.canonical import write_json

    write_json(
        os.path.join(out, "bench_manifest_quick.json" if quick else "bench_manifest.json"),
        manifest(R.path(), 1, {"repetitions": reps, "statistic": "minimum over repetitions"}),
    )
    for r in rows:
        print(
            f"{r['benchmark']:28s} n={r['n']:4d} {r['per_second_wall']:12.1f}/s "
            f"wall {r['wall_s'] * 1e3:9.2f} ms cpu {r['wall_cpu_s'] * 1e3:9.2f} ms"
        )
    return p
