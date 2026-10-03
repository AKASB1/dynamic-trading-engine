"""Command-line entry points: ``python -m dynamic_trading_engine <command>``.

Commands: generate, run, validate, audit, tune, freeze, experiment, report, bench, sample,
quickhash. Every command is simulation only: no network, no broker, no live data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dynamic_trading_engine.experiments.runner import set_thread_env


def _repo(*p):
    from dynamic_trading_engine.market.config import repo_root

    return os.path.join(repo_root(), *p)


def cmd_generate(a):
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.generator import generate
    from dynamic_trading_engine.market.store import write_market

    m, _ = generate(load_market_config(a.market), a.seed)
    write_market(m, a.out)
    print(f"wrote market {a.market} seed {a.seed}: {m.n} instruments, {m.n_bars} bars -> {a.out}")


def _spec_for(name: str, tuned: bool, quick: bool):
    from dynamic_trading_engine.experiments import protocol as PT
    from dynamic_trading_engine.experiments import registry as R

    reg = R.load_registry()
    if tuned:
        return PT.frozen_specs(reg, PT.load_tuned(reg, quick))[name]
    return R.base_spec(reg, name)


def cmd_run(a):
    from dynamic_trading_engine.experiments.jobs import Job, run_job

    spec = _spec_for(a.strategy, a.tuned, a.quick)
    loop = {"rebalance_every": a.rebalance} if a.rebalance else {}
    job = Job("RUN", a.strategy, spec, a.market, a.seed, loop, {}, "", 0, a.out)
    row = run_job(job)
    print(
        f"{a.strategy} on {a.market} seed {a.seed}: ce_ann {row['ce_ann']:.4f}, "
        f"sharpe {row['sharpe']:.3f}, "
        f"decisions {row['n_decisions']}, holds {row['n_hold']}, validated {row.get('validated')}"
    )
    print(f"logs: {os.path.join(a.out, 'run')}  market: {os.path.join(a.out, 'market')}")


def cmd_validate(a):
    from dynamic_trading_engine.engine.validator import validate_run

    pairs = []
    if a.run:
        pairs.append(
            (a.run, a.market or os.path.join(os.path.dirname(os.path.abspath(a.run)), "market"))
        )
    if a.root:
        for dirpath, dirnames, _files in os.walk(a.root):
            if "run" in dirnames and "market" in dirnames:
                pairs.append((os.path.join(dirpath, "run"), os.path.join(dirpath, "market")))
    if not pairs:
        print("nothing to validate")
        return 1
    worst = 0.0
    for run, market in sorted(pairs):
        rep = validate_run(run, market)
        worst = max(worst, rep["max_identity_residual"])
    print(
        f"validated {len(pairs)} run(s): identity, positions, cash, costs, fill rules; "
        f"largest identity residual {worst:.3e}"
    )
    return 0


def cmd_audit(a):
    from dynamic_trading_engine.audit.run import GUARDS, run_audit
    from dynamic_trading_engine.experiments.output import write_csv

    rows = run_audit(a.canaries, a.instants, a.builtin_instants)
    write_csv(a.out, rows, ["subject", "oracle", "expected_guard", "instants", *GUARDS])
    bad = [
        r for r in rows if r["subject"].startswith("canary:") and r[r["expected_guard"]] != "caught"
    ]
    alarms = [r for r in rows if not r["subject"].startswith("canary:") and r["alarm"]]
    print(
        f"audit table -> {a.out}; canaries not caught by their guard: {len(bad)}; "
        f"built-in alarms: {len(alarms)}"
    )
    return 1 if bad or alarms else 0


def cmd_tune(a):
    from dynamic_trading_engine.experiments import protocol as PT
    from dynamic_trading_engine.experiments import registry as R

    t0 = time.perf_counter()
    PT.tune(R.load_registry(), a.quick, a.workers)
    print(f"tuning done in {time.perf_counter() - t0:.0f} s")


def cmd_freeze(a):
    from dynamic_trading_engine.experiments import registry as R
    from dynamic_trading_engine.experiments.suite import freeze

    print(f"frozen configurations -> {freeze(R.load_registry(), a.quick)}")


def cmd_experiment(a):
    from dynamic_trading_engine.experiments import registry as R
    from dynamic_trading_engine.experiments.suite import run_all

    only = tuple(a.only.split(",")) if a.only else None
    run_all(R.load_registry(), a.quick, a.workers, a.out, only)


def cmd_report(a):
    from dynamic_trading_engine.experiments.report import make_report

    md, png = make_report(a.run, a.out)
    print(f"report: {md} and {png}")


def cmd_sample(a):
    """The committed sample run: logs of the first 150 bars of the tiny market."""
    from dynamic_trading_engine.contracts.costs import CostConfig
    from dynamic_trading_engine.engine.logs import write_run_logs
    from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
    from dynamic_trading_engine.engine.validator import validate_run
    from dynamic_trading_engine.experiments import registry as R
    from dynamic_trading_engine.experiments.jobs import _cost_dict
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.generator import generate
    from dynamic_trading_engine.market.store import write_market
    from dynamic_trading_engine.strategies.factory import build_strategy

    cfg = load_market_config("tiny")
    m, tr = generate(cfg, 1)
    spec = R.base_spec(R.load_registry(), "MV")
    costs = CostConfig()
    lc = LoopConfig(warmup_bars=cfg.warmup_bars)
    st = build_strategy(spec, costs, lc.rebalance_every, truth=tr, seed=1)
    res = run_loop(m, st, costs, lc, write_logs=True, stop_after_bar=a.bars - 1)
    logs = {
        k: [
            r
            for r in v
            if (
                r[0]
                if k in ("positions", "equity", "decisions")
                else r[1]
                if k == "orders"
                else r[2]
            )
            <= int(m.ts_event[a.bars - 1])
        ]
        for k, v in res.logs.items()
    }
    meta = {
        "costs": _cost_dict(costs),
        "initial_cash": lc.initial_cash,
        "strategy_id": st.strategy_id,
        "oracle": st.oracle,
        "market": "tiny",
        "seed": 1,
        "label": "",
        "note": f"first {a.bars} bars of the tiny market, seed 1, MV at its default configuration",
    }
    write_run_logs(logs, a.out, meta, m.generator, m.seed)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        write_market(m, d)
        rep = validate_run(a.out, d)
    print(f"sample run -> {a.out} ({rep['bars']} bars, {rep['fills']} fills, validated)")


def cmd_quickhash(a):
    from dynamic_trading_engine.experiments.output import hash_without_wall

    root = a.dir
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".csv"):
            print(f"{hash_without_wall(os.path.join(root, fn))}  {fn}")


def cmd_tier2(a):
    from dynamic_trading_engine.experiments import tier2 as T2

    if a.action == "tune":
        T2.tune2(a.workers, a.names.split(","))
    elif a.action == "freeze":
        print(f"frozen tier-2 configurations -> {T2.freeze_items12()}")
    elif a.action == "run":
        T2.run_items12(a.workers)
    elif a.action == "freeze-more":
        from dynamic_trading_engine.experiments import tier2_more as T3

        print(f"frozen tier-2 configurations -> {T3.freeze_more()}")
    else:
        from dynamic_trading_engine.experiments import tier2_more as T3

        out = getattr(T3, f"run_{a.action}")(a.workers)
        if isinstance(out, dict):
            print(out)
    return 0


def cmd_bench(a):
    from dynamic_trading_engine.experiments.bench import run_bench

    run_bench(a.out, a.quick)


def main(argv=None) -> int:
    set_thread_env()
    ap = argparse.ArgumentParser(prog="dynamic_trading_engine", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("generate", help="write a synthetic market to contract files")
    p.add_argument("--market", default="tiny")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("run", help="one run of a registered strategy with logs (validated)")
    p.add_argument("--strategy", default="MV")
    p.add_argument("--tuned", action="store_true", help="use the frozen tuned configuration")
    p.add_argument("--quick", action="store_true", help="with --tuned: the quick tuning")
    p.add_argument("--market", default="tiny")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rebalance", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("validate", help="independent validation of run logs")
    p.add_argument("--run")
    p.add_argument("--market")
    p.add_argument("--root")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("audit", help="information audit: canaries and built-in strategies")
    p.add_argument("--canaries", default=os.path.join("tests", "canaries"))
    p.add_argument("--instants", type=int, default=30)
    p.add_argument("--builtin-instants", type=int, default=None)
    p.add_argument("--out", default=os.path.join("experiments", "results", "audit_table.csv"))
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("tune", help="protocol tuning on the tuning seeds")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("freeze", help="write the frozen configurations")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_freeze)
    p = sub.add_parser("experiment", help="the experiments S1-S7 and SENS")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only", default="")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_experiment)
    p = sub.add_parser("report", help="Markdown and PNG report of one run")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("sample", help="write the committed sample run")
    p.add_argument("--out", default=os.path.join("docs", "examples", "sample_run"))
    p.add_argument("--bars", type=int, default=150)
    p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("quickhash", help="SHA-256 of result files without wall_ columns")
    p.add_argument("--dir", default=os.path.join("experiments", "outputs", "quick", "results"))
    p.set_defaults(fn=cmd_quickhash)
    p = sub.add_parser(
        "tier2", help="Tier-2 items: tune, freeze, run (WDRO, GP_AIM), and the other studies"
    )
    p.add_argument("action")
    p.add_argument("--names", default="WDRO,GP_AIM")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_tier2)
    p = sub.add_parser("bench", help="microbenchmarks (wall and CPU time)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    rc = a.fn(a)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
