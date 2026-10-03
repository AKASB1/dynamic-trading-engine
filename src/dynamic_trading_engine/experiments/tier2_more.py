"""Tier-2 studies 3 to 8 (each writes its own files under ``experiments/results/tier2/``):

- ``adaptive``: adaptive execution and the volume-forecast error study (paired, common random
  numbers);
- ``solvers``: the solver comparison;
- ``tails``: Student-t and jump variants of the generator, S5 protocol for MV, CVAR, ROBUST
  (frozen S2 configurations; "assumed tail model");
- ``costmis``: cost misspecification by capital and impact level (tuned MV, paired against parity);
- ``capacity``: ce_ann against capital for the tuned MV and the naive RANK_LS, cap share;
- ``export``: the export consumer run on an export written by this repository.
Loop runs use frozen configurations only (``frozen.json`` or ``frozen_tier2.json``).
"""

from __future__ import annotations

import copy
import math
import os
import tempfile

import numpy as np

from dynamic_trading_engine.analytics.stats import mean_ci, paired
from dynamic_trading_engine.experiments import protocol as PT
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.criterion import criterion, criterion_segment
from dynamic_trading_engine.experiments.jobs import Job
from dynamic_trading_engine.experiments.output import write_csv
from dynamic_trading_engine.experiments.tier2 import (
    FROZEN2,
    OUT2,
    freeze2,
    now,
    run_eval2,
    write_manifest2,
)
from dynamic_trading_engine.rng import stream


def _frozen_s2():
    reg = R.load_registry()
    return reg, PT.frozen_specs(reg, PT.load_tuned(reg, False))


def _entries(jobs):
    return [
        {
            "id": f"{j.experiment}/{j.variant}",
            "experiment": j.experiment,
            "variant": j.variant,
            "market": j.market,
            "spec": j.spec,
            "loop": j.loop,
            "costs": j.costs,
            "hash": j.config_hash,
        }
        for j in jobs
    ]


def _out(name):
    os.makedirs(R.path(OUT2), exist_ok=True)
    return R.path(OUT2, name)


# ---------------------------------------------------------------- 3 adaptive execution


def run_adaptive(workers: int, log=print) -> list[dict]:
    from dynamic_trading_engine.execution import simulator as X
    from dynamic_trading_engine.execution.adaptive import AdaptiveACPolicy

    t0 = now()
    reg = R.load_registry()
    c7 = reg["S7"]
    K, T, S0, sig, adv = c7["K"], c7["T_bars"], c7["S0"], c7["sigma_bar"], c7["adv"]
    seeds, P = reg["seeds"]["S7"], c7["paths"]
    tau = tuple([T / K] * K)
    u = X.u_profile(K, adv, T)
    flat = X.flat_profile(K, adv, T)
    Q = 0.05 * adv * T
    lam = c7["b"]["ac_lambda"]
    scenarios = [
        ("correct profile", u, None, 0.0),
        ("expected flat, true U-shaped", flat, u, 0.0),
        ("persistent volume level (sd 0.5)", u, None, 0.5),
        ("flat expected + level (sd 0.5)", flat, u, 0.5),
    ]
    rows = []
    for name, expected, true, lsd in scenarios:
        order = X.ParentOrder(1, Q, S0, sig, adv, tau, expected)
        cfg = X.ExecConfig(
            impact="sqrt",
            volume_noise_sd=0.3,
            participation_cap=0.1,
            volume_level_sd=lsd,
            true_profile=true,
        )
        q_twap = Q / K
        eta_eq = S0 * cfg.y * sig * math.sqrt(q_twap / (tau[0] * adv)) * tau[0] / q_twap
        pols = {
            "vwap": X.vwap_policy(order),
            "twap": X.twap_policy(order),
            "ac_static": X.ac_policy(order, lam, eta_eq, 0.0),
            "ac_adaptive": AdaptiveACPolicy(order, cfg, lam, adapt=True),
            "ac_replan_no_volume": AdaptiveACPolicy(order, cfg, lam, adapt=False),
        }
        per = {k: {} for k in pols}
        compl = {k: [] for k in pols}
        for seed in seeds:
            xi, zv = X.draws(seed, P, K)
            zl = stream(seed, "exec.volume_level").standard_normal(P)
            for pn, pol in pols.items():
                out = X.simulate(order, pol, cfg, xi, zv, zl=zl)
                per[pn][seed] = float(np.mean(out["IS_bps"]))
                compl[pn].append(float(np.mean(out["completion"])))
        for pn in pols:
            m, ci, n = mean_ci(list(per[pn].values()))
            row = {
                "scenario": name,
                "policy": pn,
                "IS_bps_mean": m,
                "IS_bps_ci95": ci,
                "completion": float(np.mean(compl[pn])),
                "seeds": n,
                "paths_per_seed": P,
            }
            for ref in ("ac_static", "vwap"):
                if pn != ref:
                    p = paired(per[ref], per[pn])  # positive = more cost than the reference
                    row[f"diff_vs_{ref}_bps"] = p["diff_mean"]
                    row[f"diff_vs_{ref}_ci95"] = p["diff_ci"]
                    row[f"lower_cost_than_{ref}"] = p["losses"]  # seeds where the policy cost less
            rows.append(row)
    write_csv(_out("adaptive_execution.csv"), rows, ["scenario", "policy"])
    write_manifest2(
        "adaptive", 1, {"wall_total_s": now() - t0, "Q_frac_adv": 0.05, "lambda_ac": lam}
    )
    log(f"adaptive execution: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------- 4 solver comparison


def run_solvers(workers: int, log=print) -> list[dict]:
    from dynamic_trading_engine.experiments.solvers import compare

    t0 = now()
    rows, solvers = compare(n_problems=10, sizes=(10, 30), seed=1)
    write_csv(_out("solver_comparison.csv"), rows, ["problem", "n", "solver"])
    write_manifest2(
        "solvers",
        1,
        {"wall_total_s": now() - t0, "solvers": solvers, "gurobi": "GUROBI" in solvers},
    )
    log(f"solver comparison: {len(rows)} rows, solvers {solvers}")
    return rows


# ---------------------------------------------------------------- 5 fat tails


TAIL_MARKETS = (("base_t5", "shift_t5"), ("base_jump", "shift_jump"))
TAIL_STRATEGIES = ("EW", "MV", "CVAR", "ROBUST")


def _tail_jobs():
    reg, specs = _frozen_s2()
    at = reg["S5"]["at_bar"]
    jobs = []
    for pair in TAIL_MARKETS:
        for market in pair:
            for name in TAIL_STRATEGIES:
                h = PT.run_config_hash(specs[name], market)
                for s in reg["seeds"]["S5"]:
                    jobs.append(
                        Job(
                            "T2_TAILS",
                            f"{market}|{name}",
                            specs[name],
                            market,
                            s,
                            label="TIER2",
                            segment_at=at,
                            config_hash=h,
                        )
                    )
    return jobs


def run_tails(workers: int, log=print) -> list[dict]:
    t0 = now()
    rows = run_eval2(_tail_jobs(), workers, log)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["seed"]] = r
    out = []
    for base, shift in TAIL_MARKETS:
        for name in TAIL_STRATEGIES:
            b, s = by[f"{base}|{name}"], by[f"{shift}|{name}"]
            eff = paired(
                {k: criterion_segment(v, "post") for k, v in b.items()},
                {k: criterion_segment(v, "post") for k, v in s.items()},
            )
            m, ci, n = mean_ci([criterion(v) for v in b.values()])
            row = {
                "tail_model": base.split("_")[1],
                "strategy": name,
                "ce_ann_base_mean": m,
                "ce_ann_base_ci95": ci,
                "effect_of_shift_mean": eff["diff_mean"],
                "effect_of_shift_ci95": eff["diff_ci"],
                "label": "assumed tail model",
                "n_seeds": n,
            }
            if name != "MV":
                p = paired(
                    {k: criterion(v) for k, v in by[f"{base}|MV"].items()},
                    {k: criterion(v) for k, v in b.items()},
                )
                row.update(
                    {
                        "base_diff_vs_MV": p["diff_mean"],
                        "base_diff_vs_MV_ci95": p["diff_ci"],
                        "base_wtl_vs_MV": f"{p['wins']}/{p['ties']}/{p['losses']}",
                    }
                )
                q = paired(
                    {k: criterion_segment(v, "post") for k, v in by[f"{shift}|MV"].items()},
                    {k: criterion_segment(v, "post") for k, v in s.items()},
                )
                row.update(
                    {
                        "post_shift_diff_vs_MV": q["diff_mean"],
                        "post_shift_diff_vs_MV_ci95": q["diff_ci"],
                    }
                )
            out.append(row)
    write_csv(_out("fat_tails.csv"), out, ["tail_model", "strategy"])
    write_csv(_out("fat_tails_runs.csv"), rows, ["experiment", "variant", "seed"])
    write_manifest2("tails", workers, {"wall_total_s": now() - t0, "n_runs": len(rows)})
    return out


# ---------------------------------------------------------------- 6 cost misspecification


def _costmis_jobs():
    reg, specs = _frozen_s2()
    mv = specs["MV"]
    jobs = []
    for y in (0.25, 0.5, 1.0):
        for capital in (1e7, 1e8):
            for f in (0.5, 1.0, 2.0):
                sp = copy.deepcopy(mv)
                sp["optimizer"]["cost_scale"] = (
                    f  # the optimizer's belief relative to the true costs
                )
                costs = {"impact": {"model": "sqrt", "y": y}}
                loop = {"initial_cash": capital}
                h = PT.run_config_hash(sp, reg["markets"]["main"], loop, costs)
                for s in reg["seeds"]["S3"]:
                    jobs.append(
                        Job(
                            "T2_COSTMIS",
                            f"y={y:g}|capital={capital:g}|belief={f:g}",
                            sp,
                            reg["markets"]["main"],
                            s,
                            loop,
                            costs,
                            label="TIER2",
                            config_hash=h,
                        )
                    )
    return jobs


def run_costmis(workers: int, log=print) -> list[dict]:
    t0 = now()
    rows = run_eval2(_costmis_jobs(), workers, log)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["seed"]] = r
    out = []
    for v in sorted(by, key=lambda x: [float(p.split("=")[1]) for p in x.split("|")]):
        y, cap, f = (p.split("=")[1] for p in v.split("|"))
        ref = by[f"y={y}|capital={cap}|belief=1"]
        m, ci, n = mean_ci([criterion(r) for r in by[v].values()])
        row = {
            "impact_y": float(y),
            "capital": float(cap),
            "belief": float(f),
            "ce_ann_mean": m,
            "ce_ann_ci95": ci,
            "turnover": float(np.mean([r["turnover"] for r in by[v].values()])),
            "n_seeds": n,
        }
        if f != "1":
            p = paired(
                {k: criterion(r) for k, r in ref.items()},
                {k: criterion(r) for k, r in by[v].items()},
            )
            row.update(
                {
                    "loss_vs_parity": p["diff_mean"],
                    "loss_vs_parity_ci95": p["diff_ci"],
                    "wtl": f"{p['wins']}/{p['ties']}/{p['losses']}",
                }
            )
        out.append(row)
    write_csv(_out("cost_misspecification.csv"), out, ["impact_y", "capital", "belief"])
    write_manifest2("costmis", workers, {"wall_total_s": now() - t0, "n_runs": len(rows)})
    return out


# ---------------------------------------------------------------- 7 capacity

CAPITALS = (1e6, 1e7, 3e7, 1e8, 3e8, 1e9)


def _capacity_jobs():
    reg, specs = _frozen_s2()
    jobs = []
    for name in ("MV", "RANK_LS"):
        for capital in CAPITALS:
            loop = {"initial_cash": capital}
            h = PT.run_config_hash(specs[name], reg["markets"]["main"], loop)
            for s in reg["seeds"]["S3"]:
                jobs.append(
                    Job(
                        "T2_CAPACITY",
                        f"{name}|{capital:g}",
                        specs[name],
                        reg["markets"]["main"],
                        s,
                        loop,
                        label="TIER2",
                        config_hash=h,
                    )
                )
    return jobs


def run_capacity(workers: int, log=print) -> list[dict]:
    t0 = now()
    rows = run_eval2(_capacity_jobs(), workers, log)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["seed"]] = r
    out = []
    for name in ("MV", "RANK_LS"):
        for capital in CAPITALS:
            rs = by[f"{name}|{capital:g}"]
            m, ci, n = mean_ci([criterion(r) for r in rs.values()])
            capped = sum(r["n_capped"] for r in rs.values()) / max(
                1, sum(r["n_fills"] for r in rs.values())
            )
            row = {
                "strategy": name,
                "capital": capital,
                "ce_ann_mean": m,
                "ce_ann_ci95": ci,
                "cap_binds_share": capped,
                "cost_drag": float(np.mean([r["cost_drag"] for r in rs.values()])),
                "n_seeds": n,
            }
            if name == "MV":
                p = paired(
                    {k: criterion(r) for k, r in by[f"RANK_LS|{capital:g}"].items()},
                    {k: criterion(r) for k, r in rs.items()},
                )
                row.update(
                    {"diff_vs_RANK_LS": p["diff_mean"], "diff_vs_RANK_LS_ci95": p["diff_ci"]}
                )
            out.append(row)
    write_csv(_out("capacity.csv"), out, ["strategy", "capital"])
    write_manifest2("capacity", workers, {"wall_total_s": now() - t0, "n_runs": len(rows)})
    return out


# ---------------------------------------------------------------- 8 export consumer


def run_export(workers: int, log=print) -> dict:
    """Writes an export of a tiny market (seed 5) with the plain forecasts as its forecasts.csv,
    reads it back through the consumer, and runs EW and the export-forecast strategy on it."""
    from dynamic_trading_engine.contracts.costs import CostConfig
    from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
    from dynamic_trading_engine.market.config import load_market_config
    from dynamic_trading_engine.market.export import ExportForecast, read_export, write_export
    from dynamic_trading_engine.market.generator import generate
    from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
    from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec

    cfg = load_market_config("tiny")
    m, _tr = generate(cfg, 5)
    with tempfile.TemporaryDirectory() as d:
        write_export(m, d)
        m2, man, fcs = read_export(d)
    costs = CostConfig()
    lc = LoopConfig(warmup_bars=cfg.warmup_bars)
    out = {}
    for label, mk in (("in-memory", m), ("export", m2)):
        st = PipelineStrategy(
            StrategySpec(
                "EW", forecast="none", optimizer=OptimizerConfig("equal_weight"), book="LO"
            ),
            costs,
            5,
        )
        out[label] = run_loop(mk, st, costs, lc, write_logs=True).logs
    same = out["in-memory"] == out["export"]
    st = PipelineStrategy(
        StrategySpec(
            "RANK_EXPORT", forecast="none", optimizer=OptimizerConfig("rank_ls"), book="LS"
        ),
        costs,
        5,
        provider=ExportForecast(fcs),
    )
    res = run_loop(m2, st, costs, lc)
    rep = {
        "identical_logs_in_memory_vs_export": same,
        "export_files": sorted(man["files"]),
        "rank_export_decisions": res.n_decisions,
        "external_export_dir": os.path.isdir(os.path.join(os.path.dirname(R.path()), "_exports")),
    }
    log(f"export consumer: identical logs {same}")
    from dynamic_trading_engine.contracts.canonical import write_json

    write_json(_out("export_consumer.json"), rep)
    return rep


def freeze_more() -> str:
    """Add the loop configurations of the tail, cost-misspecification, and capacity studies to
    frozen_tier2.json (before they run)."""
    freeze2(_entries(_tail_jobs() + _costmis_jobs() + _capacity_jobs()))
    return R.path(FROZEN2)
