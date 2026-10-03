"""Tier-2 experiments: each Tier-2 strategy is tuned once under the protocol of S2 (16 trials,
tuning seeds 100-103, stream ``tune.<strategy>``) and frozen in ``frozen_tier2.json``
(experiments/tuned/) before its evaluation; the Tier-1 files stay untouched and every Tier-2
result goes into new files under ``experiments/results/tier2/``. The evaluation refuses a
configuration that is in neither frozen file and applies the seed rules unchanged."""

from __future__ import annotations

import json
import os
import time

from dynamic_trading_engine.analytics.stats import mean_ci, paired
from dynamic_trading_engine.contracts.canonical import config_hash, read_json, write_json
from dynamic_trading_engine.experiments import protocol as PT
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.criterion import criterion, criterion_segment
from dynamic_trading_engine.experiments.jobs import Job, run_job
from dynamic_trading_engine.experiments.output import read_csv, write_csv
from dynamic_trading_engine.experiments.runner import manifest, run_parallel

REG2 = os.path.join("experiments", "configs", "registry_tier2.json")
FROZEN2 = os.path.join("experiments", "tuned", "frozen_tier2.json")
OUT2 = os.path.join("experiments", "results", "tier2")


def load_registry2() -> dict:
    with open(R.path(REG2), encoding="utf-8") as fh:
        return json.load(fh)


def base_spec2(reg: dict, reg2: dict, name: str) -> dict:
    d = reg["defaults"]
    spec = R._merge(
        {**d["spec"], "optimizer": dict(d["optimizer"])}, reg2["strategies"][name]["spec"]
    )
    spec["strategy_id"] = name
    return spec


def tune2(workers: int, names: list[str], log=print) -> dict:
    """The protocol tuning of the named Tier-2 strategies (same trials, seeds, and stream rule)."""
    reg, reg2 = R.load_registry(), load_registry2()
    seeds = reg["seeds"]["tuning"]
    PT.check_tuning_seeds(seeds)
    trials = reg["tuning"]["trials"]
    market = reg["markets"]["main"]
    jobs, meta = [], {}
    for name in names:
        space = reg2["strategies"][name]["space"]
        base = base_spec2(reg, reg2, name)
        for i in range(trials):
            params = R.sample_trial(space, name, i)
            spec = R.apply_params(base, params)
            h = PT.run_config_hash(spec, market)
            meta[(name, i)] = (params, spec, h)
            for s in seeds:
                jobs.append(
                    Job("TUNE2", f"{name}#{i}", spec, market, s, label="TUNE", config_hash=h)
                )
    log(f"tier-2 tuning: {len(jobs)} runs")
    rows = run_parallel(run_job, jobs, workers)
    by = {}
    for j, r in zip(jobs, rows):
        by.setdefault(j.variant, []).append(r)
    path = R.path(FROZEN2)
    frozen = (
        read_json(path)
        if os.path.exists(path)
        else {"entries": [], "hashes": [], "tuned": {}, "log": []}
    )
    for name in names:
        best = None
        for i in range(trials):
            rs = by[f"{name}#{i}"]
            score = sum(criterion(r) for r in rs) / len(rs)
            params, spec, h = meta[(name, i)]
            frozen["log"].append(
                {
                    "strategy": name,
                    "trial": i,
                    "config_hash": h,
                    "seeds": seeds,
                    "mean_ce_ann": score,
                    "params": params,
                }
            )
            if best is None or score > best[0]:
                best = (score, i, params, spec)
        frozen["tuned"][name] = {
            "trial": best[1],
            "mean_ce_ann": best[0],
            "params": best[2],
            "spec": best[3],
        }
    write_json(path, frozen)
    return frozen


def freeze2(entries: list[dict]) -> None:
    path = R.path(FROZEN2)
    frozen = read_json(path)
    known = {e["id"] for e in frozen["entries"]}
    for e in entries:
        if e["id"] not in known:
            frozen["entries"].append(e)
            known.add(e["id"])
    frozen["hashes"] = sorted({e["hash"] for e in frozen["entries"]})
    write_json(path, frozen)


def check_frozen2(jobs: list[Job]) -> None:
    allowed = set(read_json(R.path(FROZEN2))["hashes"]) | PT.load_frozen_hashes(False)
    for j in jobs:
        if j.config_hash not in allowed:
            raise PT.ProtocolError(f"tier-2 configuration {j.variant} is in neither frozen file")


def tier1_rows() -> list[dict]:
    rows = read_csv(R.path(R.RESULTS_DIR, "runs.csv"))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = (
                    float(v)
                    if k not in ("experiment", "variant", "market", "label", "config_hash")
                    else v
                )
            except ValueError:
                pass
        r["seed"] = int(r["seed"])
    return rows


def run_eval2(jobs: list[Job], workers: int, log=print) -> list[dict]:
    PT.check_eval_seeds(sorted({j.seed for j in jobs}), quick=False)
    check_frozen2(jobs)
    log(f"tier-2 evaluation: {len(jobs)} runs")
    rows = run_parallel(run_job, jobs, workers)
    rows.sort(key=lambda r: (r["experiment"], r["variant"], r["seed"]))
    return rows


def write_manifest2(name: str, workers: int, extra: dict) -> None:
    write_json(R.path(OUT2, f"{name}_manifest.json"), manifest(R.path(), workers, extra))


def summarize_vs(rows2, rows1, exp1, refs, seg=None):
    """ce_ann of the Tier-2 rows with paired differences against Tier-1 rows of the same seeds."""

    def val(r):
        return criterion_segment(r, seg) if seg else criterion(r)

    by1 = {}
    for r in rows1:
        if r["experiment"] == exp1:
            by1.setdefault(r["variant"], {})[r["seed"]] = val(r)
    by2 = {}
    for r in rows2:
        by2.setdefault(r["variant"], {})[r["seed"]] = val(r)
    out = []
    for v, d in sorted(by2.items()):
        m, ci, n = mean_ci(list(d.values()))
        row = {"strategy": v, "n_seeds": n, "ce_ann_mean": m, "ce_ann_ci95": ci}
        for ref in refs:
            if ref in by1:
                p = paired(by1[ref], d)
                row[f"diff_vs_{ref}"] = p["diff_mean"]
                row[f"diff_vs_{ref}_ci95"] = p["diff_ci"]
                row[f"wtl_vs_{ref}"] = f"{p['wins']}/{p['ties']}/{p['losses']}"
        out.append(row)
    return out


def now() -> float:
    return time.perf_counter()


def registry_hash() -> str:
    return config_hash(load_registry2())


# ---------------------------------------------------------------- items 1 and 2


def plan_items12(reg: dict, reg2: dict, frozen: dict) -> list[Job]:
    """WDRO on base and shift (the seeds of S5, segments at the shift bar), GP_AIM on base (the
    seeds of S2)."""
    jobs = []
    at = reg["S5"]["at_bar"]
    if "WDRO" in frozen["tuned"]:
        spec = frozen["tuned"]["WDRO"]["spec"]
        for exp, market in (
            ("T2_WDRO_base", reg["markets"]["main"]),
            ("T2_WDRO_shift", reg["markets"]["shift"]),
        ):
            h = PT.run_config_hash(spec, market)
            for s in reg["seeds"]["S5"]:
                jobs.append(
                    Job(exp, "WDRO", spec, market, s, label="TIER2", segment_at=at, config_hash=h)
                )
    if "GP_AIM" in frozen["tuned"]:
        spec = frozen["tuned"]["GP_AIM"]["spec"]
        market = reg["markets"]["main"]
        h = PT.run_config_hash(spec, market)
        for s in reg["seeds"]["S2"]:
            jobs.append(
                Job("T2_GP", "GP_AIM", spec, market, s, label="TIER2", segment_at=at, config_hash=h)
            )
    return jobs


def freeze_items12() -> str:
    reg, reg2 = R.load_registry(), load_registry2()
    frozen = read_json(R.path(FROZEN2))
    entries = [
        {
            "id": f"{j.experiment}/{j.variant}",
            "experiment": j.experiment,
            "variant": j.variant,
            "market": j.market,
            "spec": j.spec,
            "hash": j.config_hash,
        }
        for j in plan_items12(reg, reg2, frozen)
    ]
    freeze2(entries)
    return R.path(FROZEN2)


def run_items12(workers: int, log=print) -> dict:
    reg, reg2 = R.load_registry(), load_registry2()
    frozen = read_json(R.path(FROZEN2))
    jobs = plan_items12(reg, reg2, frozen)
    t0 = now()
    rows = run_eval2(jobs, workers, log)
    rows1 = tier1_rows()
    out = {}
    first = ["experiment", "variant", "seed", "market", "label", "config_hash"]
    os.makedirs(R.path(OUT2), exist_ok=True)
    write_csv(R.path(OUT2, "runs_tier2.csv"), rows, first)
    wb = [r for r in rows if r["experiment"] == "T2_WDRO_base"]
    ws = [r for r in rows if r["experiment"] == "T2_WDRO_shift"]
    if wb:
        # effect of the shift on WDRO (paired over seeds, bars from the shift on) next to Tier-1 S5
        eff = paired(
            {r["seed"]: criterion_segment(r, "post") for r in wb},
            {r["seed"]: criterion_segment(r, "post") for r in ws},
        )
        rows_w = summarize_vs(ws, rows1, "S5", ["EW", "MV", "CVAR", "ROBUST"], seg="post")
        for r in rows_w:
            r.update(
                {"effect_of_shift_mean": eff["diff_mean"], "effect_of_shift_ci95": eff["diff_ci"]}
            )
        full_base = summarize_vs(wb, rows1, "S2", ["CASH", "EW", "MV", "CVAR", "ROBUST"])
        for r in full_base:
            r["strategy"] = "WDRO (base, whole sample)"
        for r in rows_w:
            r["strategy"] = "WDRO (shift, bars from the shift on)"
        write_csv(
            R.path(OUT2, "wdro.csv"),
            full_base + rows_w,
            ["strategy", "n_seeds", "ce_ann_mean", "ce_ann_ci95"],
        )
        out["wdro"] = full_base + rows_w
    gp = [r for r in rows if r["experiment"] == "T2_GP"]
    if gp:
        rows_g = summarize_vs(gp, rows1, "S2", ["CASH", "EW", "MV", "MPC"])
        for r in rows_g:
            r["gross_exposure"] = sum(x["gross_exposure"] for x in gp) / len(gp)
            r["turnover"] = sum(x["turnover"] for x in gp) / len(gp)
        write_csv(
            R.path(OUT2, "gp_aim.csv"),
            rows_g,
            ["strategy", "n_seeds", "ce_ann_mean", "ce_ann_ci95"],
        )
        out["gp_aim"] = rows_g
    write_manifest2(
        "items12",
        workers,
        {
            "n_runs": len(rows),
            "wall_total_s": now() - t0,
            "frozen_tier2": FROZEN2,
            "registry_tier2_hash": registry_hash(),
        },
    )
    log(f"tier-2 items 1-2 done: {len(rows)} runs")
    return out
