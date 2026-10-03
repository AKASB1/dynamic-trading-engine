"""Tuning on the tuning seeds, the frozen configurations, and the guards of the protocol.

- Tuning seeds are 100 to 199 (the protocol uses 100 to 103, quick mode 100 and 101); the tuner
  refuses any other seed. Every strategy with parameters gets the same number of random-search
  trials on the same seeds; trial i is a function of the strategy name and i only. The best
  trial has the highest mean ``ce_ann`` over the tuning seeds (ties: the lowest trial index).
- The S2b ladder re-tunes ``gamma`` of MV on a declared grid of 8 values per rung (its own
  budget); the RETUNED bound of S5 re-tunes every strategy with the same 16 trials on shift
  markets of the tuning seeds, scored on the bars from ``at_bar`` on.
- ``frozen.json`` lists the hash of every configuration the full evaluation runs; the
  evaluation refuses a seed below 1000 and a configuration whose hash is not in it.
  ``--quick`` uses ``frozen_quick.json`` and labels everything QUICK.
"""

from __future__ import annotations

import copy
import math
import os

from dynamic_trading_engine.analytics.metrics import ce_ann
from dynamic_trading_engine.contracts.canonical import (
    canonical_dict,
    config_hash,
    read_json,
    write_json,
)
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.criterion import criterion, criterion_segment
from dynamic_trading_engine.experiments.jobs import Job, run_job
from dynamic_trading_engine.experiments.runner import run_parallel
from dynamic_trading_engine.strategies.factory import spec_from_dict

TUNING_SEED_RANGE = range(100, 200)
EVAL_SEED_MIN = 1000
DEV_SEED_RANGE = range(1, 100)


class ProtocolError(RuntimeError):
    pass


def check_tuning_seeds(seeds) -> None:
    bad = [s for s in seeds if s not in TUNING_SEED_RANGE]
    if bad:
        raise ProtocolError(f"the tuner refuses seeds outside 100-199: {bad}")


def check_eval_seeds(seeds, quick: bool) -> None:
    if quick:
        bad = [s for s in seeds if s not in DEV_SEED_RANGE]
        if bad:
            raise ProtocolError(f"quick mode runs on development seeds 1-99 only: {bad}")
        return
    bad = [s for s in seeds if s < EVAL_SEED_MIN]
    if bad:
        raise ProtocolError(f"the evaluation refuses seeds below 1000: {bad}")


def run_config_hash(
    spec: dict, market: str, loop: dict | None = None, costs: dict | None = None
) -> str:
    return config_hash(
        {
            "costs": costs or {},
            "loop": loop or {},
            "market": market,
            "spec": canonical_dict(spec_from_dict(spec)),
        }
    )


def frozen_path(quick: bool) -> str:
    return R.path(R.TUNED_DIR, "frozen_quick.json" if quick else "frozen.json")


def tuned_dir(quick: bool) -> str:
    return R.path(R.TUNED_DIR, "quick") if quick else R.path(R.TUNED_DIR)


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _tune_jobs(reg, quick, market, label, segment_at=0):
    seeds = reg["seeds"]["tuning_quick" if quick else "tuning"]
    check_tuning_seeds(seeds)
    trials = reg["tuning"]["trials_quick" if quick else "trials"]
    jobs, meta = [], {}
    for name in reg["s2_order"]:
        space = reg["strategies"][name]["space"]
        if not space:
            continue
        base = R.base_spec(reg, name)
        for i in range(trials):
            params = R.sample_trial(space, name, i)
            spec = R.apply_params(base, params)
            h = run_config_hash(spec, market)
            meta[(name, i)] = (params, spec, h)
            for s in seeds:
                jobs.append(
                    Job(
                        "TUNE",
                        f"{name}#{i}",
                        spec,
                        market,
                        s,
                        label=label,
                        segment_at=segment_at,
                        config_hash=h,
                    )
                )
    return jobs, meta, seeds, trials


def tune(reg: dict, quick: bool, workers: int, log=print) -> dict:
    """The protocol tuning run (or its quick version): the S2 strategies, the S2b ladder, and
    the RETUNED bound of S5. Writes the tuned configurations, the tuning log, and trials.csv."""
    label = "TUNE+QUICK" if quick else "TUNE"
    market = reg["markets"]["quick_main" if quick else "main"]
    shift_market = reg["markets"]["quick_shift" if quick else "shift"]
    at_bar = reg["S5"]["quick_at_bar" if quick else "at_bar"]
    jobs, meta, seeds, trials = _tune_jobs(reg, quick, market, label)
    log(f"tuning: {len(jobs)} runs ({trials} trials x {len(seeds)} seeds per strategy)")
    rows = run_parallel(run_job, jobs, workers)
    by = {}
    for j, r in zip(jobs, rows):
        by.setdefault(j.variant, []).append(r)
    tuned, log_rows, trial_rows = {}, [], []
    for name in reg["s2_order"]:
        if not reg["strategies"][name]["space"]:
            continue
        best = None
        for i in range(trials):
            rs = sorted(by[f"{name}#{i}"], key=lambda r: r["seed"])
            score = _mean([criterion(r) for r in rs])
            params, spec, h = meta[(name, i)]
            log_rows.append(
                {
                    "strategy": name,
                    "trial": i,
                    "config_hash": h,
                    "seeds": " ".join(map(str, seeds)),
                    "mean_ce_ann": score,
                    "params": _fmt_params(params),
                }
            )
            trial_rows.append(_trial_row("S2-tuning", f"{name}#{i}", h, rs))
            if best is None or score > best[0]:
                best = (score, i, params, spec)
        tuned[name] = {
            "strategy": name,
            "trial": best[1],
            "mean_ce_ann": best[0],
            "params": best[2],
            "spec": best[3],
            "seeds": seeds,
            "trials": trials,
        }
    # ladder: gamma of MV re-tuned per rung on the declared grid
    lad = reg["ladder"]
    mv = tuned[lad["base"]]["spec"]
    ljobs = []
    for rung in lad["rungs"]:
        for ci in [False, True] if lad["cost_ignorant"] else [False]:
            for gi, gamma in enumerate(lad["gamma_grid"]):
                spec = ladder_spec(mv, rung, ci, gamma)
                h = run_config_hash(spec, market)
                for s in seeds:
                    ljobs.append(
                        Job(
                            "TUNE_LADDER",
                            f"{rung['id']}|{int(ci)}|{gi}",
                            spec,
                            market,
                            s,
                            label=label + ("+ORACLE" if R.is_oracle(spec) else ""),
                            config_hash=h,
                        )
                    )
    log(f"ladder tuning: {len(ljobs)} runs")
    lrows = run_parallel(run_job, ljobs, workers)
    lby = {}
    for j, r in zip(ljobs, lrows):
        lby.setdefault(j.variant, []).append(r)
    ladder = {}
    for rung in lad["rungs"]:
        for ci in [False, True] if lad["cost_ignorant"] else [False]:
            best = None
            for gi, gamma in enumerate(lad["gamma_grid"]):
                rs = lby[f"{rung['id']}|{int(ci)}|{gi}"]
                score = _mean([criterion(r) for r in rs])
                log_rows.append(
                    {
                        "strategy": f"LADDER_{rung['id']}{'_NC' if ci else ''}",
                        "trial": gi,
                        "config_hash": rs[0]["config_hash"],
                        "seeds": " ".join(map(str, seeds)),
                        "mean_ce_ann": score,
                        "params": f"gamma={gamma!r}",
                    }
                )
                if best is None or score > best[0]:
                    best = (score, gamma)
            ladder[f"{rung['id']}|{int(ci)}"] = {
                "gamma": best[1],
                "mean_ce_ann": best[0],
                "spec": ladder_spec(mv, rung, ci, best[1]),
            }
    # RETUNED bound of S5: the same search on shift markets, scored on the bars from at_bar on
    retuned = {}
    if reg["S5"].get("retuned", False):
        rjobs, rmeta, _, _ = _tune_jobs(
            reg, quick, shift_market, label + "+RETUNED", segment_at=at_bar
        )
        rjobs = [
            Job(
                "TUNE_RETUNED",
                j.variant,
                j.spec,
                j.market,
                j.seed,
                label=j.label,
                segment_at=j.segment_at,
                config_hash=j.config_hash,
            )
            for j in rjobs
        ]
        log(f"RETUNED tuning: {len(rjobs)} runs")
        rrows = run_parallel(run_job, rjobs, workers)
        rby = {}
        for j, r in zip(rjobs, rrows):
            rby.setdefault(j.variant, []).append(r)
        for name in reg["s2_order"]:
            if not reg["strategies"][name]["space"]:
                continue
            best = None
            for i in range(trials):
                rs = rby[f"{name}#{i}"]
                score = _mean([criterion_segment(r, "post") for r in rs])
                params, _spec, _h = rmeta[(name, i)]
                log_rows.append(
                    {
                        "strategy": f"RETUNED_{name}",
                        "trial": i,
                        "config_hash": rs[0]["config_hash"],
                        "seeds": " ".join(map(str, seeds)),
                        "mean_ce_ann": score,
                        "params": _fmt_params(params),
                    }
                )
                if best is None or score > best[0]:
                    best = (score, i, params, rmeta[(name, i)][1])
            spec = dict(best[3])
            spec["strategy_id"] = f"{name}_RETUNED"
            retuned[name] = {
                "trial": best[1],
                "mean_ce_ann_post": best[0],
                "params": best[2],
                "spec": spec,
            }
    out = {
        "tuned": tuned,
        "ladder": ladder,
        "retuned": retuned,
        "log": log_rows,
        "trials": trial_rows,
    }
    _write_tuning(out, quick)
    return out


def _fmt_params(p: dict) -> str:
    return ";".join(f"{k}={v!r}" for k, v in sorted(p.items()))


def _trial_row(study, trial_id, h, rows):
    srs = [r["sr_bar"] for r in rows]
    return {
        "study_id": study,
        "trial_id": trial_id,
        "config_hash": h,
        "sr_bar": _mean(srs),
        "t_bars": sum(r["n_bars"] for r in rows),
        "skew": _mean([r["skew"] for r in rows]),
        "kurt": _mean([r["kurt"] for r in rows]),
    }


def ladder_spec(mv_spec: dict, rung: dict, cost_ignorant: bool, gamma: float) -> dict:
    spec = copy.deepcopy(mv_spec)
    spec["forecast"] = rung["forecast"]
    if "rho" in rung:
        spec["rho"] = rung["rho"]
    spec["optimizer"]["gamma"] = gamma
    spec["shrink"] = 1.0  # the oracle providers have no shrink; every rung uses the raw forecast
    if cost_ignorant:
        spec["optimizer"]["cost_scale"] = 0.0
    spec["strategy_id"] = f"MV_{rung['id']}{'_NC' if cost_ignorant else ''}"
    return spec


def _write_tuning(out: dict, quick: bool) -> None:
    from dynamic_trading_engine.experiments.output import write_csv

    d = tuned_dir(quick)
    os.makedirs(d, exist_ok=True)
    for name, t in out["tuned"].items():
        write_json(os.path.join(d, f"{name}.json"), t)
    write_json(os.path.join(d, "ladder.json"), out["ladder"])
    write_json(os.path.join(d, "retuned.json"), out["retuned"])
    write_csv(
        os.path.join(d, "tuning_log.csv"),
        out["log"],
        ["strategy", "trial", "config_hash", "seeds", "mean_ce_ann", "params"],
    )
    write_csv(
        os.path.join(d, "trials.csv"),
        out["trials"],
        ["study_id", "trial_id", "config_hash", "sr_bar", "t_bars", "skew", "kurt"],
    )


def load_tuned(reg: dict, quick: bool) -> dict:
    d = tuned_dir(quick)
    tuned = {}
    for name in reg["s2_order"]:
        if reg["strategies"][name]["space"]:
            tuned[name] = read_json(os.path.join(d, f"{name}.json"))
    return {
        "tuned": tuned,
        "ladder": read_json(os.path.join(d, "ladder.json")),
        "retuned": read_json(os.path.join(d, "retuned.json")),
    }


def frozen_specs(reg: dict, tuned: dict) -> dict[str, dict]:
    """The evaluation's strategy configurations by id (the S2 strategies at their tuned
    parameters; EW at its fixed configuration)."""
    out = {}
    for name in reg["s2_order"]:
        if name in tuned["tuned"]:
            out[name] = tuned["tuned"][name]["spec"]
        else:
            out[name] = R.base_spec(reg, name)
    return out


def write_frozen(entries: list[dict], quick: bool) -> str:
    entries = sorted(entries, key=lambda e: e["id"])
    hashes = sorted({e["hash"] for e in entries})
    write_json(
        frozen_path(quick),
        {"quick": quick, "n_configurations": len(hashes), "hashes": hashes, "entries": entries},
    )
    return frozen_path(quick)


def load_frozen_hashes(quick: bool) -> set[str]:
    p = frozen_path(quick)
    if not os.path.exists(p):
        raise ProtocolError(f"no frozen configurations at {p}: run the tuning and the freeze first")
    return set(read_json(p)["hashes"])


def check_frozen(jobs: list[Job], quick: bool) -> None:
    allowed = load_frozen_hashes(quick)
    for j in jobs:
        if j.config_hash not in allowed:
            raise ProtocolError(
                f"configuration {j.experiment}/{j.variant} ({j.config_hash[:12]}) is not frozen"
            )


def ce_from_moments(mean: float, var: float, ppy: int = 252, gamma_ce: float = 6.0) -> float:
    return ce_ann(mean, var, ppy, gamma_ce)
