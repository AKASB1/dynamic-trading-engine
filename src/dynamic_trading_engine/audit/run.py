"""The ``audit`` command: every guard on the canaries and on every built-in strategy.

Subjects: the four canaries of ``tests/canaries`` (D1, D1b, D2, D3), every strategy of the
experiment grid at its default configuration (with the ladder's oracle rungs), and every forecast
provider, risk model, and optimizer inside a one-component strategy. Market: ``tiny`` (seed 1).
Guards: ``replay`` (orders up to each sampled instant bit-identical in the real and in the
poisoned world), ``garbage_truth`` (pipelines without an oracle: identical orders when the truth
is replaced by garbage), ``label`` (a provider that claims to be honest may not correlate above
0.99 with ``m`` on the audit sample). Cells: caught, passed, or n/a.
"""

from __future__ import annotations

import copy
import importlib
import os
import shutil
import sys
import tempfile

import numpy as np

from dynamic_trading_engine.audit.poison import garbage_truth, poison
from dynamic_trading_engine.audit.replay import forecasts_in_world, replay_world
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import write_market
from dynamic_trading_engine.rng import stream

GUARDS = ("replay", "garbage_truth", "label")
LABEL_THRESHOLD = 0.99
AUDIT_MARKET = "tiny"
# a seed used by nothing else (tests, quick mode, and experiments never write this market), so
# that no committed or generated file outside the audit's own temporary worlds holds its future
AUDIT_SEED = 7919


def builtin_subjects(reg: dict) -> list[tuple[str, dict]]:
    from dynamic_trading_engine.experiments import registry as R

    out = {}
    for name in reg["s2_order"]:
        out[f"grid:{name}"] = R.base_spec(reg, name)
    mv = R.base_spec(reg, "MV")
    for rung in reg["ladder"]["rungs"]:
        if rung["forecast"] == "plain":
            continue
        sp = copy.deepcopy(mv)
        sp["forecast"] = rung["forecast"]
        if "rho" in rung:
            sp["rho"] = rung["rho"]
        sp["strategy_id"] = f"MV_{rung['id']}"
        out[f"grid:MV_{rung['id']}"] = sp
    rank = R.base_spec(reg, "RANK_LS")
    for f in ("none", "plain", "noisy_oracle", "oracle"):
        sp = copy.deepcopy(rank)
        sp["forecast"] = f
        sp["strategy_id"] = f"forecast_{f}"
        out[f"forecast:{f}"] = sp
    mvr = R.base_spec(reg, "MINVAR")
    for r in ("sample", "ewma", "lw", "pca"):
        sp = copy.deepcopy(mvr)
        sp["risk"] = r
        sp["strategy_id"] = f"risk_{r}"
        out[f"risk:{r}"] = sp
    for o in (
        "equal_weight",
        "rank_ls",
        "min_variance",
        "mean_variance",
        "cvar",
        "robust_box",
        "robust_ell",
        "mpc",
    ):
        sp = copy.deepcopy(mv)
        sp["optimizer"]["name"] = o
        if o in ("equal_weight", "min_variance"):
            sp["book"] = "LO"
        sp["strategy_id"] = f"optimizer_{o}"
        out[f"optimizer:{o}"] = sp
    return sorted(out.items())


def canary_subjects(canary_dir: str) -> list[tuple[str, dict, dict]]:
    parent = os.path.dirname(os.path.abspath(canary_dir))
    pkg = os.path.basename(os.path.abspath(canary_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    out = []
    for fn in sorted(os.listdir(canary_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        modname = f"{pkg}.{fn[:-3]}"
        mod = importlib.import_module(modname)
        meta = dict(mod.CANARY)
        out.append(
            (f"canary:{meta['id']}", {"module": modname, "class": "Strategy", "params": {}}, meta)
        )
    return out


def _orders_upto(orders: list, t: int) -> list:
    return [o for o in orders if o[0] <= t]


class Worlds:
    """The real world and the poisoned world of each sampled instant, written once."""

    def __init__(self, base_dir: str, market, truth, seed: int):
        self.base, self.market, self.truth, self.seed = base_dir, market, truth, seed
        self.real_dir = os.path.join(base_dir, "real")
        write_market(market, self.real_dir)
        self._poisoned: dict[int, tuple[str, object]] = {}

    def poisoned(self, k: int):
        if k not in self._poisoned:
            pm, ptr = poison(self.market, self.truth, k, self.seed)
            d = os.path.join(self.base, f"poison_{k}")
            write_market(pm, d)
            self._poisoned[k] = (d, ptr)
        return self._poisoned[k]


def audit_subject(worlds: Worlds, spec, bars: list[int], loop_kw: dict, is_oracle: bool) -> dict:
    m = worlds.market
    real = replay_world(worlds.real_dir, worlds.truth, spec, AUDIT_SEED, max(bars), loop_kw)
    res = {"instants": len(bars)}
    caught_at = None
    for idx, k in enumerate(bars):
        d, ptr = worlds.poisoned(k)
        got = replay_world(d, ptr, spec, AUDIT_SEED, k, loop_kw)
        t = int(m.ts_event[k])
        if got != _orders_upto(real, t):
            caught_at = idx + 1
            break
    res["replay"] = "caught" if caught_at else "passed"
    res["replay_caught_after"] = caught_at or ""
    if is_oracle:
        res["garbage_truth"] = "n/a"
    else:
        g = replay_world(
            worlds.real_dir, garbage_truth(worlds.truth, 99), spec, AUDIT_SEED, max(bars), loop_kw
        )
        res["garbage_truth"] = "caught" if g != real else "passed"
    fcs, claims_oracle = forecasts_in_world(
        worlds.real_dir, worlds.truth, spec, AUDIT_SEED, bars, loop_kw
    )
    if fcs is None:
        res["label"] = "n/a"
        res["label_corr"] = ""
    else:
        col = {iid: j for j, iid in enumerate(worlds.truth.ids)}
        num = a = b = 0.0
        for k, ids, mu in fcs:
            mm = worlds.truth.m[k, [col[i] for i in ids]]
            num += float(mu @ mm)
            a += float(mu @ mu)
            b += float(mm @ mm)
        corr = num / np.sqrt(a * b) if a > 0 and b > 0 else 0.0
        res["label_corr"] = round(float(corr), 6)
        flagged = corr > LABEL_THRESHOLD and not claims_oracle
        res["label"] = "caught" if flagged else ("n/a" if claims_oracle else "passed")
    return res


def sample_bars(market, warmup: int, every: int, n: int, seed: int) -> list[int]:
    dec = list(range(warmup, market.n_bars - 1, every))
    g = stream(seed, "audit.instants")
    pick = g.choice(len(dec), size=min(n, len(dec)), replace=False)
    return sorted(dec[i] for i in pick)


def run_audit(
    canary_dir: str,
    n_instants: int = 30,
    n_instants_builtin: int | None = None,
    subjects: list[str] | None = None,
    log=print,
) -> list[dict]:
    from dynamic_trading_engine.experiments.registry import is_oracle, load_registry

    reg = load_registry()
    cfg = load_market_config(AUDIT_MARKET)
    market, truth = generate(cfg, AUDIT_SEED)
    loop_kw = {"warmup_bars": cfg.warmup_bars, "rebalance_every": reg["loop"]["rebalance_every"]}
    bars_c = sample_bars(
        market, cfg.warmup_bars, loop_kw["rebalance_every"], n_instants, AUDIT_SEED
    )
    bars_b = (
        bars_c
        if n_instants_builtin is None
        else sample_bars(
            market, cfg.warmup_bars, loop_kw["rebalance_every"], n_instants_builtin, AUDIT_SEED
        )
    )
    rows = []
    tmp = tempfile.mkdtemp(prefix="dte-audit-")
    try:
        worlds = Worlds(tmp, market, truth, AUDIT_SEED)
        items = [(n, s, m) for n, s, m in canary_subjects(canary_dir)] + [
            (n, s, None) for n, s in builtin_subjects(reg)
        ]
        for name, spec, meta in items:
            if subjects and name not in subjects:
                continue
            bars = bars_c if meta is not None else bars_b
            orc = False if meta is not None else is_oracle(spec)
            r = audit_subject(worlds, spec, bars, loop_kw, orc)
            row = {
                "subject": name,
                "oracle": orc,
                "expected_guard": meta["guard"] if meta else "",
                **r,
            }
            caught = [g for g in GUARDS if r[g] == "caught"]
            row["alarm"] = bool(caught)
            rows.append(row)
            log(
                f"{name:28s} replay={r['replay']:7s} garbage={r['garbage_truth']:7s} "
                f"label={r['label']}"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rows
