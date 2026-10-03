"""The committed registry of the protocol (experiments/configs/registry.json): strategies with
their fixed parts and search spaces, trial counts, seeds, and the experiment grids.

A strategy configuration is a ``StrategySpec`` (as a plain dict); its configuration hash is the
contract's canonical hash of the spec dataclass (fields equal to their defaults omitted).
"""

from __future__ import annotations

import copy
import json
import math
import os

from dynamic_trading_engine.contracts.canonical import config_hash
from dynamic_trading_engine.market.config import repo_root
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.strategies.factory import spec_from_dict

REGISTRY_PATH = os.path.join("experiments", "configs", "registry.json")
TUNED_DIR = os.path.join("experiments", "tuned")
RESULTS_DIR = os.path.join("experiments", "results")
OUTPUTS_DIR = os.path.join("experiments", "outputs")


def path(*parts: str) -> str:
    return os.path.join(repo_root(), *parts)


def load_registry() -> dict:
    with open(path(REGISTRY_PATH), encoding="utf-8") as fh:
        return json.load(fh)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def base_spec(reg: dict, name: str) -> dict:
    """The default configuration of a registered strategy (registry defaults + fixed parts)."""
    d = reg["defaults"]
    spec = _merge({**d["spec"], "optimizer": dict(d["optimizer"])}, reg["strategies"][name]["spec"])
    spec["strategy_id"] = name
    return spec


def apply_params(spec: dict, params: dict) -> dict:
    out = copy.deepcopy(spec)
    for key, v in sorted(params.items()):
        if key.startswith("optimizer."):
            out["optimizer"][key.split(".", 1)[1]] = v
        else:
            out[key] = v
    return out


def _draw(rng, kind: str, args: list):
    if kind == "log":
        lo, hi = args
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if kind == "int_log":
        lo, hi = args
        return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
    if kind == "uniform":
        lo, hi = args
        return float(rng.uniform(lo, hi))
    if kind == "choice":
        vals = args[0]
        return vals[int(rng.integers(0, len(vals)))]
    raise ValueError(kind)


def sample_trial(space: dict, strategy: str, trial: int) -> dict:
    """Trial ``i`` of a strategy: a function of the strategy name and the trial index only
    (stream ``tune.<strategy>`` with seed i); parameters drawn in sorted name order."""
    rng = stream(trial, f"tune.{strategy}")
    return {k: _draw(rng, space[k][0], space[k][1:]) for k in sorted(space)}


def spec_hash(spec: dict) -> str:
    return config_hash(spec_from_dict(spec))


def is_oracle(spec: dict) -> bool:
    return spec.get("forecast") in ("oracle", "noisy_oracle")
