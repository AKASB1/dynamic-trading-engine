"""The strategy factory: builds a strategy from a spec, handing the truth to oracle providers.

A spec is either a ``StrategySpec`` (built-in pipeline) or a dict ``{"module", "class",
"params"}`` (any class, for example a canary of the tests); a class whose constructor accepts
``truth`` gets the truth of the world it runs in. This module imports the truth module and is
on the list of modules allowed to (static import test).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Callable

from dynamic_trading_engine.contracts.costs import CostConfig
from dynamic_trading_engine.forecasts.oracle import NoisyOracleForecast, OracleForecast
from dynamic_trading_engine.market.truth import Truth
from dynamic_trading_engine.optimization.optimizers import OptimizerConfig
from dynamic_trading_engine.rng import stream
from dynamic_trading_engine.strategies.pipeline import PipelineStrategy, StrategySpec


def pipeline_seed(seed: int) -> int:
    """The seed a strategy receives: derived from the run seed (stream ``pipeline.seed``), never
    the market seed itself, so that a strategy cannot regenerate the market it trades on from
    it. The same in the real and in the poisoned world of the audit."""
    return int(stream(seed, "pipeline.seed").integers(0, 2**62))


def spec_from_dict(d: dict) -> StrategySpec:
    d = dict(d)
    if "optimizer" in d and isinstance(d["optimizer"], dict):
        o = dict(d["optimizer"])
        if o.get("beta_band") is not None:
            o["beta_band"] = tuple(o["beta_band"])
        d["optimizer"] = OptimizerConfig(**o)
    return StrategySpec(**d)


def spec_to_dict(spec: StrategySpec) -> dict:
    return dataclasses.asdict(spec)


def build_strategy(
    spec,
    costs: CostConfig,
    k: int,
    truth: Truth | None = None,
    seed: int = 0,
    clock: Callable[[], float] | None = None,
):
    seed = pipeline_seed(seed)
    if isinstance(spec, dict) and "module" in spec:
        mod = importlib.import_module(spec["module"])
        cls = getattr(mod, spec["class"])
        params = dict(spec.get("params", {}))
        sig = inspect.signature(cls)
        if "truth" in sig.parameters:
            params["truth"] = truth
        if "seed" in sig.parameters:
            params["seed"] = seed
        if "costs" in sig.parameters:
            params["costs"] = costs
        if "k" in sig.parameters:
            params["k"] = k
        return cls(**params)
    if isinstance(spec, dict):
        spec = spec_from_dict(spec)
    provider = None
    if spec.forecast == "oracle":
        provider = OracleForecast(truth)
    elif spec.forecast == "noisy_oracle":
        provider = NoisyOracleForecast(truth, spec.rho, seed)
    return PipelineStrategy(spec, costs, k, provider, clock)
