"""The replay audit: a strategy replayed from its first decision up to and including an instant t
in the real and in a poisoned world must submit bit-identical orders.

Each world is written to its own directory of contract files, the market registry points at it,
and the world is the only market reachable. Before each world the harness purges from
``sys.modules`` every module that is not the standard library, a third-party package, or part of
the audit harness (the strategy modules, every helper module that may memoize a handle, and the
decision pipeline of this package), and re-imports the factory and the loop, so that nothing a
strategy computed at import time, in a constructor, or in a memoized helper carries over from one
world to the next. The strategy is instantiated again from its spec in every world.
"""

from __future__ import annotations

import importlib
import os
import sys
import sysconfig

HARNESS_PREFIXES = (
    "dynamic_trading_engine.audit",
    "dynamic_trading_engine.market",
    "dynamic_trading_engine.contracts",
    "dynamic_trading_engine.rng",
)
_PURELIB = os.path.normcase(os.path.abspath(sysconfig.get_paths()["purelib"]))
_STDLIB = os.path.normcase(os.path.abspath(sysconfig.get_paths()["stdlib"]))


def _is_kept(name: str, mod) -> bool:
    if name in ("__main__", "__mp_main__"):
        return True
    top = name.split(".")[0]
    if top in sys.stdlib_module_names or top in sys.builtin_module_names:
        return True
    if name == "dynamic_trading_engine" or name.startswith(HARNESS_PREFIXES):
        return True
    f = getattr(mod, "__file__", None)
    if f is None:
        return not name.startswith("dynamic_trading_engine")
    f = os.path.normcase(os.path.abspath(f))
    if f.startswith(_PURELIB) or f.startswith(_STDLIB) or "site-packages" in f:
        # third-party (an editable install of this package is not under site-packages)
        return not name.startswith("dynamic_trading_engine")
    return False


def purge_modules() -> list[str]:
    """Remove every non-standard, non-third-party, non-harness module; returns their names."""
    gone = [n for n, m in list(sys.modules.items()) if m is not None and not _is_kept(n, m)]
    for n in sorted(gone):
        del sys.modules[n]
    return gone


def replay_world(world_dir: str, truth, spec, seed: int, stop_bar: int, loop_kw: dict) -> list:
    """Replay ``spec`` in the world stored at ``world_dir`` up to bar ``stop_bar``; returns the
    orders of every decision (instant, ((instrument_id, quantity), ...))."""
    from dynamic_trading_engine.market import registry
    from dynamic_trading_engine.market.store import read_market

    purge_modules()
    registry.set_current_dir(world_dir)
    try:
        market = read_market(world_dir)
        factory = importlib.import_module("dynamic_trading_engine.strategies.factory")
        loop = importlib.import_module("dynamic_trading_engine.engine.loop")
        costs_mod = importlib.import_module("dynamic_trading_engine.contracts.costs")
        costs = costs_mod.CostConfig()
        cfg = loop.LoopConfig(**loop_kw)
        strategy = factory.build_strategy(spec, costs, cfg.rebalance_every, truth=truth, seed=seed)
        res = loop.run_loop(market, strategy, costs, cfg, stop_after_bar=stop_bar)
        return res.orders_by_decision
    finally:
        registry.set_current_dir(None)


def forecasts_in_world(world_dir: str, truth, spec, seed: int, bars: list[int], loop_kw: dict):
    """The forecast vectors of the strategy's provider at the given decision bars (label audit);
    None when the strategy has no provider."""
    from dynamic_trading_engine.market import registry
    from dynamic_trading_engine.market.store import read_market

    purge_modules()
    registry.set_current_dir(world_dir)
    try:
        market = read_market(world_dir)
        factory = importlib.import_module("dynamic_trading_engine.strategies.factory")
        view = importlib.import_module("dynamic_trading_engine.state.view")
        loop = importlib.import_module("dynamic_trading_engine.engine.loop")
        costs = importlib.import_module("dynamic_trading_engine.contracts.costs").CostConfig()
        cfg = loop.LoopConfig(**loop_kw)
        strategy = factory.build_strategy(spec, costs, cfg.rebalance_every, truth=truth, seed=seed)
        provider = getattr(strategy, "provider", None)
        if provider is None:
            return None, None
        out = []
        for k in bars:
            import numpy as np

            st = view.build_state(market, int(market.ts_event[k]), 1e6, np.zeros(market.n))
            out.append((k, st.ids, provider.forecast(st).mu))
        return out, bool(getattr(strategy, "oracle", False))
    finally:
        registry.set_current_dir(None)
