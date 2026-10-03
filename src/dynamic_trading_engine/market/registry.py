"""The market registry: the directory of contract files that is "the market" of this process.

Set through the environment variable ``DTE_MARKET_DIR`` (read on every call, never cached).
The decision pipeline never uses it; the audit points it at each world's own directory, so a
strategy that opens the market by path or through a handle reads only that world.
"""

from __future__ import annotations

import os

ENV = "DTE_MARKET_DIR"


def current_dir() -> str | None:
    return os.environ.get(ENV)


def set_current_dir(path: str | None) -> None:
    if path is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = path


def open_current():
    from dynamic_trading_engine.market.store import read_market

    d = current_dir()
    if not d:
        raise RuntimeError("no market registered (DTE_MARKET_DIR is not set)")
    return read_market(d)
