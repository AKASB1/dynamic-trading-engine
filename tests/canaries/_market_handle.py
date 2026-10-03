"""A helper that opens the registered market the first time it is used and memoizes it.

Re-importing only the strategy module would leave the first world's handle in place; the audit
purges this module between worlds (it is neither standard library, third party, nor harness)."""

import functools

from dynamic_trading_engine.market import registry


@functools.cache
def handle():
    return registry.open_current()
