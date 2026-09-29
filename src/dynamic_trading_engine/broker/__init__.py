"""Paper broker boundary; no live connectivity."""
from typing import Protocol
from dynamic_trading_engine.execution import Fill, Order

class Broker(Protocol):
    def submit(self, order: Order) -> Fill: ...
