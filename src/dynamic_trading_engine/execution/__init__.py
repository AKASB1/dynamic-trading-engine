"""Market order and simple fill model."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Order:
    symbol: str
    units: float

@dataclass(frozen=True)
class Fill:
    symbol: str
    units: float
    price: float

def fill_market(order: Order, mid_price: float, spread_rate: float = 0.0) -> Fill:
    if mid_price <= 0 or spread_rate < 0:
        raise ValueError("invalid price or spread")
    side = 1 if order.units > 0 else -1
    return Fill(order.symbol, order.units, mid_price * (1 + side * spread_rate / 2))
