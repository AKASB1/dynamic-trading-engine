"""Target-to-order baseline."""
from dynamic_trading_engine.execution import Order
from dynamic_trading_engine.state import PortfolioState

def target_orders(state: PortfolioState, prices: dict[str, float], weights: dict[str, float]) -> list[Order]:
    if any(price <= 0 for price in prices.values()):
        raise ValueError("prices must be positive")
    equity = state.equity(prices)
    return [Order(symbol, equity * weight / prices[symbol] - state.holdings.get(symbol, 0.0))
            for symbol, weight in sorted(weights.items())]
