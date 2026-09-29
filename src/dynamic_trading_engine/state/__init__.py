"""Portfolio state and market snapshot."""
from dataclasses import dataclass

@dataclass
class PortfolioState:
    cash: float
    holdings: dict[str, float]

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(units * prices[symbol] for symbol, units in self.holdings.items())
