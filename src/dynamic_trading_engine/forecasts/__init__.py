"""Forecast adapter boundary."""
from typing import Protocol

class ForecastProvider(Protocol):
    def expected_returns(self, symbols: list[str]) -> dict[str, float]: ...
