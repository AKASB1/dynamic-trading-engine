"""Turnover helper."""
def turnover(traded_notional: float, equity: float) -> float:
    if traded_notional < 0 or equity <= 0:
        raise ValueError("invalid turnover inputs")
    return traded_notional / equity
