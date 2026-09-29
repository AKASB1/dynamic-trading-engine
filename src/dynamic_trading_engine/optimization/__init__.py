"""Deterministic equal-weight baseline; convex optimizer is planned."""
def equal_weight(symbols: list[str]) -> dict[str, float]:
    unique = sorted(set(symbols))
    return {symbol: 1 / len(unique) for symbol in unique} if unique else {}
