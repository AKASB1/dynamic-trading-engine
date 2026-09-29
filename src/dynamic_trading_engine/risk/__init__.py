"""Simple gross exposure calculation."""
def gross_exposure(weights: dict[str, float]) -> float:
    return sum(abs(weight) for weight in weights.values())
