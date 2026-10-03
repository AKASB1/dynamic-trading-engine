"""Summary statistics across seeds: Student-t intervals, paired differences with win/tie/loss
counts (an exact tie is a tie), Wilson intervals for proportions. The independent unit is the
seed. (Adapted from the owner's rollout-engine, MIT.)"""

from __future__ import annotations

import math

from scipy.stats import norm
from scipy.stats import t as student_t


def mean_ci(xs: list[float], level: float = 0.95) -> tuple[float, float, int]:
    """Mean, half-width of the Student-t interval, and n."""
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, 0
    m = math.fsum(xs) / n
    if n < 2:
        return m, math.nan, n
    sd = math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))
    return m, float(student_t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n), n


def paired(ref: dict[int, float], pol: dict[int, float]) -> dict:
    """Paired differences pol - ref over the common seeds (higher is better): mean, interval,
    wins (pol > ref), ties (exactly equal), losses."""
    seeds = sorted(set(ref) & set(pol))
    diffs = [pol[s] - ref[s] for s in seeds]
    m, ci, n = mean_ci(diffs)
    w = sum(1 for d in diffs if d > 0)
    lo = sum(1 for d in diffs if d < 0)
    return {"diff_mean": m, "diff_ci": ci, "n": n, "wins": w, "ties": n - w - lo, "losses": lo}


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = float(norm.ppf(0.5 + level / 2))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
