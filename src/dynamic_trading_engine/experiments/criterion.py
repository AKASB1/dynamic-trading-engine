"""The one declared selection criterion: ``ce_ann`` (QC metrics, gamma_ce = 6).

The tuner and the table builder select and order strategies only through ``criterion`` (a
static test checks it). Every other metric is reported, never used to select or rank.
"""

from __future__ import annotations

from dynamic_trading_engine.analytics.metrics import GAMMA_CE, ce_ann

CRITERION = "ce_ann"


def criterion(row: dict) -> float:
    return float(row["ce_ann"])


def criterion_segment(row: dict, which: str, ppy: int = 252, gamma_ce: float = GAMMA_CE) -> float:
    """``ce_ann`` of one segment of a run (``pre`` or ``post`` of a shift bar)."""
    return ce_ann(float(row[f"{which}_mean"]), float(row[f"{which}_var"]), ppy, gamma_ce)
