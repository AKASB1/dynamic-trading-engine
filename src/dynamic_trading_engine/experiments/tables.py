"""The table builder: Markdown tables from committed result files.

It refuses QUICK rows (a quick result never enters a results table) and oracle rows without the
ORACLE marker; strategy tables keep the order of the registry, and nothing is ranked except by
the declared criterion.
"""

from __future__ import annotations

import math

from dynamic_trading_engine.experiments.criterion import CRITERION


class TableError(ValueError):
    pass


def check_rows(rows: list[dict]) -> None:
    for r in rows:
        label = str(r.get("label", ""))
        if "QUICK" in label.split("+"):
            raise TableError("a QUICK result cannot enter a results table")
        oracle = str(r.get("oracle", "")).lower() in ("true", "1")
        is_oracle_variant = any(
            x in str(r.get("variant", r.get("strategy", ""))) for x in ("oracle", "noisy")
        )
        name = str(r.get("variant", r.get("strategy", "")))
        if name.endswith("_RETUNED") and "RETUNED" not in label.split("+"):
            raise TableError(f"re-tuned bound without its RETUNED marker: {name}")
        if (oracle or is_oracle_variant) and "ORACLE" not in label.split("+"):
            raise TableError(
                f"oracle row without its ORACLE marker: {r.get('variant', r.get('strategy'))}"
            )


def _f(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def markdown(rows: list[dict], cols: list[tuple[str, str]], nd: int = 4, raw: tuple = ()) -> str:
    """Markdown table; numbers get ``nd`` decimals except in the columns listed in ``raw``."""
    check_rows(rows)
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for k, _h in cols:
            v = r.get(k, "")
            fmt_num = k not in raw and (isinstance(v, (int, float)) or _num(v))
            cells.append(_f(v, nd) if fmt_num else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def _num(v) -> bool:
    try:
        float(v)
        return isinstance(v, str) and v not in ("", "true", "false")
    except (TypeError, ValueError):
        return False


def order_by_registry(rows: list[dict], order: list[str], key: str = "strategy") -> list[dict]:
    pos = {n: i for i, n in enumerate(order)}
    return sorted(rows, key=lambda r: pos.get(r[key], len(pos)))


def best_by_criterion(rows: list[dict]) -> dict:
    """The row with the highest mean of the declared criterion (ties: the first in the given
    order, which is the registry's)."""
    best = None
    for r in rows:
        v = float(r[f"{CRITERION}_mean"])
        if best is None or v > float(best[f"{CRITERION}_mean"]):
            best = r
    return best
