"""Run logs in the contract's formats (orders, fills, positions, equity) plus ``decisions.csv``."""

from __future__ import annotations

import os

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import write_dataset, write_json
from dynamic_trading_engine.contracts.schemas import F, Schema

DECISIONS = Schema(
    "decisions",
    (
        F("ts_decision", "ts"),
        F("strategy_id", "token"),
        F("forecast_id", "token"),
        F("oracle", "bool"),
        F("risk_model", "token"),
        F("solver_status", "token"),
        F("objective", "float_opt"),
        F("n_trades", "int"),
        F("turnover", "float"),
        F("wall_solve_ms", "float"),
        F("hold", "bool"),
    ),
    key=("ts_decision",),
)

LOG_FILES = (
    ("orders_v1", S.ORDERS, "orders"),
    ("fills_v1", S.FILLS, "fills"),
    ("positions_v1", S.POSITIONS, "positions"),
    ("equity_v1", S.EQUITY, "equity"),
    ("decisions", DECISIONS, "decisions"),
)


def write_run_logs(logs: dict, directory: str, run_meta: dict, generator: dict, seed) -> None:
    """Write the five logs (each with its manifest) and ``run.json`` (what the validator needs:
    the cost configuration, the initial cash, the market directory, the strategy and its labels)."""
    os.makedirs(directory, exist_ok=True)
    for name, schema, key in LOG_FILES:
        rows = logs[key]
        if key == "fills":
            rows = sorted(rows, key=lambda r: (r[2], r[0]))
        data = S.dump_bytes(schema, rows)
        write_dataset(os.path.join(directory, name + ".csv"), data, len(rows), generator, seed)
    write_json(os.path.join(directory, "run.json"), run_meta)
