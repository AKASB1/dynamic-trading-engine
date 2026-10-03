"""The twelve CSV schemas of the shared contract (version 1): loaders, validators, writers.

A loader rejects a malformed file with a :class:`ContractError` that names the 1-based line
number (the header is line 1). Writers produce canonical bytes: UTF-8, LF, the exact header,
shortest round-trip floats, integral quantity columns without a decimal point, empty fields
for missing values. Reading a canonical file and writing it again gives identical bytes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from dynamic_trading_engine.contracts.timeutil import format_ts, parse_ts

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CURRENCY_RE = re.compile(r"^[A-Z]+$")
FLOAT_RE = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")
INT_RE = re.compile(r"^-?\d+$")
TOKEN_RE = re.compile(r"^[^,\s\"]+$")

IDENTITY_TOL = 1e-9


class ContractError(ValueError):
    """A schema violation; ``line`` is the 1-based line number (0 for the whole file)."""

    def __init__(self, schema: str, line: int, message: str):
        self.schema = schema
        self.line = line
        super().__init__(f"{schema}: line {line}: {message}")


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # id, text, text_opt, token, enum, currency, ts, ts_opt, float, float_opt, int, bool
    enum: tuple[str, ...] = ()
    integral: bool = False  # written without a decimal point when the value is integral
    check: Callable[[object], bool] | None = None
    check_msg: str = ""


@dataclass(frozen=True)
class Schema:
    name: str
    fields: tuple[Field, ...]
    key: tuple[str, ...]
    row_rule: Callable[[dict], str | None] | None = None
    table_rule: Callable[[list[dict]], tuple[int, str] | None] | None = None
    unique_cols: tuple[str, ...] = field(default=())

    @property
    def header(self) -> str:
        return ",".join(f.name for f in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass
class Table:
    schema: Schema
    rows: list[tuple]

    def column(self, name: str) -> list:
        i = self.schema.index(name)
        return [r[i] for r in self.rows]

    def dicts(self) -> list[dict]:
        names = self.schema.names
        return [dict(zip(names, r)) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


def _pos(v) -> bool:
    return v > 0


def _nonneg(v) -> bool:
    return v >= 0


def _ge_m1(v) -> bool:
    return v >= -1


def _nonzero(v) -> bool:
    return v != 0


def F(name, kind, check=None, msg="", integral=False, enum=()):  # noqa: N802 - short builder
    return Field(name, kind, tuple(enum), integral, check, msg)


# ---------------------------------------------------------------- row rules


def _instr_rule(r: dict) -> str | None:
    if (r["ts_delist"] is None) != (r["delist_return"] is None):
        return "delist_return must be empty exactly when ts_delist is empty"
    if r["ts_delist"] is not None and r["ts_delist"] < r["ts_list"]:
        return "ts_delist before ts_list (a delisting before the listing)"
    return None


def _bar_rule(r: dict) -> str | None:
    if not r["ts_open"] < r["ts_event"]:
        return "ts_open must be before ts_event"
    if r["ts_avail"] < r["ts_event"]:
        return "ts_avail before ts_event"
    o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
    if h < max(o, c, lo):
        return "high below max(open, close, low)"
    if lo > min(o, c, h):
        return "low above min(open, close, high)"
    return None


def _ca_rule(r: dict) -> str | None:
    if r["ts_avail"] > r["ts_ex"]:
        return "ts_avail after ts_ex"
    return None


def _avail_rule(r: dict) -> str | None:
    if r["ts_avail"] < r["ts_event"]:
        return "ts_avail before ts_event"
    return None


def _order_rule(r: dict) -> str | None:
    if r["order_type"] == "market" and r["limit_price"] is not None:
        return "limit_price must be empty for a market order"
    if r["order_type"] == "limit" and r["limit_price"] is None:
        return "limit_price required for a limit order"
    return None


def _position_rule(r: dict) -> str | None:
    v = r["quantity"] * r["mark_price"]
    if abs(v - r["value"]) > IDENTITY_TOL * max(1.0, abs(v)):
        return "value != quantity * mark_price"
    return None


EQUITY_FLOWS = (
    "hold_pnl",
    "trade_pnl",
    "spread_cost",
    "impact_cost",
    "commission",
    "borrow",
    "financing",
    "income",
)


def _equity_rule(r: dict) -> str | None:
    e = r["cash"] + r["position_value"]
    if abs(e - r["equity"]) > IDENTITY_TOL * max(1.0, abs(r["equity"])):
        return "equity != cash + position_value"
    return None


def equity_change(r: dict) -> float:
    return (
        r["hold_pnl"]
        + r["trade_pnl"]
        - r["spread_cost"]
        - r["impact_cost"]
        - r["commission"]
        - r["borrow"]
        - r["financing"]
        + r["income"]
    )


# ---------------------------------------------------------------- table rules


def _bars_table(rows: list[dict]) -> tuple[int, str] | None:
    prev = None
    for i, r in enumerate(rows):
        if prev is not None and prev["instrument_id"] == r["instrument_id"]:
            if r["ts_open"] < prev["ts_event"]:
                return i, "bars of one instrument overlap"
        prev = r
    return None


def _series_table(rows: list[dict]) -> tuple[int, str] | None:
    prev = None
    for i, r in enumerate(rows):
        same = prev is not None and (prev["series_id"], prev["ts_event"]) == (
            r["series_id"],
            r["ts_event"],
        )
        if same:
            if r["vintage"] != prev["vintage"] + 1:
                return i, "vintage must grow by one within a period"
            if r["ts_avail"] <= prev["ts_avail"]:
                return i, "ts_avail must increase strictly with the vintage"
        elif r["vintage"] != 0:
            return i, "the first vintage of a period must be 0"
        prev = r
    return None


def _equity_table(rows: list[dict]) -> tuple[int, str] | None:
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        d = b["equity"] - a["equity"]
        if abs(d - equity_change(b)) > IDENTITY_TOL * max(1.0, abs(a["equity"])):
            return i, "accounting identity violated (QC 5)"
    return None


# ---------------------------------------------------------------- the twelve schemas

INSTRUMENTS = Schema(
    "instruments_v1",
    (
        F("instrument_id", "id"),
        F("symbol", "text"),
        F("asset_class", "enum", enum=("equity", "crypto", "index", "other")),
        F("currency", "currency"),
        F("ts_list", "ts"),
        F("ts_delist", "ts_opt"),
        F("delist_return", "float_opt", _ge_m1, ">= -1"),
        F("lot_size", "float", _nonneg, ">= 0", integral=True),
        F("tick_size", "float", _nonneg, ">= 0"),
        F("sector", "text_opt"),
    ),
    key=("instrument_id",),
    row_rule=_instr_rule,
)

BARS = Schema(
    "bars_v1",
    (
        F("instrument_id", "id"),
        F("ts_open", "ts"),
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("open", "float", _pos, "> 0 (a non-positive price)"),
        F("high", "float", _pos, "> 0 (a non-positive price)"),
        F("low", "float", _pos, "> 0 (a non-positive price)"),
        F("close", "float", _pos, "> 0 (a non-positive price)"),
        F("volume", "float", _nonneg, ">= 0", integral=True),
    ),
    key=("instrument_id", "ts_event"),
    row_rule=_bar_rule,
    table_rule=_bars_table,
)

CORPORATE_ACTIONS = Schema(
    "corporate_actions_v1",
    (
        F("instrument_id", "id"),
        F("action", "enum", enum=("split", "cash_dividend")),
        F("ts_ex", "ts"),
        F("ts_avail", "ts"),
        F("value", "float", _pos, "> 0"),
    ),
    key=("instrument_id", "ts_ex", "action"),
    row_rule=_ca_rule,
)

SERIES = Schema(
    "series_v1",
    (
        F("series_id", "token"),
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("vintage", "int", _nonneg, ">= 0"),
        F("value", "float"),
    ),
    key=("series_id", "ts_event", "vintage"),
    row_rule=_avail_rule,
    table_rule=_series_table,
)

RETURNS = Schema(
    "returns_v1",
    (
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("instrument_id", "id"),
        F("ret", "float", _ge_m1, ">= -1"),
    ),
    key=("ts_event", "instrument_id"),
    row_rule=_avail_rule,
)

SIGNALS = Schema(
    "signals_v1",
    (
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("instrument_id", "id"),
        F("name", "token"),
        F("value", "float"),
    ),
    key=("ts_event", "name", "instrument_id"),
    row_rule=_avail_rule,
)

FORECASTS = Schema(
    "forecasts_v1",
    (
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("instrument_id", "id"),
        F("horizon_bars", "int", lambda v: v >= 1, ">= 1"),
        F("mu", "float"),
        F("sigma", "float_opt", _nonneg, ">= 0"),
    ),
    key=("ts_event", "instrument_id", "horizon_bars"),
    row_rule=_avail_rule,
)

LIQUIDITY = Schema(
    "liquidity_v1",
    (
        F("ts_event", "ts"),
        F("ts_avail", "ts"),
        F("instrument_id", "id"),
        F("adv_shares", "float", _pos, "> 0", integral=True),
        F("sigma_bar", "float", _nonneg, ">= 0"),
    ),
    key=("ts_event", "instrument_id"),
    row_rule=_avail_rule,
)

ORDERS = Schema(
    "orders_v1",
    (
        F("order_id", "token"),
        F("ts_submit", "ts"),
        F("instrument_id", "id"),
        F("quantity", "float", _nonzero, "!= 0", integral=True),
        F("order_type", "enum", enum=("market", "limit")),
        F("limit_price", "float_opt", _pos, "> 0"),
        F("tif", "enum", enum=("day", "gtc")),
        F("strategy_id", "token"),
    ),
    key=("ts_submit", "order_id"),
    row_rule=_order_rule,
    unique_cols=("order_id",),
)

FILLS = Schema(
    "fills_v1",
    (
        F("fill_id", "token"),
        F("order_id", "token"),
        F("ts_fill", "ts"),
        F("instrument_id", "id"),
        F("quantity", "float", _nonzero, "!= 0", integral=True),
        F("ref_price", "float", _pos, "> 0"),
        F("price", "float", _pos, "> 0"),
        F("spread_cost", "float", _nonneg, ">= 0"),
        F("impact_cost", "float", _nonneg, ">= 0"),
        F("commission", "float", _nonneg, ">= 0"),
    ),
    key=("ts_fill", "fill_id"),
    unique_cols=("fill_id",),
)

POSITIONS = Schema(
    "positions_v1",
    (
        F("ts_event", "ts"),
        F("instrument_id", "id"),
        F("quantity", "float", _nonzero, "!= 0", integral=True),
        F("mark_price", "float", _pos, "> 0"),
        F("value", "float"),
    ),
    key=("ts_event", "instrument_id"),
    row_rule=_position_rule,
)

EQUITY = Schema(
    "equity_v1",
    (
        F("ts_event", "ts"),
        F("cash", "float"),
        F("position_value", "float"),
        F("equity", "float"),
        F("hold_pnl", "float"),
        F("trade_pnl", "float"),
        F("spread_cost", "float", _nonneg, ">= 0"),
        F("impact_cost", "float", _nonneg, ">= 0"),
        F("commission", "float", _nonneg, ">= 0"),
        F("borrow", "float", _nonneg, ">= 0"),
        F("financing", "float", _nonneg, ">= 0"),
        F("income", "float"),
    ),
    key=("ts_event",),
    row_rule=_equity_rule,
    table_rule=_equity_table,
)

SCHEMAS: dict[str, Schema] = {
    s.name: s
    for s in (
        INSTRUMENTS,
        BARS,
        CORPORATE_ACTIONS,
        SERIES,
        RETURNS,
        SIGNALS,
        FORECASTS,
        LIQUIDITY,
        ORDERS,
        FILLS,
        POSITIONS,
        EQUITY,
    )
}


# ---------------------------------------------------------------- parsing and formatting


def _parse_field(f: Field, text: str):
    k = f.kind
    if k in ("ts_opt", "float_opt", "text_opt") and text == "":
        return None
    if text == "":
        raise ValueError("empty required field")
    if k == "id":
        if not ID_RE.match(text):
            raise ValueError(f"bad identifier {text!r}")
        return text
    if k == "token":
        if not TOKEN_RE.match(text):
            raise ValueError(f"bad token {text!r}")
        return text
    if k in ("text", "text_opt"):
        if '"' in text or text != text.strip():
            raise ValueError(f"bad text {text!r}")
        return text
    if k == "enum":
        if text not in f.enum:
            raise ValueError(f"unknown value {text!r}")
        return text
    if k == "currency":
        if not CURRENCY_RE.match(text):
            raise ValueError(f"bad currency {text!r}")
        return text
    if k in ("ts", "ts_opt"):
        return parse_ts(text)
    if k in ("float", "float_opt"):
        if not FLOAT_RE.match(text):
            raise ValueError(f"unparsable number {text!r}")
        v = float(text)
        if not math.isfinite(v):
            raise ValueError(f"non-finite number {text!r}")
        if v == 0.0 and text.startswith("-"):
            raise ValueError("-0.0 is not allowed")
        return v
    if k == "int":
        if not INT_RE.match(text):
            raise ValueError(f"unparsable integer {text!r}")
        return int(text)
    if k == "bool":
        if text not in ("true", "false"):
            raise ValueError(f"bad boolean {text!r}")
        return text == "true"
    raise AssertionError(k)


def format_float(v: float, integral: bool = False) -> str:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("NaN and inf never appear in files")
    if v == 0.0:
        v = 0.0  # never -0.0
    if integral and v.is_integer():
        return str(int(v))
    return repr(v)


def _format_field(f: Field, v) -> str:
    if v is None:
        if f.kind not in ("ts_opt", "float_opt", "text_opt"):
            raise ValueError(f"{f.name}: required value missing")
        return ""
    k = f.kind
    if k in ("ts", "ts_opt"):
        return format_ts(int(v))
    if k in ("float", "float_opt"):
        return format_float(v, f.integral)
    if k == "int":
        return str(int(v))
    if k == "bool":
        return "true" if v else "false"
    s = str(v)
    if "," in s or "\n" in s or '"' in s:
        raise ValueError(f"{f.name}: field contains a comma, quote, or newline")
    return s


def load_bytes(schema: Schema, data: bytes) -> Table:
    name = schema.name
    if len(data) == 0:
        raise ContractError(name, 0, "zero-byte file")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ContractError(name, 1, "byte order mark")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ContractError(name, 0, "not UTF-8") from e
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]
    if not lines:
        raise ContractError(name, 1, "missing header")
    if lines[0] != schema.header:
        raise ContractError(name, 1, f"header must be {schema.header!r}")
    fields = schema.fields
    nf = len(fields)
    rows: list[tuple] = []
    dict_rows: list[dict] = []
    names = schema.names
    key_idx = [schema.index(k) for k in schema.key]
    seen: dict[str, set] = {c: set() for c in schema.unique_cols}
    prev_key = None
    for ln_no, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        if len(parts) != nf:
            raise ContractError(name, ln_no, f"expected {nf} fields, found {len(parts)}")
        vals = []
        for f, p in zip(fields, parts):
            try:
                v = _parse_field(f, p)
            except ValueError as e:
                raise ContractError(name, ln_no, f"{f.name}: {e}") from None
            if v is not None and f.check is not None and not f.check(v):
                raise ContractError(name, ln_no, f"{f.name}: must be {f.check_msg}")
            vals.append(v)
        row = tuple(vals)
        d = dict(zip(names, row))
        if schema.row_rule is not None:
            msg = schema.row_rule(d)
            if msg:
                raise ContractError(name, ln_no, msg)
        key = tuple(row[i] for i in key_idx)
        if prev_key is not None:
            if key == prev_key:
                raise ContractError(name, ln_no, f"duplicate key {key}")
            if key < prev_key:
                raise ContractError(name, ln_no, "rows not sorted by the key")
        prev_key = key
        for c, s in seen.items():
            v = d[c]
            if v in s:
                raise ContractError(name, ln_no, f"duplicate {c} {v!r}")
            s.add(v)
        rows.append(row)
        dict_rows.append(d)
    if schema.table_rule is not None:
        bad = schema.table_rule(dict_rows)
        if bad is not None:
            raise ContractError(name, bad[0] + 2, bad[1])
    return Table(schema, rows)


def load(schema: Schema, path: str) -> Table:
    with open(path, "rb") as fh:
        data = fh.read()
    return load_bytes(schema, data)


def dump_bytes(schema: Schema, rows: Iterable[Sequence]) -> bytes:
    fields = schema.fields
    out = [schema.header]
    for r in rows:
        if len(r) != len(fields):
            raise ValueError(f"{schema.name}: row has {len(r)} fields")
        out.append(",".join(_format_field(f, v) for f, v in zip(fields, r)))
    return ("\n".join(out) + "\n").encode("utf-8")


def validate_rows(schema: Schema, rows: Iterable[Sequence]) -> Table:
    """Format and re-load rows: the writer and the loader agree on every rule."""
    return load_bytes(schema, dump_bytes(schema, rows))
