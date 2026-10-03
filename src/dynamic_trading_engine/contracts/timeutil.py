"""UTC instants: `YYYY-MM-DDTHH:MM:SSZ` in files, `datetime64[us]` in memory."""

from __future__ import annotations

import re

import numpy as np

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
US_PER_S = 1_000_000
US_PER_DAY = 86_400 * US_PER_S


def parse_ts(text: str) -> int:
    """Parse a contract timestamp into integer microseconds since the epoch (UTC)."""
    if not TS_RE.match(text):
        raise ValueError(f"bad timestamp {text!r}")
    try:
        v = np.datetime64(text[:-1], "us")
    except ValueError as e:
        raise ValueError(f"bad timestamp {text!r}") from e
    if np.isnat(v):
        raise ValueError(f"bad timestamp {text!r}")
    # numpy accepts some out-of-range fields silently in no case we rely on; round trip guards it
    out = int(v.astype(np.int64))
    if format_ts(out) != text:
        raise ValueError(f"bad timestamp {text!r}")
    return out


def format_ts(us: int) -> str:
    if us % US_PER_S != 0:
        raise ValueError("instants in files have seconds precision")
    return str(np.datetime64(int(us), "us").astype("datetime64[s]")) + "Z"


def format_ts_array(us: np.ndarray) -> list[str]:
    arr = np.asarray(us, dtype=np.int64)
    if np.any(arr % US_PER_S != 0):
        raise ValueError("instants in files have seconds precision")
    return [s + "Z" for s in np.datetime_as_string(arr.astype("datetime64[us]"), unit="s")]


def days_between(a_us: int, b_us: int) -> float:
    """Decimal number of days between two instants (QC 1.1)."""
    return (b_us - a_us) / US_PER_DAY
