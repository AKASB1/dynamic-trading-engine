"""The ``equity_daily`` calendar: Monday to Friday except 1 January and 25 December (a fixed,
documented holiday rule), sessions 14:30:00Z to 21:00:00Z, no daylight-saving handling."""

from __future__ import annotations

import datetime as dt

import numpy as np

from dynamic_trading_engine.contracts.timeutil import US_PER_S

OPEN_S = (14 * 60 + 30) * 60
CLOSE_S = 21 * 3600


def is_session(d: dt.date) -> bool:
    if d.weekday() >= 5:
        return False
    return (d.month, d.day) not in ((1, 1), (12, 25))


def equity_daily(start: str, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """``ts_open`` and ``ts_event`` (int64 microseconds) of the first ``n_bars`` sessions on
    or after ``start`` (``YYYY-MM-DD``)."""
    d = dt.date.fromisoformat(start)
    epoch = dt.date(1970, 1, 1)
    opens, events = [], []
    while len(events) < n_bars:
        if is_session(d):
            day_s = (d - epoch).days * 86400
            opens.append((day_s + OPEN_S) * US_PER_S)
            events.append((day_s + CLOSE_S) * US_PER_S)
        d += dt.timedelta(days=1)
    return np.array(opens, dtype=np.int64), np.array(events, dtype=np.int64)
