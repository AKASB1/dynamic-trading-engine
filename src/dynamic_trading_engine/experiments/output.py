"""Deterministic CSV output for results (shortest round-trip floats, fixed column order).
(Adapted from the owner's rollout-engine, MIT.)"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os


def fmt(v) -> str:
    if v is None:
        return ""
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):  # numpy scalar
        v = v.item()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        v = float(v)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(0.0 if v == 0 else v)
    return str(v)


def columns_of(rows: list[dict], first: list[str]) -> list[str]:
    cols = list(first)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def to_csv_bytes(rows: list[dict], first: list[str]) -> bytes:
    cols = columns_of(rows, first)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in cols])
    return buf.getvalue().encode("utf-8")


def write_csv(path: str, rows: list[dict], first: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(to_csv_bytes(rows, first))


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def hash_without_wall(path: str) -> str:
    """SHA-256 of a CSV result file with every ``wall_`` column removed."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return hashlib.sha256(b"").hexdigest()
    keep = [i for i, c in enumerate(rows[0]) if not c.startswith("wall_")]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow([r[i] for i in keep])
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()
