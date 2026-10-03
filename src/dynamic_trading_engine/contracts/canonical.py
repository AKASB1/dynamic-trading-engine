"""Canonical JSON, configuration hashes, and manifests (QC 1.5).

A configuration hash is the SHA-256 of the canonical JSON: UTF-8, sorted keys, no whitespace,
integers without a decimal point, floats in shortest round-trip form, and every field that
equals its default omitted (so that a field added later with a default changes no hash).
Configurations here are frozen dataclasses; ``canonical_dict`` omits default-valued fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from typing import Any

QC_VERSION = 1
SCHEMA_VERSION = 1


def _check_numbers(obj: Any) -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("NaN and inf are not allowed in canonical JSON")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check_numbers(v)


def _default_of(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


def _plain(v: Any) -> Any:
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return canonical_dict(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, float) and v == 0.0:
        return 0.0
    return v


def canonical_dict(cfg: Any) -> dict:
    """The dataclass as a plain dict with every default-valued field omitted (recursively)."""
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        d = _default_of(f)
        if d is not dataclasses.MISSING and _plain(v) == _plain(d):
            continue
        out[f.name] = _plain(v)
    return out


def canonical_json(obj: Any) -> str:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = canonical_dict(obj)
    obj = _plain(obj)
    _check_numbers(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def config_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(obj: Any) -> str:
    """Repository JSON: UTF-8, sorted keys, two-space indentation, trailing newline."""
    return (
        json.dumps(_plain(obj), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dump_json(obj))


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def manifest_for(data: bytes, row_count: int, generator: dict, seed: int | None) -> dict:
    return {
        "content_sha256": sha256_bytes(data),
        "generator": generator,
        "qc_version": QC_VERSION,
        "row_count": row_count,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
    }


def write_dataset(
    path_csv: str, data: bytes, row_count: int, generator: dict, seed: int | None
) -> None:
    """Write ``<name>.csv`` and its ``<name>.manifest.json``."""
    os.makedirs(os.path.dirname(os.path.abspath(path_csv)), exist_ok=True)
    with open(path_csv, "wb") as fh:
        fh.write(data)
    write_json(path_csv[:-4] + ".manifest.json", manifest_for(data, row_count, generator, seed))


def read_dataset_bytes(path_csv: str, check_manifest: bool = True) -> bytes:
    with open(path_csv, "rb") as fh:
        data = fh.read()
    if check_manifest:
        man = read_json(path_csv[:-4] + ".manifest.json")
        if man.get("schema_version") != SCHEMA_VERSION or man.get("qc_version") != QC_VERSION:
            raise ValueError(f"{path_csv}: unsupported schema or contract version")
        if man.get("content_sha256") != sha256_bytes(data):
            raise ValueError(f"{path_csv}: content hash does not match its manifest")
    return data
