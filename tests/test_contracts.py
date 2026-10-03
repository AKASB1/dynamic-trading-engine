"""Check 1 (file part): the golden fixtures, byte round trips, malformed variants, hashes."""

import dataclasses
import os

import pytest
from conftest import GOLDEN

from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import canonical_json, config_hash

FILES = {name: os.path.join(GOLDEN, name + ".csv") for name in S.SCHEMAS}


def read(name):
    with open(FILES[name], "rb") as fh:
        return fh.read()


@pytest.mark.parametrize("name", sorted(S.SCHEMAS))
def test_fixture_loads_and_round_trips(name):
    data = read(name)
    table = S.load_bytes(S.SCHEMAS[name], data)
    assert len(table) >= 1
    assert S.dump_bytes(S.SCHEMAS[name], table.rows) == data


@pytest.mark.parametrize("name", sorted(S.SCHEMAS))
def test_header_only_is_valid_and_zero_bytes_is_not(name):
    sch = S.SCHEMAS[name]
    assert len(S.load_bytes(sch, (sch.header + "\n").encode())) == 0
    with pytest.raises(S.ContractError) as e:
        S.load_bytes(sch, b"")
    assert e.value.line == 0


def test_crlf_is_accepted():
    data = read("bars_v1").replace(b"\n", b"\r\n")
    assert len(S.load_bytes(S.BARS, data)) == 6


def _replace_line(data: bytes, line_no: int, new: str) -> bytes:
    lines = data.decode().split("\n")
    lines[line_no - 1] = new
    return "\n".join(lines).encode()


BARS_L2 = "A,2024-03-04T14:30:00Z,2024-03-04T21:00:00Z,2024-03-04T21:00:00Z,100.0,101.5,99.5,101.0,1000000"

MALFORMED = [
    # (schema, mutation, expected line)
    ("bars_v1", lambda d: _replace_line(d, 1, "instrument_id,ts_open,ts_event,open"), 1),
    ("bars_v1", lambda d: _replace_line(d, 3, BARS_L2 + ",1"), 3),  # wrong field count
    (
        "bars_v1",
        lambda d: _replace_line(
            d, 2, BARS_L2.replace("2024-03-04T21:00:00Z,100", "2024-03-04 21:00:00,100")
        ),
        2,
    ),
    (
        "bars_v1",
        lambda d: _replace_line(
            d,
            2,
            "A,2024-03-04T14:30:00Z,2024-03-04T21:00:00Z,2024-03-04T20:00:00Z,100.0,101.5,99.5,101.0,1000000",
        ),
        2,
    ),
    (
        "bars_v1",
        lambda d: _replace_line(
            d,
            2,
            "A,2024-03-04T14:30:00Z,2024-03-04T21:00:00Z,2024-03-04T21:00:00Z,100.0,99.0,99.5,99.2,1000000",
        ),
        2,
    ),
    (
        "bars_v1",
        lambda d: _replace_line(
            d,
            2,
            "A,2024-03-04T14:30:00Z,2024-03-04T21:00:00Z,2024-03-04T21:00:00Z,0.0,101.5,99.5,101.0,1000000",
        ),
        2,
    ),
    ("bars_v1", lambda d: _replace_line(d, 3, BARS_L2), 3),  # duplicate key
    (
        "bars_v1",
        lambda d: b"\n".join(
            [d.split(b"\n")[0], d.split(b"\n")[2], d.split(b"\n")[1]] + d.split(b"\n")[3:]
        ),
        3,
    ),
    ("bars_v1", lambda d: b"", 0),
    (
        "series_v1",
        lambda d: _replace_line(d, 3, "A.eps,2023-12-31T21:00:00Z,2024-03-05T21:00:00Z,0,1.05"),
        3,
    ),
    (
        "series_v1",
        lambda d: _replace_line(d, 3, "A.eps,2023-12-31T21:00:00Z,2024-02-10T21:00:00Z,1,1.05"),
        3,
    ),
    (
        "corporate_actions_v1",
        lambda d: _replace_line(
            d, 2, "A,special_dividend,2024-03-06T14:30:00Z,2024-02-26T21:00:00Z,0.5"
        ),
        2,
    ),
    (
        "instruments_v1",
        lambda d: _replace_line(
            d, 2, "A,A,equity,USD,2024-01-02T21:00:00Z,2023-12-01T21:00:00Z,-0.5,0,0.01,tech"
        ),
        2,
    ),
    (
        "instruments_v1",
        lambda d: _replace_line(
            d, 2, "A,A,equity,USD,2024-01-02T21:00:00Z,2024-12-01T21:00:00Z,,0,0.01,tech"
        ),
        2,
    ),
    ("bars_v1", lambda d: _replace_line(d, 2, BARS_L2.replace("101.0,1000000", "nan,1000000")), 2),
    (
        "bars_v1",
        lambda d: _replace_line(
            d, 2, BARS_L2.replace("2024-03-04T14:30:00Z", "2024-13-04T14:30:00Z")
        ),
        2,
    ),
    (
        "equity_v1",
        lambda d: _replace_line(
            d,
            3,
            "2024-03-05T21:00:00Z,89845.94,10200.0,100045.94,0.0,51.0,2.03,1.015,1.015,0.0,0.0,0.0",
        ),
        3,
    ),
    ("positions_v1", lambda d: _replace_line(d, 2, "2024-03-05T21:00:00Z,A,100,102.0,10300.0"), 2),
    (
        "orders_v1",
        lambda d: _replace_line(d, 2, "o1,2024-03-04T21:00:00Z,A,100,market,101.0,day,fixture"),
        2,
    ),
    (
        "liquidity_v1",
        lambda d: _replace_line(d, 2, "2024-03-04T21:00:00Z,2024-03-04T21:00:00Z,A,0,0.02"),
        2,
    ),
    (
        "fills_v1",
        lambda d: _replace_line(
            d, 2, "f1,o1,2024-03-05T14:30:00Z,A,100,101.5,101.53045,-2.03,1.015,1.015"
        ),
        2,
    ),
]


@pytest.mark.parametrize("case", range(len(MALFORMED)))
def test_malformed_variants_are_rejected_with_line(case):
    name, mutate, line = MALFORMED[case]
    with pytest.raises(S.ContractError) as e:
        S.load_bytes(S.SCHEMAS[name], mutate(read(name)))
    assert e.value.line == line, str(e.value)


def test_at_least_twelve_malformed_variants():
    assert len(MALFORMED) >= 12


def test_writer_formats():
    assert S.format_float(1000000.0, integral=True) == "1000000"
    assert S.format_float(0.5, integral=True) == "0.5"
    assert S.format_float(2.0) == "2.0"
    assert S.format_float(-0.0) == "0.0"
    with pytest.raises(ValueError):
        S.format_float(float("nan"))


@dataclasses.dataclass(frozen=True)
class _Cfg:
    a: int = 1
    b: float = 0.5
    c: str = "x"


def test_config_hash_omits_defaults():
    assert canonical_json(_Cfg()) == "{}"
    assert canonical_json(_Cfg(a=2, b=0.25)) == '{"a":2,"b":0.25}'
    assert config_hash(_Cfg()) == config_hash({})
    assert config_hash(_Cfg(a=2)) != config_hash(_Cfg())
