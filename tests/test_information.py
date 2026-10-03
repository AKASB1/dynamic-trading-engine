"""Check 2 (the rest): the static import test, the replay audit with the canaries, the purge
self-test, and the comparison with the committed audit table."""

import ast
import csv
import os

import pytest

import dynamic_trading_engine
from dynamic_trading_engine.audit import replay as RP
from dynamic_trading_engine.audit.run import run_audit

PKG_DIR = os.path.dirname(dynamic_trading_engine.__file__)
ROOT = os.path.dirname(os.path.dirname(PKG_DIR))
CANARIES = os.path.join(ROOT, "tests", "canaries")
TABLE = os.path.join(ROOT, "experiments", "results", "audit_table.csv")
TRUTH = "dynamic_trading_engine.market.truth"

ALLOWED = {
    "dynamic_trading_engine.market.truth",
    "dynamic_trading_engine.market.generator",
    "dynamic_trading_engine.forecasts.oracle",
    "dynamic_trading_engine.strategies.factory",
}
ALLOWED_PREFIXES = (
    "dynamic_trading_engine.audit",
    "dynamic_trading_engine.experiments",
    "dynamic_trading_engine.cli",
    "dynamic_trading_engine.__main__",
)
PIPELINE = (
    "dynamic_trading_engine.state",
    "dynamic_trading_engine.forecasts.base",
    "dynamic_trading_engine.forecasts.simple",
    "dynamic_trading_engine.forecasts",
    "dynamic_trading_engine.risk",
    "dynamic_trading_engine.optimization",
    "dynamic_trading_engine.policies",
    "dynamic_trading_engine.execution",
    "dynamic_trading_engine.engine",
    "dynamic_trading_engine.strategies.pipeline",
    "dynamic_trading_engine.strategies",
    "dynamic_trading_engine.market",
    "dynamic_trading_engine.market.data",
    "dynamic_trading_engine.market.store",
)


@pytest.fixture(scope="module", autouse=True)
def _restore_purged_modules():
    """The audit purges this package's pipeline modules from ``sys.modules`` and re-imports them.
    Put the original module objects back after this file, so that later test files see the same
    classes as when they run alone (an import inside a later test would otherwise return a fresh
    copy whose classes differ from those imported at collection)."""
    import sys

    before = {n: m for n, m in sys.modules.items() if m is not None and not RP._is_kept(n, m)}
    yield
    for n, m in list(sys.modules.items()):
        if n not in before and m is not None and not RP._is_kept(n, m):
            del sys.modules[n]
    sys.modules.update(before)
    for n in sorted(before):
        parent, _, child = n.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], child, before[n])


def _modules():
    mods = {}
    for dirpath, _d, files in os.walk(PKG_DIR):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, os.path.dirname(PKG_DIR))[:-3].replace(os.sep, ".")
            name = rel[: -len(".__init__")] if rel.endswith(".__init__") else rel
            mods[name] = path
    return mods


def _imports(name, path, mods):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    is_pkg = path.endswith("__init__.py")
    base = name if is_pkg else name.rsplit(".", 1)[0]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = base.split(".")
                parent = ".".join(parts[: len(parts) - node.level + 1])
                mod = f"{parent}.{node.module}" if node.module else parent
            else:
                mod = node.module or ""
            out.add(mod)
            for a in node.names:
                if f"{mod}.{a.name}" in mods:
                    out.add(f"{mod}.{a.name}")
    # importing a.b.c imports the packages a and a.b (their __init__) too
    full = set()
    for m in out:
        p = m.split(".")
        for i in range(1, len(p) + 1):
            full.add(".".join(p[:i]))
    return {m for m in full if m in mods and m != name}


def test_static_import_graph_keeps_the_truth_out_of_the_pipeline():
    mods = _modules()
    graph = {n: _imports(n, p, mods) for n, p in mods.items()}
    reaches = {}

    def reach(n, seen=()):
        if n in reaches:
            return reaches[n]
        if n in seen:
            return False
        r = n == TRUTH or any(reach(m, seen + (n,)) for m in graph[n])
        reaches[n] = r
        return r

    importers = sorted(n for n in mods if reach(n) and n != TRUTH)
    for n in importers:
        assert n in ALLOWED or n.startswith(ALLOWED_PREFIXES), f"{n} imports the truth module"
    for n in PIPELINE:
        assert n in mods and not reach(n), f"pipeline module {n} reaches the truth"


def _table():
    with open(TABLE, encoding="utf-8", newline="") as fh:
        return {r["subject"]: r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def audit_rows():
    return run_audit(CANARIES, n_instants=30, n_instants_builtin=3, log=lambda *a: None)


def test_every_canary_is_caught_by_its_guard(audit_rows):
    can = [r for r in audit_rows if r["subject"].startswith("canary:")]
    assert {r["subject"] for r in can} == {"canary:D1", "canary:D1b", "canary:D2", "canary:D3"}
    for r in can:
        assert r[r["expected_guard"]] == "caught", r
    d3 = next(r for r in can if r["subject"] == "canary:D3")
    assert d3["garbage_truth"] == "caught"


def test_no_builtin_strategy_raises_an_alarm(audit_rows):
    for r in audit_rows:
        if not r["subject"].startswith("canary:"):
            assert not r["alarm"], r


def test_audit_cells_match_the_committed_table(audit_rows):
    committed = _table()
    assert set(committed) == {r["subject"] for r in audit_rows}
    for r in audit_rows:
        c = committed[r["subject"]]
        for g in ("replay", "garbage_truth", "label"):
            assert c[g] == r[g], (r["subject"], g)


def test_without_the_purge_the_memoized_handle_escapes(monkeypatch):
    """Self-test of the mechanism: if the harness did not purge the helper module between
    worlds, D1's memoized handle would keep the first world's market and the leak would pass."""
    monkeypatch.setattr(RP, "purge_modules", lambda: [])
    rows = run_audit(CANARIES, n_instants=4, subjects=["canary:D1"], log=lambda *a: None)
    assert rows[0]["replay"] == "passed"
