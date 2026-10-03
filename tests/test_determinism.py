"""Check 15 (the rest): byte-identical results in one process, across fresh processes, serial
and parallel; the in-memory market and the market read back from its files; seeds differ;
adding a strategy changes no other row; aggregates independent of completion order."""

import os
import random
import subprocess
import sys

import numpy as np

from dynamic_trading_engine.contracts.costs import CostConfig
from dynamic_trading_engine.engine.loop import LoopConfig, run_loop
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.jobs import Job, run_job
from dynamic_trading_engine.experiments.output import to_csv_bytes
from dynamic_trading_engine.experiments.runner import manifest, run_parallel
from dynamic_trading_engine.market.config import load_market_config
from dynamic_trading_engine.market.generator import generate
from dynamic_trading_engine.market.store import read_market, write_market
from dynamic_trading_engine.strategies.factory import build_strategy

HERE = os.path.dirname(os.path.abspath(__file__))


def _jobs(names=("EW", "MV", "CVAR", "MV_ORACLE"), seeds=(1, 2)):
    reg = R.load_registry()
    out = []
    for n in names:
        spec = R.base_spec(reg, "MV") if n == "MV_ORACLE" else R.base_spec(reg, n)
        if n == "MV_ORACLE":
            spec["forecast"], spec["strategy_id"] = "oracle", n
        for s in seeds:
            out.append(Job("T", n, spec, "tiny", s))
    return out


def _bytes(rows):
    rows = sorted(rows, key=lambda r: (r["variant"], r["seed"]))
    clean = [{k: v for k, v in r.items() if not k.startswith("wall_")} for r in rows]
    return to_csv_bytes(clean, ["experiment", "variant", "seed"])


def test_same_process_twice_and_parallel_are_identical():
    a = _bytes([run_job(j) for j in _jobs()])
    b = _bytes([run_job(j) for j in _jobs()])
    c = _bytes(run_parallel(run_job, _jobs(), 2))
    assert a == b == c


CODE = (
    "import sys;from test_determinism import _jobs,_bytes;"
    "from dynamic_trading_engine.experiments.jobs import run_job;"
    "sys.stdout.buffer.write(_bytes([run_job(j) for j in _jobs()]))"
)


def test_fresh_processes_are_identical():
    env = dict(os.environ, PYTHONPATH=HERE, DTE_ROOT=R.path())
    outs = []
    for h in ("1", "2"):
        env["PYTHONHASHSEED"] = h
        p = subprocess.run(
            [sys.executable, "-c", CODE], cwd=HERE, env=env, capture_output=True, timeout=300
        )
        assert p.returncode == 0, p.stderr.decode()[-2000:]
        outs.append(p.stdout)
    assert outs[0] == outs[1] == _bytes([run_job(j) for j in _jobs()])


def test_in_memory_market_equals_market_read_from_files(tmp_path):
    cfg = load_market_config("tiny")
    m, tr = generate(cfg, 3)
    write_market(m, str(tmp_path))
    m2 = read_market(str(tmp_path))
    spec = R.base_spec(R.load_registry(), "MV")
    logs = []
    for mk in (m, m2):
        st = build_strategy(spec, CostConfig(), 5, truth=tr, seed=3)
        logs.append(
            run_loop(mk, st, CostConfig(), LoopConfig(warmup_bars=100), write_logs=True).logs
        )
    assert logs[0] == logs[1]


def test_seeds_differ_and_adding_a_strategy_changes_nothing_else():
    rows_two = {(r["variant"], r["seed"]): r for r in (run_job(j) for j in _jobs(("EW", "MV")))}
    rows_one = {(r["variant"], r["seed"]): r for r in (run_job(j) for j in _jobs(("MV",)))}
    for k, r in rows_one.items():
        a = {x: v for x, v in r.items() if not x.startswith("wall_")}
        b = {x: v for x, v in rows_two[k].items() if not x.startswith("wall_")}
        assert a == b
    assert rows_two[("MV", 1)]["ce_ann"] != rows_two[("MV", 2)]["ce_ann"]


def test_aggregates_do_not_depend_on_completion_order():
    from dynamic_trading_engine.experiments.suite import agg_simple

    rows = [run_job(j) for j in _jobs(("EW", "MV"))]
    for r in rows:
        r["experiment"] = "S3"
    a = agg_simple(rows, "S3", "EW")
    shuffled = rows[:]
    random.Random(4).shuffle(shuffled)
    assert a == agg_simple(shuffled, "S3", "EW")


def test_manifest_lists_solvers_tolerances_threads():
    m = manifest(R.path(), 2)
    assert m["solvers"]["CLARABEL"] and m["solvers"]["cvxpy"]
    assert "max_threads" in m["solver_settings"]["clarabel"]
    assert set(m["threads"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    assert m["git"]["commit"]
    assert np.isfinite(m["hardware"]["logical_cpus"])
