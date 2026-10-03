"""Check 17: the quick experiments run S1 to S7 (and SENS) at the quick scale, every log they
write passes the independent validator, the results are labelled QUICK (and refused by the table
builder), and the report of one run is produced by the command of the README."""

import os
import time

import pytest

from dynamic_trading_engine.cli.main import main as cli
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.output import read_csv
from dynamic_trading_engine.experiments.suite import run_all
from dynamic_trading_engine.experiments.tables import TableError, check_rows

EXPECTED = (
    "runs",
    "s2_summary",
    "s2_paired",
    "s2_dsr",
    "s2b_ladder",
    "s3",
    "s4",
    "s5_shift",
    "s6c",
    "sens",
    "exact_references",
    "s1_timing",
    "s6a_gap",
    "s6b_proportional",
    "s7_frontier",
    "s7_frontier_checks",
    "s7_policies",
    "s7_latency_cap",
    "s7_decomposition",
)


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    out = tmp_path_factory.mktemp("quick")
    t0 = time.perf_counter()
    res = run_all(R.load_registry(), True, 2, str(out), log=lambda *a: None)
    return out, res, time.perf_counter() - t0


def test_quick_runs_every_experiment(quick):
    out, res, wall = quick
    for name in EXPECTED:
        assert os.path.exists(os.path.join(out, f"{name}_quick.csv")), name
    assert os.path.exists(os.path.join(out, "manifest_quick.json"))
    assert wall < 600


def test_quick_rows_are_labelled_and_refused_by_the_table_builder(quick):
    out, res, _ = quick
    rows = res["rows"]
    assert all("QUICK" in r["label"].split("+") for r in rows)
    with pytest.raises(TableError):
        check_rows(read_csv(os.path.join(out, "s2_summary_quick.csv")))


def test_every_written_log_was_validated(quick):
    _, res, _ = quick
    logged = [r for r in res["rows"] if r.get("validated")]
    variants = {
        (r["experiment"], r["variant"])
        for r in res["rows"]
        if r["experiment"] in ("S2", "S3", "S6c")
    }
    assert {(r["experiment"], r["variant"]) for r in logged} >= variants
    assert all(r["validator_max_residual"] <= 1e-9 for r in logged)
    assert max(r["max_identity_residual"] for r in res["rows"]) <= 1e-9


def test_validate_and_report_commands(quick, tmp_path):
    root = R.path(R.OUTPUTS_DIR, "quick", "logs", "S2", "MV")
    assert cli(["validate", "--root", root]) == 0
    assert cli(["report", "--run", os.path.join(root, "run"), "--out", str(tmp_path)]) == 0
    assert os.path.exists(tmp_path / "report.md") and os.path.exists(tmp_path / "report.png")
