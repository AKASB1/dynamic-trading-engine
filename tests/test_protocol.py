"""Check 16: protocol integrity (seed guards, frozen configurations, quick labels, the table
builder's refusals, equal tuning effort, one selection criterion)."""

import ast
import csv
import math
import os

import pytest

import dynamic_trading_engine
from dynamic_trading_engine.experiments import protocol as PT
from dynamic_trading_engine.experiments import registry as R
from dynamic_trading_engine.experiments.criterion import CRITERION
from dynamic_trading_engine.experiments.jobs import Job
from dynamic_trading_engine.experiments.tables import TableError, check_rows, markdown

PKG = os.path.dirname(dynamic_trading_engine.__file__)


def test_tuner_refuses_seeds_outside_100_199():
    for bad in ([99], [200], [1000], [5, 100]):
        with pytest.raises(PT.ProtocolError):
            PT.check_tuning_seeds(bad)
    PT.check_tuning_seeds([100, 101, 199])
    reg = R.load_registry()
    reg["seeds"]["tuning"] = [100, 1000]
    with pytest.raises(PT.ProtocolError):
        PT._tune_jobs(reg, False, "base", "TUNE")


def test_evaluation_refuses_low_seeds_and_unfrozen_configurations():
    with pytest.raises(PT.ProtocolError):
        PT.check_eval_seeds([999, 1000], quick=False)
    PT.check_eval_seeds([1000, 1029], quick=False)
    with pytest.raises(PT.ProtocolError):
        PT.check_eval_seeds([100], quick=True)
    job = Job("S2", "X", R.base_spec(R.load_registry(), "MV"), "tiny", 1, config_hash="0" * 64)
    with pytest.raises(PT.ProtocolError):
        PT.check_frozen([job], quick=True)


def test_quick_plan_is_frozen_and_labelled():
    from dynamic_trading_engine.experiments.suite import plan

    reg = R.load_registry()
    jobs = plan(reg, PT.load_tuned(reg, True), True)
    PT.check_frozen(jobs, quick=True)
    assert all("QUICK" in j.label.split("+") for j in jobs)
    assert {j.seed for j in jobs} <= {1, 2, 3}
    # a changed grid changes a hash, and the evaluation refuses it
    reg["S3"]["cost_scale"] = [0.0, 0.3]
    with pytest.raises(PT.ProtocolError):
        PT.check_frozen(plan(reg, PT.load_tuned(reg, True), True, ("S3",)), quick=True)


def test_full_plan_is_frozen_when_the_protocol_run_exists():
    if not os.path.exists(PT.frozen_path(False)):
        pytest.skip("the protocol tuning run and the freeze have not happened yet")
    from dynamic_trading_engine.experiments.suite import plan

    reg = R.load_registry()
    jobs = plan(reg, PT.load_tuned(reg, False), False)
    PT.check_frozen(jobs, quick=False)
    assert min(j.seed for j in jobs) >= 1000


def test_table_builder_refuses_quick_and_unmarked_oracle_rows():
    with pytest.raises(TableError):
        check_rows([{"strategy": "MV", "label": "QUICK", "ce_ann_mean": 0.1}])
    with pytest.raises(TableError):
        check_rows([{"variant": "MV_oracle", "label": "", "oracle": "true"}])
    with pytest.raises(TableError):
        check_rows([{"variant": "MV_noisy_0.5", "label": ""}])
    md = markdown(
        [{"variant": "MV_oracle", "label": "ORACLE", "ce_ann_mean": 0.1}],
        [("variant", "v"), ("label", "l")],
    )
    assert "ORACLE" in md


def _tuning_log():
    full = os.path.join(R.path(R.TUNED_DIR), "tuning_log.csv")
    quick = os.path.join(R.path(R.TUNED_DIR), "quick", "tuning_log.csv")
    path = full if os.path.exists(full) else quick
    with open(path, encoding="utf-8", newline="") as fh:
        return path, list(csv.DictReader(fh))


def _same_params(committed: str, sampled: dict) -> bool:
    """The committed ``k=repr(v);...`` string against a fresh draw: the keys exactly and in order,
    floats to a relative 1e-12 (a log-uniform draw goes through ``exp``, whose last bit differs
    between math libraries), ints, bools, and strings exactly (type included)."""
    got = {}
    for item in committed.split(";"):
        k, _, v = item.partition("=")
        got[k] = ast.literal_eval(v)
    if list(got) != sorted(sampled):
        return False
    for k, a in got.items():
        b = sampled[k]
        if isinstance(a, float) and isinstance(b, float):
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0):
                return False
        elif type(a) is not type(b) or a != b:
            return False
    return True


def test_same_params_compares_keys_exactly_and_floats_numerically():
    assert _same_params(
        "gamma=3.907414346090997;risk='lw'", {"risk": "lw", "gamma": 3.9074143460909974}
    )
    assert not _same_params("gamma=3.9074;risk='lw'", {"risk": "lw", "gamma": 3.9075})
    assert not _same_params("gamma=1.0;risk='lw'", {"risk": "ewma", "gamma": 1.0})
    assert not _same_params("window=60", {"window": 60.0})
    assert not _same_params("gamma=1.0", {"gamma": 1.0, "risk": "lw"})


def test_equal_tuning_effort_in_the_committed_log():
    reg = R.load_registry()
    path, rows = _tuning_log()
    quick = "quick" in path
    want_trials = reg["tuning"]["trials_quick" if quick else "trials"]
    want_seeds = " ".join(map(str, reg["seeds"]["tuning_quick" if quick else "tuning"]))
    for name in reg["s2_order"]:
        space = reg["strategies"][name]["space"]
        mine = [r for r in rows if r["strategy"] == name]
        if not space:
            assert not mine  # EW has no parameters
            continue
        assert len(mine) == want_trials, name
        assert {r["seeds"] for r in mine} == {want_seeds}
        for r in mine:
            params = R.sample_trial(space, name, int(r["trial"]))
            assert _same_params(r["params"], params), (name, r["trial"], r["params"], params)
        retuned = [r for r in rows if r["strategy"] == f"RETUNED_{name}"]
        assert len(retuned) == want_trials


def test_one_selection_criterion():
    assert CRITERION == "ce_ann"
    allowed = {"ce_ann", "ce_ann_mean", "strategy", "seed", "variant", "label", "id"}
    for mod in ("experiments/protocol.py", "experiments/tables.py"):
        with open(os.path.join(PKG, mod), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Compare) and any(
                isinstance(o, (ast.Gt, ast.Lt, ast.GtE, ast.LtE)) for o in node.ops
            ):
                targets.append(node)  # an ordering comparison (a ranking or a selection)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in (
                "max",
                "min",
                "sorted",
            ):
                targets.extend(k.value for k in node.keywords if k.arg == "key")
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant):
                        if isinstance(sub.slice.value, str):
                            assert sub.slice.value in allowed, (mod, sub.slice.value)
