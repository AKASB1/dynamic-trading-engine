"""Parallel execution of jobs (spawn-based process pool) and run manifests.

Workers set the thread variables of every numerical library to 1 before importing them; CVXPY
problem objects are built and cached inside each worker (never pickled); results are sorted
by (experiment, variant, seed) before they are written, so that the output does not depend on
the order in which jobs finish.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Iterable

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def set_thread_env() -> None:
    for k in THREAD_VARS:
        os.environ[k] = "1"
    os.environ.setdefault("PYTHONUTF8", "1")


def _init_worker(root: str | None) -> None:
    set_thread_env()
    if root:
        os.environ["DTE_ROOT"] = root


def _call(fn_job):
    fn, job = fn_job
    return fn(job)


def run_parallel(fn: Callable, jobs: Iterable, workers: int, sort_key=None, progress=None) -> list:
    """Run ``fn(job)`` for every job; serial when workers <= 1. Results in job order."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        out = []
        for i, j in enumerate(jobs):
            out.append(fn(j))
            if progress:
                progress(i + 1, len(jobs))
        return out
    from dynamic_trading_engine.market.config import repo_root

    ctx = mp.get_context("spawn")
    results = [None] * len(jobs)
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(repo_root(),)
    ) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in cf.as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            if progress:
                progress(done, len(jobs))
    return results


RESULT_DIRS = ("experiments/results/", "experiments/outputs/")


def git_state(root: str) -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20
        ).stdout.strip()
        # clean = no tracked file modified outside the result directories that runs write
        # (a run that overwrites committed results must not mark its own tree dirty)
        lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()
        changed = [ln[3:] for ln in lines if ln.strip()]
        dirty = any(not c.startswith(RESULT_DIRS) for c in changed)
        return {
            "commit": sha or "unknown",
            "dirty": dirty,
            "results_dirs_excluded": list(RESULT_DIRS),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1e9, 1)
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except OSError:
        pass
    return None


def manifest(root: str, workers: int, extra: dict | None = None) -> dict:
    import cvxpy
    import numpy
    import scipy

    from dynamic_trading_engine import __version__
    from dynamic_trading_engine.optimization.optimizers import solver_versions
    from dynamic_trading_engine.optimization.problems import (
        CLARABEL_DEFAULT,
        CLARABEL_RETRIES,
        SCALE,
    )

    out = {
        "package_version": __version__,
        "git": git_state(root),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cvxpy": cvxpy.__version__,
        "solvers": solver_versions(),
        "solver_settings": {
            "default": "CLARABEL",
            "clarabel": {**CLARABEL_DEFAULT, "tolerances": "default (1e-8)"},
            "retries": list(CLARABEL_RETRIES),
            "fallback": "SOC form of |z|^1.5 when the power-cone solve fails",
            "warm_start": False,
            "objective_scale": SCALE,
        },
        "threads": {k: os.environ.get(k) for k in THREAD_VARS},
        "workers": workers,
        "hardware": {
            "cpu": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "ram_gb": ram_gb(),
            "os": f"{platform.system()} {platform.release()}",
        },
    }
    if extra:
        out.update(extra)
    return out
