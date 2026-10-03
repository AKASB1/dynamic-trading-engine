import os

# Single-threaded numerical libraries before anything imports NumPy (TASK: shared machine,
# deterministic reductions).
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("PYTHONUTF8", "1")

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "golden")
