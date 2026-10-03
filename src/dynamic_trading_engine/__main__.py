import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

from dynamic_trading_engine.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
