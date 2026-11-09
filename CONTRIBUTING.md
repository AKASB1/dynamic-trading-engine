# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Python 3.12 (`pyproject.toml`). `scripts/setup_env.ps1` (PowerShell) or `bash scripts/setup_env.sh` creates `.venv` and installs the package with its development tools; `setup_env.sh` takes the interpreter from `PYTHON`, else `python3`, else `python`.
- The stack: NumPy, SciPy, CVXPY with Clarabel (OSQP and SCS as second solvers in the tests), and Matplotlib; the tests use pytest, pytest-repeat, and ruff.
- The tests and the experiments generate their synthetic markets and download no data.

## Checks

Run these before opening a pull request. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python -m dynamic_trading_engine experiment --quick --workers 2
python -m dynamic_trading_engine validate --root experiments/outputs/quick/logs
python -m dynamic_trading_engine audit --out experiments/outputs/audit_table.csv
```

## Results

The results in the README, in [simulations/README.md](simulations/README.md), and in `docs/results*.md` come from the committed files in `experiments/results/`. The cross-platform comparison of the quick results comes from an independent Linux check and has no committed file.

A change must leave the quick results of the evaluation unchanged: after `experiment --quick`, `python -m dynamic_trading_engine quickhash` must reproduce `experiments/results/quick_sha256.txt` on the same platform and software stack. A change that moves a result goes through the protocol of [docs/protocol.md](docs/protocol.md): tuning, freezing, and one evaluation on the evaluation seeds, with the result files and manifests committed. `python scripts/make_tables.py` and `python scripts/make_tables_tier2.py` write the result tables; the README's highlights and tables are written by hand from them.

## Figures and diagrams

`python scripts/plot_results.py` draws the figures in `docs/figures/` from the committed results, including the two README versions, and `python scripts/plot_tier2.py` draws those of the later studies. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs none of these.

## Style

- Decisions see the market only through the knowledge-bounded `DecisionState`; the truth module stays out of the decision pipeline's import graph, and a static test checks it ([docs/information.md](docs/information.md)).
- `ce_ann` is the only quantity that selects a configuration, orders a table, or decides a highlight ([docs/protocol.md](docs/protocol.md)).
- Seeds keep their ranges: development and tests 1 to 99, tuning 100 to 199, evaluation 1000 and up.
- Oracle forecasts are labeled bounds and never compete.
- Documentation states what was measured, under which assumptions, and names the result file.
