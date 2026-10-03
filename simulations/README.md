# Simulations

Every number produced here is **simulated** on the synthetic market of `docs/data.md` with the contract's cost model and the assumptions stated in `docs/protocol.md`. It shows how the formulations, the controller, and the execution simulator behave in this simulation, under these assumptions; it says nothing about real markets, profitability, or production performance.

## How to reproduce

From the repository root, with the environment of the README (`.venv` from `scripts/setup_env.*`) active:

```bash
python -m dynamic_trading_engine tune --workers 2          # protocol tuning run (tuning seeds 100-103) -> experiments/tuned/
python -m dynamic_trading_engine freeze                    # experiments/tuned/frozen.json
python -m dynamic_trading_engine experiment --workers 2    # full evaluation (seeds >= 1000) -> experiments/results/
python -m dynamic_trading_engine audit                     # information audit -> experiments/results/audit_table.csv
python scripts/plot_results.py                             # figures -> docs/figures/
python scripts/make_tables.py                              # tables -> docs/results.md
```

`--quick` versions (the `tiny` market, development seeds 1-3, tuning seeds 100-101, 4 trials, its own frozen file, outputs labelled QUICK under `experiments/outputs/quick/`):

```bash
python -m dynamic_trading_engine tune --quick --workers 2
python -m dynamic_trading_engine freeze --quick
python -m dynamic_trading_engine experiment --quick --workers 2
```

The evaluation refuses a configuration whose hash is not in the frozen file and any seed below 1000; the tuner refuses any seed outside 100-199. Per-run logs (the contract's order, fill, position, and equity files plus `decisions.csv`) are written for the first seed of every variant of S2, S2b, S3, S5 (without its `RETUNED` bound), and S6(c) to the ignored `experiments/outputs/` and checked there by the independent validator; every run asserts the accounting identity in memory.

Tier 2 (after the Tier-1 evaluation; `experiments/configs/registry_tier2.json`, frozen in `experiments/tuned/frozen_tier2.json`; results in `experiments/results/tier2/`):

```bash
python -m dynamic_trading_engine tier2 tune --workers 2         # WDRO and GP_AIM, 16 trials each on tuning seeds 100-103
python -m dynamic_trading_engine tier2 freeze                   # their tuned configurations -> frozen_tier2.json
python -m dynamic_trading_engine tier2 freeze-more              # the loop configurations of the fat-tail, cost and capacity studies
python -m dynamic_trading_engine tier2 run --workers 2          # WDRO and GP_AIM on base and shift
python -m dynamic_trading_engine tier2 adaptive --workers 2     # adaptive execution (runs in one process)
python -m dynamic_trading_engine tier2 solvers --workers 2      # solver comparison (runs in one process)
python -m dynamic_trading_engine tier2 tails --workers 2        # Student-t and jump markets
python -m dynamic_trading_engine tier2 costmis --workers 2      # cost misspecification
python -m dynamic_trading_engine tier2 capacity --workers 2     # capacity
python -m dynamic_trading_engine tier2 export --workers 2       # export consumer
python scripts/plot_tier2.py                                    # figures -> docs/figures/tier2_*.png
python scripts/make_tables_tier2.py                             # tables -> docs/results_tier2.md
```

Tier-2 code must leave the Tier-1 quick results unchanged: after `python -m dynamic_trading_engine experiment --quick --workers 2`, `python -m dynamic_trading_engine quickhash` prints the SHA-256 of every quick result file without its `wall_` columns, which must equal `experiments/results/quick_sha256.txt` (recorded at the evaluation commit on the build machine: Windows 11, Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, CVXPY 1.9.3 with Clarabel 0.11.1, OSQP 1.1.3, and SCS 3.3.1). Bit-for-bit equality holds on that platform only: an independent check on Linux with the same versions got quick results that agree to about 1e-3 relative per run but not byte for byte, so compare numerically there. The committed full results were produced on Windows only.

## Results

See the README's result section and `docs/results.md` (generated). The experiment list (S1 exact references and solver accuracy, S2 the main comparison and the information ladder, S3 constraints/costs/schedule, S4 risk models, S5 distribution shift, S6 multi-period control, S7 execution, S8 post-hoc re-costing, SENS sensitivity) and every grid are in `experiments/configs/registry.json`; the Tier-2 items (Wasserstein-robust mean-CVaR, the multi-asset Garleanu-Pedersen rule, adaptive execution, the solver comparison, fat tails and jumps, cost misspecification, capacity, the export consumer) are in the README's Tier-2 section and `docs/results_tier2.md` (generated).

## TODO

- Repeat the evaluation on more than one synthetic market family (other factor structures, other signal strengths); today every result comes from one generator with one set of assumed parameters.
- Measure wall-clock numbers on an idle machine and record the load in the manifests; today they come from a shared workstation (the `wall_` columns of the results and `python -m dynamic_trading_engine bench`, whose full-scale output `experiments/results/bench.csv` is not committed), and the manifests record the hardware, the worker count, and the thread settings but not the load.
- Run the Tier-2 strategies through the information audit; today the audit table covers the Tier-1 strategies and components only.
- Choose the Wasserstein radius out of sample (nested or walk-forward); today `WDRO`'s tuned configuration did not carry over from the tuning to the evaluation seeds.
