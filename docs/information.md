# Information discipline

A strategy may use at a decision instant `t` only rows known at `t` (`ts_avail <= t`). Python cannot sandbox a strategy, so the boundary is enforced by construction and tested from several sides; a leak must be caught by at least one guard.

## The knowledge view

`state.view.build_state(market, t, ...)` is the only way the decision pipeline sees market data. It cuts every array at the knowledge boundary (bars, returns, liquidity rows, and series rows with `ts_avail <= t`; the latest vintage known at `t` of every series value), applies the point-in-time split adjustment of the splits with `ts_ex <= t` only, builds the total-return series from the returns known at `t`, lists the universe at `t` (listed at or before `t`, not delisted at or before `t`, including instruments that delist later), and hides `ts_delist` and `delist_return` until `ts_delist`. Arrays are read-only copies; the state holds no reference to the market or to the truth (an object-graph walk checks it). Tests: random markets and random instants including extreme ones (`tests/test_view.py`), brute-force universes and split adjustments, the A.eps hand case, the fixture answers, and "the risk estimate from a market cut at `t` equals the one from the full market, bit for bit" (`tests/test_risk.py`).

## The truth handle

`market.generate` returns the market and, separately, a `Truth` (latent signal, conditional mean `m`, regime, loadings, true covariance). The decision pipeline never receives it. A static test (`tests/test_information.py`) reads the import graph of the package from the source and lists the modules that import `market.truth` directly or through other modules (a package `__init__` counts as importing what it imports): only the generator, the oracle providers, the audit, the experiment evaluators, the tuner and runner, the strategy factory, and the command-line entry points. No module of the pipeline (the state, the honest providers, the risk models, the optimizers and the controller, the execution simulator and policies, the loop) is on the list.

## The replay audit

For a strategy given as a spec (built-in specs, or `{module, class, params}` for any class), the audit samples decision instants of the `tiny` market (stream `audit.instants`), builds a poisoned copy of the market for each instant `t` (stream `audit.poison`), and replays the strategy from its first decision up to and including `t` in the real and in the poisoned world. The orders of every replayed decision must be bit-identical.

- **Worlds.** Each world is written to its own directory of contract files; the market registry (`DTE_MARKET_DIR`) points at it, so a module-level handle, a handle opened in a constructor, and a read by path all resolve to that world only.
- **Fresh state per world.** Before each world the harness purges from `sys.modules` every module that is not the standard library, a third-party package, or part of the audit harness (the strategy modules, every helper that may memoize a handle, and the decision pipeline), re-imports the factory and the loop, and instantiates the strategy again from its spec.
- **The poison** keeps every row on its side of `t` and changes what lies after it: bar prices times random factors in [0.1, 10] (high and low still bound open and close) and redrawn volumes, with the liquidity inputs and returns recomputed from them; `signal_x` rows after `t` redrawn and later vintages appended to periods known at `t`; corporate actions announced after `t` changed, removed, or added; delistings after `t` cancelled for some instruments (their bars continue) and added for others; instruments listing after `t` removed and a new one added; another generator seed and parameters in the manifest. The truth after `t` is redrawn, except `m` at `t` and the regime value of the next bar that it contains (the labelled oracles read `m` at `t` by design) and the constants. Pipeline streams (`forecast.*`) come from the seed the factory passes to a strategy, the same in both worlds; that seed is derived from the run seed (stream `pipeline.seed`), never the market seed itself, so that a strategy cannot regenerate the market it trades on from it.
- **Garbage truth.** A pipeline without an oracle must give identical orders when the truth is replaced by unrelated random numbers.
- **Label audit.** A provider that claims to be honest may not correlate above 0.99 (pooled uncentred correlation) with `m` on the audit sample.

## Canaries

Under `tests/canaries/` (not in the package; the loader skips modules whose names start with an underscore):

| Canary | Leak | Guard that must catch it |
|---|---|---|
| D1 `peek_next_close_forecast` | reads the next bar's close through a market handle that a helper module opens on first use and memoizes with `functools.lru_cache` | replay |
| D1b `peek_by_path` | opens the market's contract files by path and reads the next close | replay |
| D2 `full_sample_covariance` | a risk model that estimates on the whole sample in its constructor | replay |
| D3 `unlabelled_oracle` | returns a near copy of `m` with `oracle` false | label (and garbage truth) |

A self-test shows why the purge matters: with the purge switched off, D1's memoized handle keeps the first world's market and D1 passes the replay guard.

## The audit table

`python -m dynamic_trading_engine audit` runs every guard on the canaries and on every Tier-1 built-in strategy (the experiment grid at its default configuration, the oracle rungs, and every Tier-1 forecast provider, risk model, and optimizer inside a one-component strategy; the Tier-2 strategies `WDRO` and `GP_AIM` and the export forecast provider were added after the audit table and are not in it) at 30 sampled instants of `tiny` with a seed used by nothing else (7919, so that no file outside the audit's temporary worlds holds that market's future) and writes `experiments/results/audit_table.csv` (committed). The test suite replays the canaries at 30 instants and every other subject at 3 and compares every cell with the committed table. Oracle rows are `n/a` for the garbage and label guards (they read the truth by design and are labelled bounds).

| subject | oracle | replay | garbage truth | label | caught after (instants) |
|---|---|---|---|---|---|
| canary:D2 | false | caught | passed | passed | 1 |
| canary:D1b | false | caught | passed | passed | 1 |
| canary:D1 | false | caught | passed | passed | 1 |
| canary:D3 | false | passed | caught | caught | - |
| forecast:noisy_oracle | true | passed | n/a | n/a | - |
| forecast:none | false | passed | passed | passed | - |
| forecast:oracle | true | passed | n/a | n/a | - |
| forecast:plain | false | passed | passed | passed | - |
| grid:CASH | false | passed | passed | passed | - |
| grid:CVAR | false | passed | passed | passed | - |
| grid:EW | false | passed | passed | passed | - |
| grid:MINVAR | false | passed | passed | passed | - |
| grid:MPC | false | passed | passed | passed | - |
| grid:MV | false | passed | passed | passed | - |
| grid:MV_NC | false | passed | passed | passed | - |
| grid:MV_noisy_0.25 | true | passed | n/a | n/a | - |
| grid:MV_noisy_0.5 | true | passed | n/a | n/a | - |
| grid:MV_noisy_0.75 | true | passed | n/a | n/a | - |
| grid:MV_oracle | true | passed | n/a | n/a | - |
| grid:RANK_LS | false | passed | passed | passed | - |
| grid:ROBUST | false | passed | passed | passed | - |
| optimizer:cvar | false | passed | passed | passed | - |
| optimizer:equal_weight | false | passed | passed | passed | - |
| optimizer:mean_variance | false | passed | passed | passed | - |
| optimizer:min_variance | false | passed | passed | passed | - |
| optimizer:mpc | false | passed | passed | passed | - |
| optimizer:rank_ls | false | passed | passed | passed | - |
| optimizer:robust_box | false | passed | passed | passed | - |
| optimizer:robust_ell | false | passed | passed | passed | - |
| risk:ewma | false | passed | passed | passed | - |
| risk:lw | false | passed | passed | passed | - |
| risk:pca | false | passed | passed | passed | - |
| risk:sample | false | passed | passed | passed | - |

Every canary was caught by its guard after the first sampled instant; no Tier-1 built-in strategy raised an alarm.

## What the audit cannot prove

A pass is evidence, not proof. The audit samples 30 of the 40 decision instants of one small market and one seed; a leak that only matters at instants it did not sample, or that changes orders only through a quantity the poison leaves intact (for example a future value that equals its past value in both worlds), would pass. The label guard only sees forecasts that are almost exact copies of `m`; a partial leak with a correlation below 0.99 is not flagged by it (the replay and garbage guards remain). The purge cannot reach state kept outside Python modules (files written by the strategy itself, environment variables, other processes). "Read by path" is caught only for reads that resolve through the registry (`DTE_MARKET_DIR`): a strategy that opens a fixed path holding the same market's future in both worlds, or that regenerates the market from a configuration and a seed it finds or guesses, would submit identical orders in both worlds and pass every guard; the audit uses a market seed that nothing else writes, and strategies receive a derived seed, which closes the trivial versions of these paths, not every version.
