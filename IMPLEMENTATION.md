# Implementation

Simulation only: a synthetic market, no broker, no live data, no investment advice.

## Boundary with quant-research-engine

The two projects share a written contract (file formats, knowledge-time rules, the cost model, accounting, metrics, and the meaning of the synthetic market's parameters; `docs/contracts.md`) and no code. This project reads and writes the contract's files, generates its own synthetic market, and estimates its own risk models and liquidity inputs. A consumer of the contract's export format (`market/export.py`: every file checked against its schema and hash, and the knowledge rule) is implemented and tested on exports written by this repository; no export of the research engine was available to run it on (`README.md`, TODO).

## Portfolio optimizer

A family of convex problems built once with CVXPY parameters (DPP) and re-solved: `equal_weight`, `rank_ls` (naive baseline), `min_variance`, `mean_variance`, `cvar` (Rockafellar-Uryasev inside the problem), `robust_box` and `robust_ell` (worst case over a box or an ellipsoid around the forecast), the controller `mpc`, and two Tier-2 strategies: `wdro` (mean-CVaR in the worst case over a Wasserstein ball around the scenarios) and `gp_aim` (the multi-asset Garleanu-Pedersen trading rule, projected onto the book). Per holding period of `k` bars:

```text
maximize    mu_h' w - (gamma/2) w' (k Sigma) w - cost_scale * c(w - w0)      (mean_variance)
            mu_h' w - eta_cvar * CVaR_alpha(holding-period loss) - cost_scale * c(w - w0)   (cvar)
            mean_variance - kappa_rob * sum_i se_h,i |w_i|   (robust_box)
            mean_variance - kappa_rob * ||diag(se_h) w||_2    (robust_ell)
subject to  the book (LO: sum w = 1, 0 <= w_i <= max(0.10, 2/n);
                      LS: |sum w| <= 0.02, sum |w_i| <= 2, |w_i| <= 0.10)
            optional: turnover, sector net exposure, market-beta band, liquidity per instrument
c(z) = sum_i a_i |z_i| + b_i |z_i|^1.5   (the contract's cost model in units of equity)
```

Open-source solvers only (Clarabel by default; OSQP and SCS as second solvers in the agreement tests); the objective is scaled by 1e4; every solve passes `warm_start=False`; a solve that is not `optimal`/`optimal_inaccurate` or that violates a constraint by more than 1e-6 is a hold. Gurobi is not used. Details, conventions, and the exact references: `docs/optimization.md`.

## Execution simulator

One parent order over `K` intervals: arithmetic mid-price walk with optional permanent impact, temporary impact (the contract's square-root or linear model, or a direct linear form), half spread, commission, a participation cap on noisy realized volume, partial fills with carry, latency, and the opportunity cost of what is left. Policies `market`, `twap`, `vwap`, `pov(rate)`, `ac(lambda)`; an exact implementation-shortfall decomposition; the Almgren-Chriss reference in closed form, as a quadratic program, and as a linear solve. Market orders and next-open fills only (no limit orders, no queue model). `docs/execution.md`.

## Policy interface

Two kinds of policies. Portfolio strategies map a `DecisionState` (the knowledge-bounded view at the decision instant) to target weights (`engine/types.py`: `decide(state) -> Decision`); the built-in one composes a forecast provider, a risk model, an optimizer or the controller, and a book. Execution policies map the state known at the start of an interval (index, remaining quantity, earlier prices and volumes, the expected profile) to a desired quantity or a participation rate. An adaptive execution policy (`execution/adaptive.py`, Tier 2) re-plans the Almgren-Chriss schedule of the remaining quantity at every interval, with the expected profile rescaled by the volume observed so far.

## Delivery order

- [x] portfolio state and knowledge-bounded decision state
- [x] contract formats: loaders, validators, writers, golden fixtures
- [x] synthetic market with planted signal, regimes, events, and a distribution shift
- [x] order/fill model and the rolling loop with exact accounting and an independent validator
- [x] forecast providers and risk models
- [x] convex optimizers with costs and constraints, exact references R1-R5
- [x] multi-period control: controller, Garleanu-Pedersen closed form, grid dynamic programs (R6, R7)
- [x] execution simulator, baseline policies, Almgren-Chriss reference (R8)
- [x] information audit with canaries
- [x] protocol: registry, equal-budget tuning, frozen configurations, experiments S1-S8 and SENS
- [x] Wasserstein distributionally robust mean-CVaR (Tier 2)
- [x] multi-asset Garleanu-Pedersen rule (Tier 2)
- [x] adaptive execution policy (Tier 2)
- [x] solver comparison, fat tails and jumps, cost misspecification, capacity, export consumer (Tier 2)
- [ ] consuming an export of the research engine (none was available)
- the paper-trading adapter of the original plan is out of scope (no credentials, no broker connectivity)

## Decision problem

At each decision instant `t` (the close of a decision bar, every `rebalance_every` bars after the warmup): build the state from rows with `ts_avail <= t`; select the optimized instruments (at least 60 known bars, a known close and return at `t`); forecast `mu`, `se`, `decay`; estimate `Sigma` over the common window; solve; convert target weights to whole-lot market orders submitted at the close; they fill at the next bar's open with the contract's costs. Leverage is a constraint of the book (the gross bound), never a source of return; no experiment varies it.

## Evaluation and acceptance

Data: the generator's `base` market (30 instruments, 1260 daily bars) and its variants; the first decision comes after a warmup of 260 bars, the estimators use the last 250 known bars (the window `W`, tuned per strategy where its search space includes it), and each seed is an independent market. One declared criterion (`ce_ann`, the annualized certainty equivalent with `gamma_ce = 6`), equal tuning effort (16 random-search trials per strategy on tuning seeds 100-103), frozen configurations (`experiments/tuned/frozen.json`), disjoint evaluation seeds (1000 and up), paired differences with intervals. `docs/protocol.md`. One command per stage: `tune`, `freeze`, `experiment` (`README.md`).

## Cross-project contracts

This repository follows quant-contracts version 1 as restated in `docs/contracts.md`, with its extensions (`decisions.csv`, permanent impact, the direct linear impact form, the liquidity estimator) and departures (the trial set of the deflated Sharpe ratio, the leverage-scaling detail) listed there.

## Checkpoint

Built and tested: everything in the delivery-order list marked done, with one protocol tuning run and one full evaluation (Tier 1) and one run of every Tier-2 study. Not done: the item marked open, a run of the GitHub workflow (the tests and README commands ran once on Linux in an independent check; two platform-dependent test failures found there are fixed but were not re-run on Linux), the audit of the Tier-2 strategies, limit orders. See the README's `## TODO`.
