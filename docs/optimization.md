# Optimization: formulations, conventions, exact references

Every formulation here is a convex program solved with CVXPY on open-source solvers (Clarabel by default). Each is checked against a closed form, an optimality certificate, or an independent solver (`tests/test_optimizers.py`, `tests/test_multiperiod.py`, `tests/test_execution.py`). Everything is simulation only.

## Units and conventions

- `E` equity at the decision instant `t`; `w` target weights (value / `E`; cash weight `1 - sum(w)`); `w0` the current weights at the closes of `t`; `z = w - w0` the trade; `k` the number of bars between decisions (`rebalance_every`, default 5).
- Per holding period: `mu_h = sum_{j=0}^{k-1} decay^j mu` (the provider's per-bar forecast and decay), covariance `k * Sigma`, trading cost charged once.
- **Cost** (the contract's convex form, units of equity): `c(z) = sum_i a_i |z_i| + b_i |z_i|^1.5` with `a_i = (half_spread_bps + commission_bps) / 1e4 + commission_per_share / P_i` and `b_i = y * sigma_i * sqrt(E / (V_i * P_i))` (`sigma_i`, `V_i` the decision-time liquidity inputs, `P_i` the close at `t`). With the contract's `linear` impact the cost form is quadratic: `a_i |z_i| + 0.5 * l_i * z_i^2` with `l_i = 2 y sigma_i E / (V_i P_i)`. `min_commission` and lots are not modelled. `cost_scale` multiplies `c` (1: the parity setting, 0: cost-ignorant). Unknown liquidity gives zero impact (as in the simulator).
- **Scaled units**: every objective is multiplied by `SCALE = 1e4` so that its optimum is of order 1 (solver tolerances are absolute). Reported objectives are divided back.
- **DPP**: every problem is built once per (kind, ladder size, book, cost form, optional constraints, horizon, discount) with CVXPY parameters and re-solved; `problem.is_dpp()` is asserted. Products of parameters are passed as one parameter: `F` is a factor of `SCALE * (gamma/2) * k * Sigma` (risk term `sum_squares(F @ w)`; the eigen factor of the floored covariance), `ca = SCALE * cost_scale * a`, `cb = SCALE * cost_scale * b`, `kse = SCALE * kappa_rob * se_h`, `pe = SCALE * eta_cvar / ((1 - alpha) S)`. The trade is a variable `z` with `z == w - w0`. The 1.5 power is `cp.power(cp.abs(z), 1.5, approx=False)` (exact power cone).
- **Slots**: the instruments listed at `t` in `instrument_id` order, padded to the next size of the ladder 8, 16, 32, 64, 128, 256. Instruments with fewer than `min_history` (60) known bars, or without a known close or return at `t`, are not optimized: their bounds are 0 (so a position in them is sold) and their forecast and risk rows are 0. The slot layout depends only on what is listed at `t`.
- **Warm starts**: every solve passes `warm_start=False`; solving A, B, A on one cached object returns identical bits for A (tested for every (book, optimizer) pair).

## Books and constraints

| Book | Budget | Positions | Gross | Net | Gross bound (leverage rule uses 1.05x) |
|---|---|---|---|---|---|
| `LO` | `sum(w) = 1` | `0 <= w_i <= max(0.10, 2/n)` | (implied 1) | | 1 |
| `LS` | none | `\|w_i\| <= 0.10` | `sum \|w_i\| <= 2` | `-0.02 <= sum(w) <= 0.02` | 2 |
| `BUDGET` (references R1, R2) | `sum(w) = 1` | free | | | |
| `UNC` (references R3, R6, R7) | none | free | | | |

Optional constraints: turnover `sum |z_i| <= turn_max`; sector net `|sum_{i in s} w_i| <= s_max` per sector (sector rows of a parameter matrix, unused rows zero); a market-beta band `lo <= beta' w <= hi` with `beta` the regression of each instrument on the equal-weighted return of the optimized instruments over the common window; a liquidity bound `|z_i| E <= part V_i P_i`. Leverage is a constraint of the book, never varied by an experiment.

## The optimizer family

- `cash`: a reference that never holds a position (its `ce_ann` is exactly 0).
- `equal_weight`: `1/n` over the optimized instruments (LO), no solve.
- `rank_ls`: `r_i` the cross-sectional rank of `mu` scaled to [-1, 1] (centred), `w ~ sign(r_i) |r_i|^p`, scaled to gross 2 and clipped to the position bound; no risk model, no cost term.
- `min_variance`: minimize `w' Sigma w` under the book's constraints (the mean-variance problem object with `mu = 0` and zero cost coefficients).
- `mean_variance`: maximize `mu_h' w - (gamma/2) w' (k Sigma) w - cost_scale * c(z)`.
- `cvar`: maximize `mu_h' w - eta_cvar * CVaR_alpha - cost_scale * c(z)`, with the Rockafellar-Uryasev program inside: scenario returns `R` (S x n) are sums of `k` consecutive known bar returns over the last `h <= 250` bars of the common window (`S = h - k + 1`, overlapping), column means removed; `CVaR_alpha = zeta + 1/((1 - alpha) S) sum_s u_s` with `u_s >= 0`, `u_s >= -R_s' w - zeta`. The problem has 250 fixed scenario rows; an unused row is switched off by a parameter (`u_s >= -R_s' w - act_s zeta` with `act_s = 0`) and carries a weight of 1e-3, so that `u_s = 0` there and the objective is unchanged (a zero-weight free direction made the interior-point solver stall).
- `robust_box`: subtract `kappa_rob * sum_i se_h,i |w_i|`, the worst case of `mu' w` over the box `|mu_i - mu_h,i| <= kappa_rob se_h,i` with `se_h = se * sum_j decay^j`; `robust_ell`: subtract `kappa_rob * ||diag(se_h) w||_2` (the ellipsoid). Both keep the variance term.
- `mpc`: variables `w_1 .. w_H` (`w_0` the current weights), step `j` with `mu_j = decay^((j-1) k) mu_h`, covariance `k Sigma`, cost `c(w_j - w_{j-1})`, objective `sum_j delta^(j-1) [mu_j' w_j - (gamma/2) w_j' (k Sigma) w_j - cost_scale c(w_j - w_{j-1})]`, the book's constraints at every step; the first action is taken. `delta` is a constant of the cached problem.
- **Routing**: `mpc` with `H = 1` and `robust` with `kappa_rob = 0` are routed to the `mean_variance` problem object, so they are bit-identical to it (tested); the non-routed robust object with a vanishing radius agrees to the solver tolerance.

## Solutions and failures

Every solve returns the weights, the status, the objective, the solver name and version, the iterations, a `wall_` solve time (from a clock injected by the runner), and the largest constraint violation, measured on the returned weights (inequalities: the excess; equalities: the absolute residual). A solve succeeds if its status is `optimal` or `optimal_inaccurate` and the violation is at most `1e-6`; anything else is a failure, and the strategy holds (no orders at that decision). Statuses are counted separately in every manifest.

Clarabel can stop early ("insufficient progress") on power-cone instances where many `|z_i|` sit at 0 (a no-trade optimum). A deterministic fallback ladder is applied per solve: default settings, then `max_step_fraction` 0.9 and 0.8, then the same problem with the 1.5 power written through second-order cones (CVXPY's rational form, exact for `p = 3/2`). Each solution records which step solved it. Non-finite weights are a failure; a CVaR problem without any complete holding-period scenario in the common window (fewer common bars than `k`) is a failure ("no data"), not an unbounded solve. `mpc` with `H = 1` reaches the `mean_variance` object for every discount (the discount exists only for the controller).

## Risk models

Covariance of the per-bar simple total returns over the common window (the last `h = min(W, ...)` bars in which every optimized instrument has a known return): `sample` (ddof 1); `ewma` (weights `0.5^(age/hl)` normalized to sum 1, weighted mean removed); `lw` (Ledoit and Wolf 2004: `S = X'X/h` of the demeaned returns, `m = tr(S)/n`, `d^2 = ||S - m I||^2`, `b_bar^2 = h^-2 sum_k ||x_k x_k' - S||^2`, `b^2 = min(b_bar^2, d^2)`, intensity `b^2/d^2` in [0, 1], `Sigma = (b^2/d^2) m I + (1 - b^2/d^2) S`, with `||A||^2 = tr(AA')/n`); `pca` (`n_pc` = 3 principal components of the sample covariance plus the diagonal of the residual variances). Every model floors the smallest eigenvalue at `1e-10 * trace / n`.

## Forecast providers

`none` (0); `plain`: `mu_i = shrink * b_t * sigma_i * x_i` with `x_i` the latest `signal_x`, `sigma_i` the latest `sigma_bar`, and `b_t` the slope of the pooled regression through the origin of `r_{i,u+1} / sigma_{i,u}` on `x_{i,u}` over the last 250 bars in which both are known at `t`; `se_i = SE(b_t) sigma_i |x_i|`; `decay` = the uncentred lag-1 autocorrelation of `x` pooled over instruments, floored at 0; `noisy_oracle(rho)` and `oracle` read the truth and are labelled bounds (`docs/information.md`).

## Exact references

| Ref | Statement | Check (test) |
|---|---|---|
| R1 | budget equality only, no cost: `w = (1/gamma) Sigma^-1 (mu - nu 1)`, `nu = (1' Sigma^-1 mu - gamma) / (1' Sigma^-1 1)` | 20 random problems, `1e-6` |
| R2 | minimum variance, budget only: `w = Sigma^-1 1 / (1' Sigma^-1 1)` | 20 random problems, `1e-6` |
| R3 | one period, quadratic costs, `a = 0`, no constraint: `w = (gamma Sigma + Lq)^-1 (mu + Lq w0)` | 20 random problems, `1e-6` |
| R4 | Rockafellar-Uryasev value equals the CVaR definition (mean of the worst `(1-alpha)S` losses with the fractional weight); the optimum equals a grid search for 2 and 3 instruments; CVaR nondecreasing in `alpha` | `1e-12` (no solve); grid within its resolution |
| R5 | `kappa_rob = 0` equals mean-variance; the optimum equals the nominal objective minus the penalty; no feasible weight has a better worst case (inner minimum in closed form); the weighted norm of the solution is nonincreasing in `kappa_rob` | solver tolerance |
| R6 | Garleanu-Pedersen: `A` the positive root of `delta A^2 + (g + Lam(1-delta)) A - Lam g = 0`, `B = Lam / (g + Lam + delta A - Lam delta (1-phi_gp))`, `a = (g + delta A)/(g + Lam + delta A)`, `aim = (1 + delta B (1-phi_gp))/(g + delta A)`; known answer (`g` 0.08, `Lam` 0.4, `rho_gp` 0.02, `phi_gp` 0.3): `A` = 0.14129787310950273, `B` = 1.1625476588156523, `a` = 0.3532446827737568, `aim` = 8.227637353852478 | `1e-12`; quadratic residual `1e-15`; grid DP within one grid step; controller `H = 60`, `delta` 0.98 to `1e-6` relative |
| R7 | one asset, proportional cost: grid DP with a no-trade band that widens with `c_prop` and closes as `c_prop -> 0` (`x = mu / g`) | band properties; the controller's gap is a result (S6) |
| R8 | Almgren-Chriss (see `docs/execution.md`) | known answers `1e-9`; QP and linear solve `1e-6` |

Solver agreement (check 6): on 50 random problems of at most 10 instruments, in scaled units, Clarabel (tolerances 1e-10) is checked against SciPy SLSQP (absolute values split into positive and negative parts, constraints written by hand) and a second CVXPY solver (OSQP for the quadratic cost form, SCS for both), with the tolerance `1e-6 * max(1, |objective|)`: (a) SLSQP started away from the optimum never returns a point that is feasible to 1e-8 (violation measured by the test, not taken from SLSQP) with a higher objective than Clarabel's optimum; only this side is checked, because a local solver can stop short of the optimum (on one platform SLSQP reported success 5.7 percent below it); (b) SLSQP started at Clarabel's solution finds no improvement (a local optimum of a convex problem is global); (c) the second solver agrees in both directions. At least 45 of the 50 SLSQP points must pass the feasibility gate (all 50 do on the build machine). An optimizer that returns a feasible point 1e-4 below the optimum fails (a), (b), and (c) on nearly every problem; the KKT conditions of the quadratic programs hold to `1e-6` with the dual variables.

## Multi-period references (R6, R7)

The grid dynamic program (`policies/dp.py`, NumPy only) works on a position grid (241 points on [-0.8, 0.8]) and an expected-return grid (61 points on +-5 stationary standard deviations), with Gauss-Hermite quadrature (9 nodes) for the next expected return and linear interpolation (linear extrapolation at the edges) in `mu`; modified policy iteration until the value changes by less than 1e-11. The single-asset controller exists twice: the cached CVXPY `mpc` problem with one active slot (book `UNC`), and a linear-algebra version for the unconstrained quadratic case (a tridiagonal system; the first action is `alpha_H xp + beta_H mu`), which agree to `1e-8` on 20 random states.

## Tier 2: Wasserstein-robust mean-CVaR (`wdro`)

The worst case of `E[-w'xi] + eta_cvar * CVaR_alpha(-w'xi)` over every distribution within 1-Wasserstein distance `eps_w` (ground norm l2 on R^n, support R^n) of the empirical distribution of the holding-period scenarios `xi_i = mu_h + R_i` (the scenarios of `cvar`, shifted by the forecast). The loss is the maximum of two affine functions of `xi` (`l_1 = -w'xi + eta tau`, `l_2 = -(1 + eta/(1-alpha)) w'xi + eta tau (1 - 1/(1-alpha))`), and by the result of Mohajerin Esfahani and Kuhn (2018) for piecewise-affine losses the worst case is the convex program `min lam eps + (1/S) sum_i s_i` with `s_i >= l_k(xi_i)` and `lam >= (1 + eta/(1-alpha)) ||w||_2`; the optimizer maximizes minus that value minus the trading cost under the book (`optimization/wdro.py`, `optimization/tier2.py`). Checks (`tests/test_tier2.py`): with `eps_w = 0` it equals the empirical mean-CVaR, which is the `cvar` optimizer with the forecast multiplied by `1 + eta_cvar` (CVaR is translation equivariant; weights to 1e-4, objective to 1e-6); with `eta_cvar = 0` (a linear loss) the optimum equals `max mu_h'w - eps ||w||_2 - c(z)` solved directly (the empirical expectation plus `eps` times the dual norm); the optimal objective is nonincreasing in `eps_w` (the worst-case loss nondecreasing).

## Tier 2: the multi-asset Garleanu-Pedersen rule (`gp_aim`)

Per decision step of `k` bars, with the risk matrix `gamma k Sigma`, a quadratic trading cost `(lam_gp/2) dx' (k Sigma) dx` proportional to the risk matrix, and the holding-period forecast decaying by `decay^k` per step, the coordinates `y = (k Sigma)^(1/2) x` decouple the problem into scalar problems of R6 (`g = gamma`, `Lam = lam_gp`, `phi_gp = 1 - decay^k`, `rho_gp = 1 - delta_gp`), so the policy is `x = (1 - a) x_prev + a * aim * (k Sigma)^(-1) mu_h` (trade the fraction `a` of the way to the aim portfolio; `policies/gp_aim.py`). `lam_gp = lam_mult * calibrated_lam`, where `calibrated_lam` matches `(lam/2) z' (k Sigma) z` to the contract's convex cost of a 2 percent trade on average over the instruments. The target is projected onto the book (the closest feasible weights in the Euclidean norm, on the cached mean-variance object). Checks: with a diagonal covariance each instrument follows the scalar rule of R6 exactly; the projected weights satisfy the book.
