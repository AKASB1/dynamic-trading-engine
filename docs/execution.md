# Execution simulator

A separate simulator for one parent order executed over `K` intervals (`src/dynamic_trading_engine/execution/simulator.py`). The rolling loop does not use it (the loop fills at the next bar's open with the contract's costs); the two are tied by the three-way cost parity test. Everything is simulated with assumed parameters.

## Model

- **Parent order.** Side (`+1` buy, `-1` sell), quantity `Q`, arrival mid `S_0`, intervals of `tau_k` bars (sum `T`), an expected volume profile `v_k` (U-shaped for intraday intervals, flat for daily ones; sums to `ADV * T`), and the decision-time `sigma_bar` and `ADV`.
- **Mid price.** `S_k = S_{k-1} + sigma_abs sqrt(tau_k) xi_k + gamma_ac q_k` with `sigma_abs = sigma_bar S_0` per bar, `xi` from the stream `exec.price`, and `q_k` the signed executed quantity (permanent impact `gamma_ac`, an extension of the contract; default 0).
- **Volume.** `V_k = v_k exp(s_v z_k - s_v^2 / 2)` with `z` from `exec.volume` and `s_v` = `volume_noise_sd` (default 0.3).
- **Execution price.** `p_k = S_{k-1} + side (half_spread + h_k)`, `half_spread = S_0 half_spread_bps / 1e4`, temporary impact `h_k = S_0 y sigma_bar (|q_k| / (tau_k ADV))^beta` (`beta` 0.5 for the contract's `sqrt` model, 1 for `linear`) or `eta_ac |q_k| / tau_k` (direct linear form for the Almgren-Chriss reference). Commission as in the contract with the reference price `S_{k-1}`. Impact uses the decision-time `sigma_bar` and `ADV`, never the realized volume.
- **Fills.** The policy's desired quantity for an interval plus the carried shortfall executes up to `participation_cap * V_k` (default 0.1) and up to the remaining quantity; the shortfall is carried (option, default on); what is left after the last interval is not executed (the opportunity term). Latency `d`: a quantity decided at the start of interval `k` reaches the market in interval `k + d`; policies do not plan for latency (a slice that would arrive after the last interval is lost).
- **Policies.** A policy sees only the state at the start of an interval (index, remaining quantity net of fills and of orders in flight, earlier prices and volumes, the expected profile). `market` (everything in the first interval), `twap` (`Q/K`), `vwap` (`Q v_k / sum v`, the expected profile), `pov(rate)` (a rate order: the venue executes `rate * V_k` of the concurrent volume from interval `d` on until complete; the policy never sees that volume), `ac(lambda_ac)` (the static Almgren-Chriss schedule below). A test perturbs the realized prices and volumes of the current and later intervals and checks that no earlier decision changes.

## Implementation shortfall and its exact decomposition

`IS = side [sum_f |q_f| (p_f - S_0) + (Q - Q_filled)(S_end - S_0)] + commission` (positive is a cost), and per run exactly:

- `timing = side sum_f |q_f| RW_{f-1}` (the random-walk innovations before the fill's interval),
- `permanent = gamma_ac sum_f q_f sum_{j before f} q_j`,
- `spread = sum_f |q_f| half_spread`, `temporary = sum_f |q_f| h_f`,
- `opportunity = side (Q - Q_filled)(S_end - S_0)`, `commission`.

The components sum to `IS` within `1e-12 Q S_0` on every run (1000 random runs over all policies, buys and sells, both impact models, spread, commission, latency, cap, volume noise, and permanent impact; the vectorized simulator also equals the path-by-path reference). Reported: `IS` in bps of `Q S_0`, the completion rate, and the slippage against the market's interval VWAP (volume-weighted average of the interval mid prices).

## The Almgren-Chriss reference (R8)

Discrete time, zero fixed cost. `tau = T / N`, `eta_tilde = eta_ac - gamma_ac tau / 2`, `kappa_tilde^2 = lambda_ac sigma_ac^2 / eta_tilde`, `kappa_ac = (2/tau) asinh(kappa_tilde tau / 2)` (exact for tiny `lambda_ac`; at `kappa_ac` = 0 the TWAP schedule is returned). Holdings after interval `j`: `x_j = X sinh(kappa_ac (T - t_j)) / sinh(kappa_ac T)`; trades `n_j = x_{j-1} - x_j`; `E_ac = gamma_ac X^2 / 2 + (eta_tilde / tau) sum n_j^2`; `V_ac = sigma_ac^2 tau sum_{j=1}^{N-1} x_j^2`; the schedule minimizes `E_ac + lambda_ac V_ac`.

| Known answer | X | T | N | sigma | gamma | eta | lambda | kappa | E + lambda V |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1e6 | 5 | 5 | 0.95 | 2.5e-7 | 2.5e-6 | 1e-6 | 0.607076163247063 | 1212855.55836193 |
| 2 | 1 | 1 | 4 | 0.3 | 0.1 | 0.5 | 2 | 0.607060859211631 | 0.575561755385669 |

Checks: both known answers (kappa, holdings, `E_ac`, `V_ac`, the objective) to `1e-9` relative; a CVXPY quadratic program (scaled units, tolerances 1e-10) and a direct tridiagonal linear solve agree with the closed form to `1e-6`; as `lambda_ac -> 1e-14` the schedule is TWAP to `1e-5`; TWAP's objective exceeds the optimum's; the first trade grows with `lambda_ac`. In the simulator's AC configuration (arithmetic price, direct linear permanent and temporary impact, no spread, commission, volume noise, or cap) with 20000 paths and common random numbers: the mean shortfall is within 3 standard errors of `E_ac`, the sample variance within 5 percent of `V_ac`, the timing term has mean zero within 3 standard errors, and the permanent and temporary terms equal `gamma_ac (X^2 - sum n_j^2) / 2` and `(eta_ac / tau) sum n_j^2` to `1e-9` on every path.

In S7(b) the `ac` policy runs in a square-root impact world; its schedule uses a linear `eta` matched to the square-root impact at the TWAP slice size (`eta = h_sqrt(Q/K) tau / (Q/K)`), so it is guaranteed optimal only in the exact world of S7(a).

## Cost parity

With noise off, no cap, `min_commission` 0, and no lots, the optimizer's `E c(z)` (the CVXPY expression evaluated at `z`), the cost the loop's fill charges at the reference price, and the shortfall of the `market` policy with one interval of one bar and `beta` 0.5 agree to `1e-10` relative on 200 random trades.

## Tier 2: adaptive execution and the volume-forecast error study

`execution/adaptive.py`: at the start of every interval the policy re-plans the remaining quantity (net of fills and of orders in flight) with the static Almgren-Chriss schedule over the remaining intervals, using a volume level estimated from the realized volumes of the earlier intervals (`sum V_j / sum v_j`; 1 before any observation) and a linear impact matched to the square-root model at the remaining TWAP slice with that volume (the true `eta_ac` in the exact world); the slice is capped at the participation cap of the expected, rescaled volume. A variant re-plans without the volume estimate. Checks: in the exact Almgren-Chriss world without anything to learn it reproduces the static schedule (time consistency, 1e-9); a perturbation of the current and later intervals changes no earlier decision; the decomposition stays exact. The simulator gained two options for the study, both off by default (the Tier-1 results are reproduced bit for bit): a persistent per-path volume level `exp(s_l z_l - s_l^2/2)` (stream `exec.volume_level`) and a realized volume shape that differs from the expected profile the policies plan with.
