# The synthetic market

Every number below is an assumed value of this simulation, not an estimate from real data. The generator (`src/dynamic_trading_engine/market/generator.py`) is this repository's own implementation of the contract's market semantics; its random streams are not byte-compatible with any other implementation. Configurations are committed JSON under `configs/market/`; a field that is absent takes the default of `MarketConfig` (`market/config.py`).

## Configurations

| Name | What differs from `base` | Used by |
|---|---|---|
| `base` | 30 instruments, 1260 bars on `equity_daily` from 2018-01-02, `ppy` 252, warmup 260 | tuning, S2, S2b, S3, S4, S6(c), S8, SENS; Tier 2: `WDRO`, `GP_AIM`, cost misspecification, capacity |
| `tiny` | 8 instruments, 300 bars, warmup 100, higher event rates (delisting hazard 0.15 per year, split rate 0.3 per year, a quarter of the instruments list late) so that small runs see events | tests, quick mode, the audit |
| `tiny_shift` | `tiny` with the shift of `shift` at bar 200 | quick S5 |
| `wide` | 100 instruments, 1000 bars | S4 |
| `null` | `ic` = 0 | forecast tests |
| `shift` | a distribution shift at bar 630: `vol_mult` 1.5, `corr_mult` 1.5, `ic_mult` 0.5, `mu_shift_annual` -0.05 | S5; Tier 2: `WDRO` |
| `clean` | no factors, equal idiosyncratic volatility, regime off, no events, no premium | fundamental-law test |
| `stress` | calm-to-stress transition probability 0.03 per bar (stationary stress share 0.375 instead of 0.167) | SENS |

## The return model (defaults of `base`)

- **Calendar.** Monday to Friday except 1 January and 25 December; sessions 14:30Z to 21:00Z.
- **Factors.** One market factor (annual volatility 0.16) and three sector factors (annual volatility 0.10); market betas uniform on 0.6 to 1.4; each instrument loads 1 on one sector (sectors assigned round-robin after a random permutation). Premium `c_t` = 5 percent a year, constant per bar.
- **Idiosyncratic part.** Annual volatility uniform on 0.15 to 0.35 (per bar: divided by sqrt(252)); Gaussian tails.
- **Regime.** Calm (multiplier 1) and stress (multiplier 2) on all volatilities; transition probabilities 0.01 (calm to stress) and 0.05 (stress to calm) per bar; the first bar is calm.
- **Planted signal.** Latent `s_{i,t} = phi s_{i,t-1} + sqrt(1 - phi^2) u_{i,t}` with `phi` 0.9 and unit variance; the idiosyncratic return `e_{i,t} = sigma_i v_t (ic s_{i,t-1} + sqrt(1 - ic^2) z_{i,t})` with `ic` 0.02; the conditional mean of the next bar's idiosyncratic return `m_{i,t} = ic sigma_i v_{t+1} s_{i,t}` (it contains the next bar's regime).
- **Observable proxy.** `signal_x = (s + w) / sqrt(2)` (`observable_noise` 1, correlation 0.71 with `s`), published as the series `<instrument_id>.signal_x` with the bar's `ts_avail`.
- **Total return.** `r_{i,t} = beta_i' f_t + e_{i,t} + c_t`, floored at -0.9 (never binds in practice). The true covariance of bar `t` given the information of bar `t - 1` is `vm_t^2 v_t^2 [B_t diag(f) B_t' + diag(sigma_i^2 (1 - ic^2))]`.
- **Overnight split.** With `mu = c_t + m_{t-1}` and `eps = r - mu`, the overnight return is `r_on = g mu + g eps + sqrt(g (1 - g)) sd_t omega` (`g` = `gap_share` 0.25, `omega` standard normal, `sd_t` the bar's true standard deviation), so the overnight part carries the share `g` of the variance and of each mean; the intraday part is `(1 + r) / (1 + r_on) - 1`. A fill at the next open therefore misses the share 0.25 of the planted mean of that bar.
- **Prices.** Initial prices log-normal around the geometric middle of 20 and 200 (clipped to [20, 200]). `close_t = ((1 + r_t) close_{t-1} - D_t) / ratio_t` and `open_t = ((1 + r_on,t) close_{t-1} - D_t) / ratio_t`, so the returns of the contract's definition equal the drawn returns (to rounding, below 1e-15). High and low extend the open-close range by half-normal multiples of half the intraday volatility.
- **Volume.** Average daily volume log-uniform on 0.5 to 5 million shares per instrument; the log-volume deviation is AR(1) with coefficient 0.7 and standard deviation 0.3; volumes are whole shares and scale with the cumulative split ratio. Lot size 1 share.

## Events

- **Late listings.** A tenth of the instruments (a quarter in `tiny`) list at a bar drawn uniformly from 20 to half the sample; `ts_list` is the `ts_event` of the first bar; the first bar has no return.
- **Delistings.** Hazard 1 percent a year per instrument, at least 30 bars after the listing; `delist_return` uniform on -0.6 to 0.1; `ts_delist` is the `ts_event` of the last bar; no bar after it.
- **Splits.** Rate 3 percent a year (at least 15 bars after the listing), ratio 2 or 3 with equal probability, announced 10 bars before the ex bar (`ts_ex` = the ex bar's `ts_open`).
- **Dividends.** Yield 1.5 percent a year paid quarterly (every 63 bars, a random phase per instrument), announced 15 bars before the ex bar, amount = yield / 4 times the close at the announcement.

## The distribution shift

From bar `at_bar` on: all volatilities times `vol_mult`; each instrument's factor loadings scaled by `sqrt(min(k, cap_i))` and its idiosyncratic variance lowered so that its total variance is unchanged before the volatility multiplier (`cap_i` keeps at least 5 percent of the variance idiosyncratic, so correlations stay below 1), with `k` found by bisection so that the average pairwise correlation is multiplied by `corr_mult` exactly; `ic` times `ic_mult`; `mu_shift_annual / ppy` added to the premium per bar. The shift changes parameters, not draws: `shift` and `base` of one seed use the same draws and are identical before `at_bar` (tested on every bar array). Dividend amounts announced after the shift follow the shifted prices.

## Liquidity inputs

`adv_shares` at bar `t` is the mean volume of the last 20 bars up to `t`; a split inside the window is treated by scaling the volume of the bars before it by the split ratios between that bar and `t` (post-split shares as of `t`). `sigma_bar` is the square root of the weighted mean of the squared returns of the last 250 bars with weight `0.5^(age/20)` (age 0 for the newest), fewer when fewer are known. A row exists once 20 returns are known. Each row depends only on its own window, so rows up to `t` are bit-identical when bars after `t` change.

## Defaults of the experiments

The `base` market; the contract's cost model with its defaults (commission 1 bp, half spread 2 bp, square-root impact with `y` 0.5, borrow 50 bp a year, financing 100 bp a year, no interest on cash, participation cap 0.1); `next_open` fills; the `LS` book; `rebalance_every` 5; `warmup_bars` 260; initial capital 10 million in the quote currency; `ppy` 252.

## Tier 2: fat tails and jumps (assumed tail models)

`tails: student_t` replaces the Gaussian factor and idiosyncratic draws by unit-variance Student-t draws with `t_dof` = 5 degrees of freedom (`Z sqrt((nu - 2) / W)`, `W` chi-squared from the streams `market.tails.*`); `jump_rate_annual` > 0 adds zero-mean idiosyncratic jumps (Bernoulli per bar and instrument at the annual rate divided by `ppy`, size normal with standard deviation `jump_size` = 4 times the bar's idiosyncratic standard deviation; stream `market.jumps`). The truth's covariance does not include the jumps. Configurations: `base_t5`, `shift_t5`, `base_jump`, `shift_jump` (5 jumps per instrument and year). The Gaussian draws are unchanged, so the default markets are identical to before (tested through the recorded hashes of the quick results).
