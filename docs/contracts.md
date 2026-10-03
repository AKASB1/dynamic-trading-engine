# Contracts

This repository follows **quant-contracts version 1**, a language-neutral contract shared with a separate research substrate. No code is shared; the two sides meet only through the file formats and definitions below. This page restates everything a reader of this repository needs. Extensions of this project and the places where it departs from the contract are listed at the end.

## 1. Conventions

**Time.** Every instant is UTC. In files: `YYYY-MM-DDTHH:MM:SSZ` (seconds precision, `Z` suffix). In memory: `datetime64[us]` (int64 microseconds since the epoch). A bar covers `[ts_open, ts_event)`; its open, high, low, close, and volume are final at `ts_event`. `ts_avail` is the earliest instant at which a user may know a row (`>= ts_event` for bars). Calendar `equity_daily`: Monday to Friday except a fixed holiday rule (here: 1 January and 25 December when they fall on a weekday), sessions `14:30:00Z` to `21:00:00Z` (no daylight-saving handling). The number of bars per year `ppy` is declared in the configuration (252 for `equity_daily`), never inferred. A day count is seconds / 86400; an annual rate accrues as `rate * days / 365`.

**Randomness.** One stream per component: `stream(seed, name)` is a NumPy `Generator` over `PCG64DXSM` with `c = fnv1a64(name)`, `s1 = splitmix64(seed XOR c)`, `s2 = splitmix64(s1 XOR 0x9E3779B97F4A7C15 XOR c)`, state `(s1 << 64) | s2`, increment `((6364136223846793005 << 64) | 1442695040888963407) | 1`. Stream names used here: `market.<component>`, `forecast.<name>` (`forecast.noise.<instrument_id>`), `exec.<component>`, `tune.<strategy>`, `audit.poison`, `audit.instants`, `experiment.<id>`, `bootstrap`. Adding a component never changes another component's numbers; a market depends only on its configuration and seed (common random numbers).

**Numbers.** Prices, money, and costs are float64 in the quote currency; quantities are float64 shares (rounded toward zero to `lot_size` when `lot_size > 0`); returns are decimals; 1 bp = 1e-4 of the notional; annual rates in bp per year. In files: shortest round-trip floats (`0.5`, `101.53045`, `2.0`); `volume`, `adv_shares`, `quantity`, `lot_size`, `horizon_bars`, `vintage` without a decimal point when integral; `NaN`, `inf`, `-0.0` never appear; a missing value is an empty field where the schema allows it.

**Determinism.** Same configuration, data, and seed give identical bytes except `wall_` columns and the manifest fields for time, host, and load. Ties break by `instrument_id` (byte order), then time, then input row order. Reductions run on arrays in a fixed order with one BLAS thread (`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`). A result that depends on a solver records the solver name, version, and tolerances and is reproducible byte for byte only on that stack, the platform's math library included (with the same package versions, the quick results of this repository on Linux differ from those on Windows in the last digits: by about 1e-3 relative per run, at most 3.4e-3 in one column of one run).

**Files.** CSV: UTF-8 without BOM, LF (CRLF accepted on read), comma separated, one exact header, no quoting, no padding. A header-only file is a valid empty dataset; a zero-byte file is invalid. A loader rejects, naming the 1-based line number: a missing or different header, a wrong field count, an unparsable or out-of-range value, an empty required field, an unsorted or duplicated key, and any violated rule of the schema. Beside `<name>.csv` lies `<name>.manifest.json` with `schema_version` (1), `qc_version` (1), `generator` (name, version, parameters), `seed`, `row_count`, `content_sha256`; a reader checks the versions and the hash. JSON is UTF-8 with sorted keys and two-space indentation. A **configuration hash** is the SHA-256 of the canonical JSON (sorted keys, no whitespace, integers without a decimal point, shortest round-trip floats, every field equal to its default omitted).

## 2. Schemas (version 1)

| File | Header | Sorted by | Rules |
|---|---|---|---|
| `instruments_v1.csv` | `instrument_id,symbol,asset_class,currency,ts_list,ts_delist,delist_return,lot_size,tick_size,sector` | `instrument_id` | `asset_class` in equity, crypto, index, other; `currency` uppercase letters; `ts_delist` empty or the `ts_event` of the last bar (not before `ts_list`); `delist_return` empty exactly when `ts_delist` is, else >= -1; `lot_size`, `tick_size` >= 0 |
| `bars_v1.csv` | `instrument_id,ts_open,ts_event,ts_avail,open,high,low,close,volume` | (`instrument_id`, `ts_event`) | `ts_open < ts_event <= ts_avail`; no overlap; prices > 0; `high >= max(open, close, low)`; `low <= min(open, close, high)`; `volume >= 0` (0 = no trading); raw prices (not adjusted) |
| `corporate_actions_v1.csv` | `instrument_id,action,ts_ex,ts_avail,value` | (`instrument_id`, `ts_ex`, `action`) | `split` (new shares per old share) or `cash_dividend` (cash per pre-split share); `ts_avail <= ts_ex`; the first bar with `ts_open >= ts_ex` is quoted ex the action; a dividend is credited at that bar's `ts_event` |
| `series_v1.csv` | `series_id,ts_event,ts_avail,vintage,value` | (`series_id`, `ts_event`, `vintage`) | vintages start at 0 and grow by one per period; `ts_avail` grows strictly with the vintage; `ts_avail >= ts_event`; as of `t` a period's value is its largest vintage with `ts_avail <= t` |
| `returns_v1.csv` | `ts_event,ts_avail,instrument_id,ret` | (`ts_event`, `instrument_id`) | `ret = (close_t * ratio_t + dividend_t) / close_{t-1} - 1`; the first bar has none; `ts_avail` is the bar's |
| `signals_v1.csv` | `ts_event,ts_avail,instrument_id,name,value` | (`ts_event`, `name`, `instrument_id`) | finite value |
| `forecasts_v1.csv` | `ts_event,ts_avail,instrument_id,horizon_bars,mu,sigma` | (`ts_event`, `instrument_id`, `horizon_bars`) | `mu` = expected simple return over the next `horizon_bars` bars; `sigma` empty or >= 0 |
| `liquidity_v1.csv` | `ts_event,ts_avail,instrument_id,adv_shares,sigma_bar` | (`ts_event`, `instrument_id`) | `adv_shares > 0`, `sigma_bar >= 0`; the only inputs of the impact model |
| `orders_v1.csv` | `order_id,ts_submit,instrument_id,quantity,order_type,limit_price,tif,strategy_id` | (`ts_submit`, `order_id`) | signed nonzero quantity (buy positive); `market` (empty `limit_price`) or `limit`; `tif` day or gtc |
| `fills_v1.csv` | `fill_id,order_id,ts_fill,instrument_id,quantity,ref_price,price,spread_cost,impact_cost,commission` | (`ts_fill`, `fill_id`) | `ref_price` = reference price `m`; costs >= 0; a forced delisting exit has `order_id` `DELIST` and zero costs |
| `positions_v1.csv` | `ts_event,instrument_id,quantity,mark_price,value` | (`ts_event`, `instrument_id`) | one row per nonzero position at each bar's end; `value = quantity * mark_price` |
| `equity_v1.csv` | `ts_event,cash,position_value,equity,hold_pnl,trade_pnl,spread_cost,impact_cost,commission,borrow,financing,income` | `ts_event` | one row per bar of the engine calendar from the first (the opening row: initial cash, flows 0); `equity = cash + position_value`; the identity of section 5 from row to row |

The golden fixtures of the contract live in `tests/data/golden/` and pass these loaders; reading and writing them again gives identical bytes.

## 3. Knowledge time, decisions, fills

- **Knowledge.** A row is known at `t` iff `ts_avail <= t`. An instrument row is known from `ts_list`; its `ts_delist` and `delist_return` read as empty before `ts_delist`. A corporate action is known from its announcement `ts_avail`. Everything used at decision instant `t` is a function of rows known at `t`.
- **Point-in-time adjustment.** The split-adjusted close as of `t` divides the raw closes before a split's ex-bar by its ratio, only for splits with `ts_ex <= t`. The total-return series as of `t` is the cumulative product of the returns known at `t`, scaled so that its last value equals the last split-adjusted close.
- **Universe at `t`.** Every instrument with `ts_list <= t` and (`ts_delist` empty or `ts_delist > t`), including those that delist later.
- **Decision.** A function of rows known at `t` and of the engine's state after all fills up to `t`; orders are submitted at `t`; a decision never sees a fill of an order submitted at the same instant.
- **Fills.** An order submitted at the end of bar `k` fills at the earliest in bar `k + 1` of its instrument, at the reference price `m` = the open of that bar (`next_open`, the default for `equity_daily`). A market order fills in full unless the participation cap binds (filled quantity at most `participation_cap * volume` of the fill bar; the rest of a `day` order is cancelled). Nothing fills in a zero-volume bar. Quantities are rounded toward zero to `lot_size`. A same-bar fill model does not exist in this repository.
- **Shorts and leverage.** Shorts only where the book allows them. Gross leverage = gross position value / equity. Orders that would take the gross exposure, at the fill's reference prices, above `max_leverage` by more than `1e-6` are scaled down proportionally and logged (here: to `max(max_leverage, gross before the orders)`).
- **Order of events in a bar.** (1) splits rescale the opening quantity (`q * ratio`), the previous close (`P_prev / ratio`), and pending orders (`* ratio`); (2) dividends on the opening quantity before the split, credited at the close; (3) fills; (4) in the last bar of a delisting instrument, any position is closed at `close * (1 + delist_return)` (`DELIST`, no costs); (5) accruals; (6) mark to the close.

## 4. Cost model v1

Defaults: `commission_bps` 1.0, `commission_per_share` 0.0, `min_commission` 0.0, `half_spread_bps` 2.0, impact `sqrt` with `y` 0.5, `borrow_bps_annual` 50, `financing_bps_annual` 100, `cash_rate_bps_annual` 0, `participation_cap` 0.1. For a fill of signed quantity `q` at reference price `m`, with the decision-time `sigma` (`sigma_bar`) and `V` (`adv_shares`):

- `spread_cost = |q| m half_spread_bps / 1e4`
- `impact_bps = 1e4 y sigma sqrt(|q| / V)` (`sqrt`), `1e4 y sigma |q| / V` (`linear`), 0 (`none`); 0 and counted as `impact_unavailable` when `sigma` or `V` is unknown
- `impact_cost = |q| m impact_bps / 1e4`; `price = m (1 + sign(q) (half_spread_bps + impact_bps) / 1e4)`
- `commission = max(min_commission, |q| m commission_bps / 1e4 + |q| commission_per_share)`
- accruals at each bar's close over `dt` days: `borrow = sum_short |q| P_prev borrow_bps / 1e4 dt / 365`; `financing = max(0, -cash_prev) financing_bps / 1e4 dt / 365`; interest income `max(0, cash_prev) cash_rate_bps / 1e4 dt / 365`.
- Impact never reads the fill bar's volume or range; the participation cap does (it is a physical limit, not a cost).
- **Convex form for the optimizer**, for a trade `z_i` in units of equity `E`: `c(z) = sum_i a_i |z_i| + b_i |z_i|^1.5` with `a_i = (half_spread_bps + commission_bps) / 1e4 + commission_per_share / P_i` and `b_i = y sigma_i sqrt(E / (V_i P_i))`; `min_commission` and lots are not modelled. With noise off, no cap, `min_commission` 0, and no lots, `E * c(z)` equals the loop's charged cost and the execution simulator's single-interval cost (the three-way parity test).

## 5. Accounting

`E_t = cash_t + sum_i q_{i,t} P_{i,t}` (marks at the close; no external flows). Per bar, exactly:

`E_t - E_{t-1} = hold_pnl + trade_pnl - spread_cost - impact_cost - commission - borrow - financing + income`

with `hold_pnl = sum_i q_{i,t-1}^adj (P_{i,t} - P_{i,t-1}^adj)` (after the split adjustment of the bar), `trade_pnl = sum_fills q_f (P_{i,t} - m_f)`, `income` = dividends received minus dividends paid on shorts plus interest. The identity holds to `1e-9 * max(1, |E_{t-1}|)` per bar and cumulatively; every run asserts it in memory, and an independent validator recomputes `hold_pnl`, `trade_pnl`, and `cash` from the bars, the corporate actions, and the fills. Per-bar net return `r_t = (E_t - E_{t-1}) / E_{t-1}`; turnover of a bar `sum |q_f| m_f / E_{t-1}`; gross exposure `sum |q_i P_i| / E`, net `sum q_i P_i / E`. Attribution by instrument and by long and short book sums to the totals.

## 6. Metrics and inference

Per bar with `ppy`: annualized return `(E_T / E_0)^(ppy / T) - 1`; volatility `std(r, ddof=1) sqrt(ppy)`; Sharpe `mean(r) / std(r, ddof=1) sqrt(ppy)`; Sortino with the downside root mean square over all bars; maximum drawdown against the running peak including `E_0`; historical `CVaR_95` = minus the mean of the `ceil(0.05 T)` smallest returns; skewness and raw kurtosis with `1/T` normalization; PSR, the expected maximum Sharpe `SR0` over `N` trials with variance `V`, and the deflated Sharpe ratio `DSR = PSR(SR0)` (Bailey and Lopez de Prado); `trials.csv` with `study_id,trial_id,config_hash,sr_bar,t_bars,skew,kurt`.

The **declared criterion** of this project (its own, not part of the contract): the annualized certainty equivalent of the net per-bar returns after the first decision, `ce_ann = ppy * (mean(r) - (gamma_ce / 2) * var(r, ddof=1))`, `gamma_ce = 6`.

## 7. Synthetic market (meaning of the parameters)

`r_{i,t} = beta_i' f_t + e_{i,t} + c_t` with a market factor and sector factors, a constant premium `c_t`, a two-state volatility regime `v_t` (calm 1, stress `stress_vol_mult`), Gaussian tails, a latent signal `s_{i,t} = phi s_{i,t-1} + sqrt(1 - phi^2) u_{i,t}`, the idiosyncratic return `e_{i,t} = sigma_i v_t (ic s_{i,t-1} + sqrt(1 - ic^2) z_{i,t})`, the conditional mean `m_{i,t} = ic sigma_i v_{t+1} s_{i,t}`, the published proxy `signal_x = (s + observable_noise w) / sqrt(1 + observable_noise^2)`, an overnight share `gap_share` of the variance and of each mean, events (delistings, splits 2:1 or 3:1, quarterly dividends, late listings), and a distribution shift `{at_bar, vol_mult, corr_mult, ic_mult, mu_shift_annual}`. The generator of this repository is its own implementation (see `docs/data.md`); its streams are not byte-compatible with any other implementation.

## 8. Wording

Every result is "simulated", states its data (synthetic, generator configuration and seed), cost model, schedule, tuning, number of trials, and number of seeds, and is phrased "in this simulation, under these assumptions". No claim of profitability, alpha, or real performance. An oracle result is a bound and is labelled `ORACLE`.

## Extensions of this project

- **`decisions.csv`** (one row per decision): `ts_decision,strategy_id,forecast_id,oracle,risk_model,solver_status,objective,n_trades,turnover,wall_solve_ms,hold`.
- **Permanent impact** in the execution simulator: the mid moves by `gamma_ac * q_k` per executed signed quantity (default 0, which is the contract's model).
- **Direct linear temporary impact** `h_k = eta_ac |q_k| / tau_k` for the Almgren-Chriss reference, next to the contract's `sqrt` and `linear` models.
- **Liquidity function** (the contract leaves the estimator to the producer): `adv_shares` = mean volume of the last 20 bars (pre-split volumes scaled by the split ratio); `sigma_bar` = square root of the weighted mean of the squared returns of the last 250 bars with weight `0.5^(age/20)`; a row exists once 20 returns are known.

## Departures from the contract

- Deflated Sharpe ratio in the main comparison: the registered trials are the eight strategies of the table on each seed (`N` = 8), not every configuration evaluated during tuning, because the tuning trials ran on disjoint tuning seeds.
- Leverage rule detail: when the gross exposure before the orders already exceeds `max_leverage` (an overnight move), the orders are scaled so that the gross does not exceed the gross before the orders.

## Export consumer (Tier 2)

`market/export.py` reads an export directory of the contract's export format (version 1): `manifest.json` and `instruments.csv`, `bars.csv`, `corporate_actions.csv`, `returns.csv`, `signals.csv`, `forecasts.csv`, `liquidity.csv` (each with its own manifest; `series.csv` optional). Every listed file is checked against the hash in `manifest.json` and in its own manifest, loaded with the schema's loader, and checked for the knowledge rule (no row available before its event); a `derived_only` export must carry header-only bars. `signals.csv` rows named `signal_x` become the `signal_x` series; `forecasts.csv` (horizon 1) feeds the `export` forecast provider (the latest forecast available at the decision instant). `write_export` writes such a directory from a market of this repository; a loop run on the export equals the run on the in-memory market (tested), and a tampered file is refused.
