"""Synthetic market generator (the meaning of its parameters follows the contract, section 7).

``generate(config, seed)`` returns the market (contract datasets) and, separately, its
``Truth``. All random draws come from named streams of the seed; the draws do not depend on
the parameters, so ``shift`` and ``base`` of one seed share every draw and are identical
before ``at_bar``. Details and every assumed number: docs/data.md.
"""

from __future__ import annotations

import math

import numpy as np

from dynamic_trading_engine.market.calendar import equity_daily
from dynamic_trading_engine.market.config import MarketConfig
from dynamic_trading_engine.market.data import NO_TS, Market, SeriesStore, derive
from dynamic_trading_engine.market.truth import Truth
from dynamic_trading_engine.rng import stream

GENERATOR_NAME = "dte-synthetic"
GENERATOR_VERSION = 1
RETURN_FLOOR = -0.9
DIVIDEND_PERIOD = 63
DIVIDEND_NOTICE = 15
SPLIT_NOTICE = 10
MIN_BARS_BEFORE_EVENT = 15
MIN_BARS_BEFORE_DELIST = 30
CORR_CAP_SHARE = 0.95


def _avg_corr(B, fvar, sig2, scale2):
    """Average pairwise correlation of the unconditional calm covariance with loadings scaled
    by sqrt(scale2) per instrument and idiosyncratic variances sig2."""
    Bs = B * np.sqrt(scale2)[:, None]
    C = (Bs * fvar) @ Bs.T + np.diag(sig2)
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    iu = np.triu_indices(len(d), 1)
    return float(R[iu].mean()) if len(iu[0]) else 0.0


def _shift_arrays(cfg: MarketConfig, B, fvar, sig_pre, T):
    """Per-bar shift parameters. The correlation shift rescales each instrument's loadings by
    sqrt(min(k, cap_i)) and lowers its idiosyncratic variance so that its total variance is
    unchanged (then everything is multiplied by vol_mult); cap_i keeps at least 5 percent of
    the variance idiosyncratic (correlations stay below 1), and k is found by bisection so that
    the average pairwise correlation is multiplied by corr_mult."""
    v_len = T + 1
    vm = np.ones(v_len)
    ic_t = np.full(v_len, cfg.ic)
    c = np.full(v_len, cfg.premium_annual / cfg.ppy)
    sig_post = sig_pre.copy()
    lsc_post = np.ones(len(sig_pre))
    at = v_len
    sh = cfg.shift
    if sh is not None:
        at = sh.at_bar
        if B.shape[1]:
            fac_var = (B**2 * fvar).sum(axis=1)
            total = fac_var + sig_pre**2
            cap = CORR_CAP_SHARE * total / fac_var
            base = _avg_corr(B, fvar, sig_pre**2, np.ones(len(total)))
            target = sh.corr_mult * base

            def scale2(k):
                return np.minimum(k, cap)

            def avg(k):
                s2 = scale2(k)
                return _avg_corr(B, fvar, total - s2 * fac_var, s2)

            lo_k, hi_k = 0.0, float(np.max(cap))
            if avg(hi_k) <= target:
                k = hi_k
            else:
                for _ in range(200):
                    mid = 0.5 * (lo_k + hi_k)
                    if avg(mid) < target:
                        lo_k = mid
                    else:
                        hi_k = mid
                k = 0.5 * (lo_k + hi_k)
            s2 = scale2(k)
            lsc_post = np.sqrt(s2)
            sig_post = np.sqrt(total - s2 * fac_var)
        vm[at:] = sh.vol_mult
        ic_t[at:] = cfg.ic * sh.ic_mult
        c[at:] += sh.mu_shift_annual / cfg.ppy
    return vm, ic_t, c, sig_post, lsc_post, at


def generate(cfg: MarketConfig, seed: int) -> tuple[Market, Truth]:
    T, n, ppy = cfg.n_bars, cfg.n_instruments, cfg.ppy
    ts_open, ts_event = equity_daily(cfg.start, T)
    ids = tuple(f"I{j:03d}" for j in range(n))

    # ---- static parameters
    g = stream(seed, "market.static")
    beta_m = g.uniform(cfg.beta_lo, cfg.beta_hi, n)
    perm = g.permutation(n)
    idio_ann = g.uniform(cfg.idio_vol_lo, cfg.idio_vol_hi, n)
    if cfg.equal_idio_vol:
        idio_ann = np.full(n, 0.5 * (cfg.idio_vol_lo + cfg.idio_vol_hi))
    lp_mid = 0.5 * (math.log(cfg.price_lo) + math.log(cfg.price_hi))
    lp_sd = (math.log(cfg.price_hi) - math.log(cfg.price_lo)) / 4
    p0 = np.exp(np.clip(g.normal(lp_mid, lp_sd, n), math.log(cfg.price_lo), math.log(cfg.price_hi)))
    adv0 = np.exp(g.uniform(math.log(cfg.adv_lo), math.log(cfg.adv_hi), n))
    n_sec = cfg.n_sectors
    sector_idx = perm % n_sec
    sector = tuple(f"S{k}" for k in sector_idx)

    if cfg.factors:
        K = 1 + n_sec
        B = np.zeros((n, K))
        B[:, 0] = beta_m
        B[np.arange(n), 1 + sector_idx] = 1.0
        fvar = np.array([cfg.market_vol_annual**2] + [cfg.sector_vol_annual**2] * n_sec) / ppy
    else:
        B = np.zeros((n, 0))
        fvar = np.zeros(0)
    sig_pre = idio_ann / math.sqrt(ppy)

    # ---- regime path (T+1 values: m of the last bar needs the regime of the next one)
    u = stream(seed, "market.regime").random(T + 1)
    v = np.ones(T + 1)
    if cfg.regime:
        state = 0
        for t in range(1, T + 1):
            if state == 0 and u[t] < cfg.p_calm_to_stress:
                state = 1
            elif state == 1 and u[t] < cfg.p_stress_to_calm:
                state = 0
            v[t] = cfg.stress_vol_mult if state else 1.0

    vm, ic_t, c, sig_post, lsc_post, at = _shift_arrays(cfg, B, fvar, sig_pre, T)
    ones_n = np.ones(n)

    # ---- draws (shapes depend only on sizes, never on parameters)
    f = stream(seed, "market.factors").standard_normal((T, max(B.shape[1], 1)))[:, : B.shape[1]]
    us = stream(seed, "market.signal").standard_normal((T, n))
    z = stream(seed, "market.idio").standard_normal((T, n))
    w = stream(seed, "market.observable").standard_normal((T, n))
    om = stream(seed, "market.overnight").standard_normal((T, n))
    rr = stream(seed, "market.range").standard_normal((T, n, 2))
    eta = stream(seed, "market.volume").standard_normal((T, n))
    if cfg.tails == "student_t":
        # unit-variance Student-t: Z * sqrt((nu - 2) / W), W ~ chi^2_nu from streams of their own
        # (the Gaussian draws above are unchanged, so the default market is untouched)
        nu = cfg.t_dof
        f = f * np.sqrt((nu - 2.0) / stream(seed, "market.tails.factors").chisquare(nu, f.shape))
        z = z * np.sqrt((nu - 2.0) / stream(seed, "market.tails.idio").chisquare(nu, z.shape))
    jumps = np.zeros((T, n))
    if cfg.jump_rate_annual > 0:
        gj = stream(seed, "market.jumps")
        hit = gj.random((T, n)) < cfg.jump_rate_annual / ppy
        jumps = np.where(hit, gj.standard_normal((T, n)) * cfg.jump_size, 0.0)

    phi = cfg.persistence
    s = np.empty((T, n))
    s[0] = us[0]
    a_s = math.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + a_s * us[t]

    def sig_at(t):
        return sig_pre if t < at else sig_post

    m = np.empty((T, n))
    for t in range(T):
        m[t] = ic_t[t + 1] * sig_at(t + 1) * vm[t + 1] * v[t + 1] * s[t]

    on = cfg.observable_noise
    x = (s + on * w) / math.sqrt(1.0 + on * on)

    # ---- returns (bar t >= 1), overnight split
    r = np.full((T, n), np.nan)
    e_all = np.zeros((T, n))
    r_on = np.full((T, n), np.nan)
    sd_tot = np.full((T, n), np.nan)
    sqf = np.sqrt(fvar)
    gs = cfg.gap_share
    for t in range(1, T):
        scale = vm[t] * v[t]
        lsc = ones_n if t < at else lsc_post
        fac = lsc * scale * (B @ (sqf * f[t])) if B.shape[1] else 0.0
        ic = ic_t[t]
        sg = sig_at(t)
        e = scale * sg * (ic * s[t - 1] + math.sqrt(1.0 - ic * ic) * z[t])
        if cfg.jump_rate_annual > 0:
            e = e + scale * sg * jumps[t]  # zero-mean jumps (not in the truth's covariance)
        rt = np.maximum(fac + e + c[t], RETURN_FLOOR)
        var = scale**2 * (lsc**2 * (B**2 * fvar).sum(axis=1) + sg**2 * (1.0 - ic * ic))
        sd = np.sqrt(var)
        mu = c[t] + m[t - 1]
        eps = rt - mu
        ron = gs * mu + gs * eps + math.sqrt(gs * (1.0 - gs)) * sd * om[t]
        r[t] = rt
        e_all[t] = e
        r_on[t] = np.maximum(ron, RETURN_FLOOR)
        sd_tot[t] = sd

    # ---- events
    list_bar = np.zeros(n, dtype=int)
    delist_bar = np.full(n, T - 1, dtype=int)
    delisted = np.zeros(n, dtype=bool)
    ge = stream(seed, "market.events")
    u_late = ge.random(n)
    late_bar = ge.integers(20, max(21, T // 2), n)
    u_del = ge.random((T, n))
    del_ret = ge.uniform(cfg.delist_return_lo, cfg.delist_return_hi, n)
    u_split = ge.random((T, n))
    u_ratio = ge.random((T, n))
    div_phase = ge.integers(0, DIVIDEND_PERIOD, n)
    split_events: list[tuple[int, int, int, float]] = []  # (j, ex, announce, ratio)
    div_events: list[tuple[int, int, int]] = []  # (j, ex, announce)
    if cfg.events:
        n_late = int(round(cfg.late_listing_share * n))
        late = np.argsort(u_late, kind="stable")[:n_late]
        list_bar[late] = late_bar[late]
        h = cfg.delist_hazard_annual / ppy
        for j in range(n):
            cand = np.nonzero(u_del[:, j] < h)[0]
            cand = cand[cand >= list_bar[j] + MIN_BARS_BEFORE_DELIST]
            if len(cand):
                delist_bar[j] = int(cand[0])
                delisted[j] = True
        hs = cfg.split_rate_annual / ppy
        for j in range(n):
            lo = list_bar[j] + MIN_BARS_BEFORE_EVENT
            for t in np.nonzero(u_split[:, j] < hs)[0]:
                if lo <= t <= delist_bar[j]:
                    ratio = 2.0 if u_ratio[t, j] < 0.5 else 3.0
                    split_events.append((j, int(t), max(list_bar[j], int(t) - SPLIT_NOTICE), ratio))
            if cfg.dividend_yield_annual > 0:
                for t in range(lo, delist_bar[j] + 1):
                    if (t - div_phase[j]) % DIVIDEND_PERIOD == 0:
                        div_events.append((j, t, t - DIVIDEND_NOTICE))

    has = np.zeros((T, n), dtype=bool)
    for j in range(n):
        has[list_bar[j] : delist_bar[j] + 1, j] = True
    r = np.where(has & np.roll(has, 1, axis=0) & (np.arange(T)[:, None] > 0), r, np.nan)

    ratio_arr = np.ones((T, n))
    for j, t, _a, rt in split_events:
        ratio_arr[t, j] = rt
    div_amt = np.zeros((T, n))
    announce_at: dict[int, list[tuple[int, int]]] = {}
    for j, t, a in div_events:
        announce_at.setdefault(a, []).append((j, t))

    # ---- prices and volume
    close = np.full((T, n), np.nan)
    opn = np.full((T, n), np.nan)
    hi = np.full((T, n), np.nan)
    lo_ = np.full((T, n), np.nan)
    vol = np.full((T, n), np.nan)
    ar, vs = cfg.volume_ar, cfg.volume_sd
    d = eta[0] * vs
    share_mult = np.ones(n)
    for t in range(T):
        if t > 0:
            d = ar * d + math.sqrt(1.0 - ar * ar) * vs * eta[t]
        first = has[t] & (list_bar == t)
        cont = has[t] & (list_bar < t)
        share_mult = np.where(cont, share_mult * ratio_arr[t], share_mult)
        if first.any():
            close[t, first] = p0[first]
            opn[t, first] = p0[first]
        if cont.any():
            prev = close[t - 1]
            dv = div_amt[t]
            rt = ratio_arr[t]
            close[t, cont] = (((1.0 + r[t]) * prev - dv) / rt)[cont]
            opn[t, cont] = (((1.0 + r_on[t]) * prev - dv) / rt)[cont]
        live = first | cont
        sd_id = np.where(np.isnan(sd_tot[t]), sig_pre, sd_tot[t]) * math.sqrt(1.0 - gs)
        hi[t] = np.where(
            live, np.maximum(opn[t], close[t]) * np.exp(0.5 * sd_id * np.abs(rr[t, :, 0])), np.nan
        )
        lo_[t] = np.where(
            live, np.minimum(opn[t], close[t]) * np.exp(-0.5 * sd_id * np.abs(rr[t, :, 1])), np.nan
        )
        vol[t] = np.where(
            live, np.floor(adv0 * share_mult * np.exp(d - 0.5 * vs * vs) + 0.5), np.nan
        )
        for j, ex in announce_at.get(t, []):
            if has[t, j]:
                div_amt[ex, j] = cfg.dividend_yield_annual / 4.0 * close[t, j]

    bar_avail = np.where(has, ts_event[:, None], NO_TS).astype(np.int64)
    x_val = np.where(has, x, np.nan)

    # ---- instruments and corporate actions
    ts_list = ts_event[list_bar].astype(np.int64)
    ts_delist = np.where(delisted, ts_event[delist_bar], NO_TS).astype(np.int64)
    dret = np.where(delisted, del_ret, np.nan)
    ca = []
    for j, t, a, rt in split_events:
        ca.append((ids[j], "split", int(ts_open[t]), int(ts_event[a]), float(rt)))
    for j, t, a in div_events:
        if div_amt[t, j] > 0:
            ca.append(
                (ids[j], "cash_dividend", int(ts_open[t]), int(ts_event[a]), float(div_amt[t, j]))
            )
    ca.sort(key=lambda row: (row[0], row[2], row[1]))

    market = Market(
        name=cfg.name,
        seed=seed,
        generator={
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
            "params": cfg.generator_params(),
        },
        ppy=ppy,
        calendar_start=cfg.start,
        ts_open=ts_open,
        ts_event=ts_event,
        ids=ids,
        ts_list=ts_list,
        ts_delist=ts_delist,
        delist_return=dret,
        lot_size=np.full(n, cfg.lot_size),
        tick_size=np.full(n, 0.01),
        sector=sector,
        symbol=ids,
        asset_class=("equity",) * n,
        currency=("USD",) * n,
        open=opn,
        high=hi,
        low=lo_,
        close=close,
        volume=vol,
        bar_avail=bar_avail,
        corporate_actions=ca,
        series_x=SeriesStore("signal_x", [x_val], [bar_avail.copy()]),
    )
    derive(market)
    truth = Truth(
        ids=ids,
        phi=phi,
        s=s,
        m=m,
        v=v,
        vm=vm,
        lsc_post=lsc_post,
        ic_t=ic_t,
        c=c,
        B=B,
        fvar=fvar,
        sig_pre=sig_pre,
        sig_post=sig_post,
        at_bar=at,
        r=r,
        e=e_all,
    )
    return market, truth
