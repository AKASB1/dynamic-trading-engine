"""The poisoned copy of a market at instant t (stream ``audit.poison``).

Everything known at t stays as it is; everything after t changes: bars with ``ts_avail > t``
get prices multiplied by random factors in [0.1, 10] (open, high, low, close of a bar by the
same factor, so high and low still bound) and redrawn volumes; ``signal_x`` rows after t are
redrawn; periods that have a vintage known at t get later vintages appended; corporate actions
announced after t are changed, removed, or added; delistings after t are cancelled for some
instruments (their bars continue) and added for others; instruments that list after t are
removed and a new one is added; the generator seed and parameters named in the manifest change.
Returns and liquidity inputs are recomputed from the poisoned bars (pure causal functions, so
their rows up to t are bit-identical). The truth arrays after t are redrawn, except ``m`` at t
(it contains the next bar's regime, read by design by the labelled oracles) and the constants.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

from dynamic_trading_engine.market.data import NO_TS, Market, SeriesStore, derive
from dynamic_trading_engine.market.truth import Truth
from dynamic_trading_engine.rng import stream


def poison_seed(seed: int, k: int) -> int:
    return seed * 1_000_003 + k


def poison(market: Market, truth: Truth, k: int, seed: int) -> tuple[Market, Truth]:
    """Poisoned copy at the close of bar ``k`` (t = ts_event[k])."""
    g = stream(poison_seed(seed, k), "audit.poison")
    t = int(market.ts_event[k])
    T, n = market.n_bars, market.n
    m = copy.deepcopy(market)
    after = np.arange(T) > k
    known_bar = m.bar_avail <= t
    # --- delisting changes (instruments whose delisting, if any, is unknown at t)
    alive_after = (m.ts_list <= int(m.ts_event[-1])) & ~(
        (m.ts_delist != NO_TS) & (m.ts_delist <= t)
    )
    cand = [j for j in range(n) if alive_after[j] and m.ts_list[j] <= t]
    g.shuffle(cand)
    cancel = [j for j in cand if m.ts_delist[j] != NO_TS][:2]
    add = [j for j in cand if m.ts_delist[j] == NO_TS][:2]
    for j in cancel:  # bars continue to the end with a random walk from the last close
        last = int(np.nonzero(m.bar_avail[:, j] != NO_TS)[0][-1])
        p = m.close[last, j]
        for u in range(last + 1, T):
            p = p * float(np.exp(g.normal(0.0, 0.02)))
            m.open[u, j] = m.close[u, j] = p
            m.high[u, j], m.low[u, j] = p * 1.01, p * 0.99
            m.volume[u, j] = float(np.floor(g.uniform(1e5, 1e6)))
            m.bar_avail[u, j] = int(m.ts_event[u])
        m.ts_delist[j] = NO_TS
        m.delist_return[j] = np.nan
    for j in add:
        d = int(g.integers(k + 1, T))
        m.ts_delist[j] = int(m.ts_event[d])
        m.delist_return[j] = float(g.uniform(-0.9, 0.2))
        for a in ("open", "high", "low", "close", "volume"):
            getattr(m, a)[d + 1 :, j] = np.nan
        m.bar_avail[d + 1 :, j] = NO_TS
    # --- prices and volumes of the bars after t
    unknown = (m.bar_avail != NO_TS) & ~known_bar
    fac = np.exp(g.uniform(np.log(0.1), np.log(10.0), (T, n)))
    for a in ("open", "high", "low", "close"):
        arr = getattr(m, a)
        arr[unknown] = arr[unknown] * fac[unknown]
    vol = np.floor(g.uniform(1e4, 5e6, (T, n)))
    m.volume[unknown] = vol[unknown]
    # --- signal_x: rows after t redrawn; later vintages for periods known at t
    sx = m.series_x
    x0 = sx.value[0].copy()
    xa = sx.avail[0].copy()
    redraw = (xa != NO_TS) & (xa > t)
    x0[redraw] = g.standard_normal(int(redraw.sum()))
    vals, avs = [x0], [xa]
    for v in range(1, len(sx.value)):
        vv, aa = sx.value[v].copy(), sx.avail[v].copy()
        later = (aa != NO_TS) & (aa > t)
        vv[later] = g.standard_normal(int(later.sum()))
        vals.append(vv)
        avs.append(aa)
    known_period = (xa != NO_TS) & (xa <= t)
    rows = np.nonzero(known_period[max(0, k - 30) : k + 1])
    v_new = np.full((T, n), np.nan)
    a_new = np.full((T, n), NO_TS, dtype=np.int64)
    pick = g.random(len(rows[0])) < 0.3
    for r, c in zip(rows[0][pick], rows[1][pick]):
        u = max(0, k - 30) + int(r)
        if len(vals) == 1:
            v_new[u, c] = float(g.standard_normal())
            a_new[u, c] = int(m.ts_event[min(T - 1, k + 1 + int(g.integers(0, 5)))])
    if np.any(a_new != NO_TS):
        vals.append(v_new)
        avs.append(a_new)
    m.series_x = SeriesStore("signal_x", vals, avs)
    # --- corporate actions announced after t: changed, removed, added
    ca = []
    for row in m.corporate_actions:
        if row[3] <= t:
            ca.append(row)
            continue
        u = g.random()
        if u < 0.3:
            continue  # removed
        if u < 0.7:
            row = (row[0], row[1], row[2], row[3], float(row[4] * g.uniform(0.5, 2.0)))
        ca.append(row)
    for j in g.choice(n, size=min(2, n), replace=False):
        has = np.nonzero((m.bar_avail[:, j] != NO_TS) & after)[0]
        if len(has) < 2:
            continue
        ex = int(has[int(g.integers(1, len(has)))])
        ann = int(g.integers(k + 1, ex))  # announced after t and before the ex instant
        if g.random() < 0.5:
            ca.append((m.ids[j], "split", int(m.ts_open[ex]), int(m.ts_event[ann]), 2.0))
        else:
            ca.append((m.ids[j], "cash_dividend", int(m.ts_open[ex]), int(m.ts_event[ann]), 0.01))
    # one action per kind and ex instant
    seen, uniq = set(), []
    for row in sorted(ca, key=lambda r: (r[0], r[2], r[1])):
        key = (row[0], row[1], row[2])
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    m.corporate_actions = uniq
    # --- instruments that list after t: removed; one new instrument added
    keep = [j for j in range(n) if m.ts_list[j] <= t]
    m = _select(m, keep)
    m = _add_instrument(m, g, k)
    m.generator = {
        "name": market.generator.get("name", "x"),
        "version": market.generator.get("version", 1),
        "params": {"poisoned": True, "noise": float(g.random())},
    }
    m.seed = int(g.integers(10**6, 10**7))
    derive(m)
    tr = _poison_truth(truth, market, m, k, g)
    return m, tr


def _select(m: Market, keep: list[int]) -> Market:
    kw = {}
    for f in dataclasses.fields(m):
        v = getattr(m, f.name)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == m.n:
            v = v[:, keep]
        elif (
            isinstance(v, np.ndarray)
            and v.ndim == 1
            and v.shape[0] == m.n
            and f.name not in ("ts_open", "ts_event")
        ):
            v = v[keep]
        elif isinstance(v, tuple) and len(v) == m.n and f.name != "ids" or f.name == "ids":
            v = tuple(v[j] for j in keep)
        kw[f.name] = v
    ids = set(kw["ids"])
    kw["corporate_actions"] = [r for r in m.corporate_actions if r[0] in ids]
    sx = m.series_x
    kw["series_x"] = SeriesStore(
        "signal_x", [a[:, keep] for a in sx.value], [a[:, keep] for a in sx.avail]
    )
    return Market(**kw)


def _add_instrument(m: Market, g, k: int) -> Market:
    T = m.n_bars
    start = int(g.integers(k + 1, T)) if k + 1 < T else T - 1
    iid = "Z" + str(int(g.integers(100, 999)))
    p = float(g.uniform(20, 200))
    col = {a: np.full(T, np.nan) for a in ("open", "high", "low", "close", "volume")}
    av = np.full(T, NO_TS, dtype=np.int64)
    for u in range(start, T):
        p *= float(np.exp(g.normal(0, 0.02)))
        col["open"][u] = col["close"][u] = p
        col["high"][u], col["low"][u] = p * 1.01, p * 0.99
        col["volume"][u] = float(np.floor(g.uniform(1e5, 1e6)))
        av[u] = int(m.ts_event[u])
    order = sorted(list(m.ids) + [iid])
    pos = order.index(iid)

    def ins(a, v):
        return np.insert(a, pos, v, axis=1 if np.ndim(a) == 2 else 0)

    kw = {f.name: getattr(m, f.name) for f in dataclasses.fields(m)}
    for a in ("open", "high", "low", "close", "volume"):
        kw[a] = ins(kw[a], col[a])
    kw["bar_avail"] = ins(kw["bar_avail"], av)
    kw["ids"] = tuple(order)
    kw["ts_list"] = ins(m.ts_list, int(m.ts_event[start]))
    kw["ts_delist"] = ins(m.ts_delist, NO_TS)
    kw["delist_return"] = ins(m.delist_return, np.nan)
    kw["lot_size"] = ins(m.lot_size, 1.0)
    kw["tick_size"] = ins(m.tick_size, 0.01)
    for name, v in (
        ("sector", "S0"),
        ("symbol", iid),
        ("asset_class", "equity"),
        ("currency", "USD"),
    ):
        lst = list(getattr(m, name))
        lst.insert(pos, v)
        kw[name] = tuple(lst)
    sx = m.series_x
    xv = np.where(av != NO_TS, g.standard_normal(T), np.nan)
    vals = [ins(sx.value[0], xv)] + [ins(a, np.full(T, np.nan)) for a in sx.value[1:]]
    avs = [ins(sx.avail[0], av)] + [ins(a, np.full(T, NO_TS, dtype=np.int64)) for a in sx.avail[1:]]
    kw["series_x"] = SeriesStore("signal_x", vals, avs)
    for a in ("split_ratio", "dividend", "returns", "ret_avail", "adv", "sigma_bar", "liq_avail"):
        kw[a] = None
    return Market(**kw)


def _poison_truth(tr: Truth, real: Market, pm: Market, k: int, g) -> Truth:
    T = real.n_bars
    old = {iid: j for j, iid in enumerate(tr.ids)}
    n2 = pm.n

    def remap(a, fill):
        out = np.empty((a.shape[0], n2))
        for c, iid in enumerate(pm.ids):
            out[:, c] = a[:, old[iid]] if iid in old else fill(a.shape[0])
        return out

    s = remap(tr.s, g.standard_normal)
    m_ = remap(tr.m, lambda L: g.standard_normal(L) * 1e-3)
    e = remap(tr.e, lambda L: g.standard_normal(L) * 1e-2)
    r = remap(tr.r, lambda L: g.standard_normal(L) * 1e-2)
    s[k + 1 :] = g.standard_normal((T - k - 1, n2))
    m_[k + 1 :] = g.standard_normal((T - k - 1, n2)) * 1e-3  # m[k] (the instant t) is kept
    e[k + 1 :] = g.standard_normal((T - k - 1, n2)) * 1e-2
    r[k + 1 :] = g.standard_normal((T - k - 1, n2)) * 1e-2
    v = tr.v.copy()
    v[k + 2 :] = g.uniform(1, 3, len(v) - k - 2)

    def remap1(a):
        return np.array([a[old[i]] if i in old else float(np.mean(a)) for i in pm.ids])

    B = np.array([tr.B[old[i]] if i in old else tr.B.mean(axis=0) for i in pm.ids]).reshape(
        n2, tr.B.shape[1]
    )
    return Truth(
        ids=pm.ids,
        phi=tr.phi,
        s=s,
        m=m_,
        v=v,
        vm=tr.vm.copy(),
        ic_t=tr.ic_t.copy(),
        c=tr.c.copy(),
        B=B,
        fvar=tr.fvar.copy(),
        sig_pre=remap1(tr.sig_pre),
        sig_post=remap1(tr.sig_post),
        lsc_post=remap1(tr.lsc_post),
        at_bar=tr.at_bar,
        r=r,
        e=e,
    )


def garbage_truth(tr: Truth, seed: int) -> Truth:
    """A truth of the same shapes filled with unrelated random numbers."""
    g = stream(seed, "audit.garbage")

    def rnd(a):
        return g.standard_normal(np.shape(a)) * (np.std(a) + 1e-3)

    return Truth(
        ids=tr.ids,
        phi=float(g.uniform(0, 1)),
        s=rnd(tr.s),
        m=rnd(tr.m),
        v=np.abs(rnd(tr.v)) + 1,
        vm=tr.vm.copy(),
        ic_t=tr.ic_t.copy(),
        c=tr.c.copy(),
        B=rnd(tr.B),
        fvar=np.abs(rnd(tr.fvar)),
        sig_pre=np.abs(rnd(tr.sig_pre)),
        sig_post=np.abs(rnd(tr.sig_post)),
        lsc_post=tr.lsc_post.copy(),
        at_bar=tr.at_bar,
        r=rnd(tr.r),
        e=rnd(tr.e),
    )
