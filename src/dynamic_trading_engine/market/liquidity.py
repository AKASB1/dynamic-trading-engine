"""Liquidity inputs (QC 2.8): one documented pure function of the bars known at the row's instant.

- ``adv_shares`` at bar ``t``: the mean volume of the last 20 bars up to and including ``t``,
  in post-split shares as of ``t`` (the volume of a bar before a split in the window is
  multiplied by the ratios of the splits between that bar and ``t``).
- ``sigma_bar`` at bar ``t``: the square root of the weighted mean of the squared simple
  total returns (QC 2.5) of the last 250 bars up to and including ``t``, weight ``0.5^(age/20)``
  with age 0 for the newest; fewer bars when fewer are known.
- A row exists once 20 returns are known (and ``adv_shares > 0``); its ``ts_avail`` is the
  bar's ``ts_avail``.

Every row is computed from its own window only, so the rows up to ``t`` do not change when
bars after ``t`` change (the replay audit relies on it, bit for bit).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ADV_WINDOW = 20
SIGMA_WINDOW = 250
SIGMA_HALF_LIFE = 20.0
MIN_RETURNS = 20


def liquidity(volume, bar_avail, returns, split_ratio, no_ts):
    T, n = volume.shape
    adv = np.full((T, n), np.nan)
    sig = np.full((T, n), np.nan)
    avail = np.full((T, n), no_ts, dtype=np.int64)
    ages = np.arange(SIGMA_WINDOW - 1, -1, -1, dtype=float)  # window position -> age
    w_full = 0.5 ** (ages / SIGMA_HALF_LIFE)
    for j in range(n):
        r = returns[:, j]
        has_r = ~np.isnan(r)
        cnt = np.cumsum(has_r)
        # split-adjusted volume in shares of bar t: vol_u * S_t / S_u, S = cumulative ratio
        ratio = split_ratio[:, j]
        S = np.cumprod(ratio)
        vol = np.where(np.isnan(volume[:, j]), 0.0, volume[:, j])
        vpad = np.concatenate([np.zeros(ADV_WINDOW - 1), vol / S])
        vwin = sliding_window_view(vpad, ADV_WINDOW)  # (T, 20)
        adv_j = vwin.sum(axis=1) / ADV_WINDOW * S
        r2 = np.where(has_r, r, 0.0) ** 2
        rpad = np.concatenate([np.zeros(SIGMA_WINDOW - 1), r2])
        mpad = np.concatenate([np.zeros(SIGMA_WINDOW - 1), has_r.astype(float)])
        rwin = sliding_window_view(rpad, SIGMA_WINDOW)
        mwin = sliding_window_view(mpad, SIGMA_WINDOW)
        wm = mwin * w_full
        num = (rwin * wm).sum(axis=1)
        den = wm.sum(axis=1)
        ok = (cnt >= MIN_RETURNS) & (bar_avail[:, j] != no_ts) & (adv_j > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.sqrt(num / den)
        adv[:, j] = np.where(ok, adv_j, np.nan)
        sig[:, j] = np.where(ok, s, np.nan)
        avail[:, j] = np.where(ok, bar_avail[:, j], no_ts)
    return adv, sig, avail
