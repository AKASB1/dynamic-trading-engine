"""The truth of a generated market: latent signal, conditional mean, regime, loadings, covariance.

Only the generator, the oracle forecast providers, the audit, the experiment evaluators, the
strategy factory, the tuner and runner, the entry points, and the tests import this module
(a static import test enforces it). The truth is never written to a contract file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Truth:
    ids: tuple[str, ...]  # instrument ids, the column order of every array
    phi: float  # persistence of the latent signal (the oracles' decay)
    s: np.ndarray  # (T, n) latent signal
    m: np.ndarray  # (T, n) m[t] = conditional mean of the idiosyncratic return of bar t+1
    v: np.ndarray  # (T+1,) regime multiplier per bar
    vm: np.ndarray  # (T+1,) shift volatility multiplier per bar
    ic_t: np.ndarray  # (T+1,) information coefficient per bar
    c: np.ndarray  # (T+1,) premium per bar
    B: np.ndarray  # (n, K) factor loadings
    fvar: np.ndarray  # (K,) calm per-bar factor variances
    sig_pre: np.ndarray  # (n,) calm per-bar idiosyncratic vol before the shift
    sig_post: np.ndarray  # (n,) after the shift (equal to sig_pre without a shift)
    lsc_post: np.ndarray  # (n,) loading scale of each instrument from at_bar on (1 without a shift)
    at_bar: int  # first bar of the shift (T+1 when none)
    r: np.ndarray  # (T, n) drawn simple total returns (NaN where no bar or first bar)
    e: np.ndarray  # (T, n) idiosyncratic part of the drawn returns (row 0 is 0)

    def idio_vol(self, t: int) -> np.ndarray:
        return self.sig_pre if t < self.at_bar else self.sig_post

    def cov(self, t: int, idx: np.ndarray | None = None) -> np.ndarray:
        """Covariance of the returns of bar ``t`` given the information of bar ``t-1``."""
        lsc = self.lsc_post if t >= self.at_bar else np.ones(len(self.sig_pre))
        B = self.B * lsc[:, None]
        sig = self.idio_vol(t)
        if idx is not None:
            B, sig = B[idx], sig[idx]
        scale = (self.vm[t] * self.v[t]) ** 2
        fac = (B * self.fvar) @ B.T
        return scale * (fac + np.diag(sig**2 * (1.0 - self.ic_t[t] ** 2)))

    def cond_mean(self, t: int) -> np.ndarray:
        """Expected return of bar ``t`` given the information of bar ``t-1`` (t >= 1)."""
        return self.c[t] + self.m[t - 1]
