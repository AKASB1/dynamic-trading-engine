"""Risk models: covariance of the per-bar simple total returns of the optimized instruments,
estimated from the returns known at t over the common window of the decision state.

- ``sample``: sample covariance (ddof 1) over the last h = min(W, common window) bars.
- ``ewma``: weights proportional to ``0.5^(age/hl)`` (age 0 = newest), normalized to sum 1,
  weighted mean removed: ``Sigma = sum_j w_j (r_j - mean_w)(r_j - mean_w)'``.
- ``lw``: Ledoit and Wolf (2004): ``S = X'X / h`` (X demeaned), ``m = tr(S)/n``,
  ``d^2 = ||S - m I||^2``, ``b_bar^2 = h^-2 sum_k ||x_k x_k' - S||^2``, ``b^2 = min(b_bar^2, d^2)``,
  intensity ``delta = b^2 / d^2`` in [0, 1], ``Sigma = delta m I + (1 - delta) S`` (the norm is
  ``||A||^2 = tr(A A') / n``, which cancels in the ratio).
- ``pca``: ``n_pc`` principal components of the sample covariance plus the diagonal of the
  residual variances.

Every model floors the smallest eigenvalue at ``1e-10 * trace / n`` and returns a factor ``F``
with ``F'F = Sigma`` (used by the optimizers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamic_trading_engine.state.view import DecisionState, common_window

FLOOR_REL = 1e-10


@dataclass(frozen=True)
class RiskEstimate:
    sigma: np.ndarray
    factor: np.ndarray  # F with F.T @ F == sigma (up to rounding)
    h: int  # bars used
    info: dict


def floor_psd(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = 0.5 * (S + S.T)
    n = S.shape[0]
    lam, Q = np.linalg.eigh(S)
    tr = float(np.trace(S))
    f = FLOOR_REL * tr / n if tr > 0 else FLOOR_REL
    lam = np.maximum(lam, f)
    F = (Q * np.sqrt(lam)).T  # rows: sqrt(lam_k) q_k'
    Sf = Q @ np.diag(lam) @ Q.T
    return 0.5 * (Sf + Sf.T), F


def sample_cov(R: np.ndarray) -> np.ndarray:
    X = R - R.mean(axis=0)
    return X.T @ X / (R.shape[0] - 1)


def ewma_cov(R: np.ndarray, hl: float) -> np.ndarray:
    h = R.shape[0]
    age = np.arange(h - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / hl)
    w = w / w.sum()
    mu = w @ R
    X = R - mu
    return (X * w[:, None]).T @ X


def ledoit_wolf(R: np.ndarray) -> tuple[np.ndarray, float]:
    h, n = R.shape
    X = R - R.mean(axis=0)
    S = X.T @ X / h
    m = float(np.trace(S)) / n
    D = S - m * np.eye(n)
    d2 = float(np.sum(D * D)) / n
    # sum_k ||x_k x_k' - S||_F^2 = sum_k (|x_k|^4 - 2 x_k' S x_k + ||S||_F^2)
    sq = np.einsum("ij,ij->i", X, X)
    xsx = np.einsum("ij,jk,ik->i", X, S, X)
    b_bar2 = float(np.sum(sq * sq - 2.0 * xsx + float(np.sum(S * S)))) / n / (h * h)
    b2 = min(b_bar2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return shrink_to_identity(S, delta), delta


def shrink_to_identity(S: np.ndarray, delta: float) -> np.ndarray:
    n = S.shape[0]
    m = float(np.trace(S)) / n
    return delta * m * np.eye(n) + (1.0 - delta) * S


def pca_cov(R: np.ndarray, n_pc: int) -> np.ndarray:
    S = sample_cov(R)
    lam, Q = np.linalg.eigh(S)
    k = min(n_pc, S.shape[0])
    Qk, lk = Q[:, -k:], lam[-k:]
    L = (Qk * lk) @ Qk.T
    resid = np.maximum(np.diag(S - L), FLOOR_REL * float(np.trace(S)) / S.shape[0])
    return L + np.diag(resid)


class RiskModel:
    def __init__(self, name: str = "lw", window: int = 250, hl: float = 60.0, n_pc: int = 3):
        if name not in ("sample", "ewma", "lw", "pca"):
            raise ValueError(f"unknown risk model {name!r}")
        self.name = name
        self.window = int(window)
        self.hl = float(hl)
        self.n_pc = int(n_pc)
        self.risk_id = name

    def estimate_from_returns(self, R: np.ndarray) -> RiskEstimate:
        info: dict = {}
        if self.name == "sample":
            S = sample_cov(R)
        elif self.name == "ewma":
            S = ewma_cov(R, self.hl)
        elif self.name == "lw":
            S, delta = ledoit_wolf(R)
            info["intensity"] = delta
        else:
            S = pca_cov(R, self.n_pc)
        Sf, F = floor_psd(S)
        return RiskEstimate(Sf, F, R.shape[0], info)

    def estimate(self, state: DecisionState, cols: np.ndarray) -> RiskEstimate | None:
        """Covariance of the instruments ``cols`` (indices into ``state.ids``); None when fewer
        than 2 common bars are known."""
        h = common_window(np.asarray(state.returns), cols, self.window)
        if h < 2:
            return None
        R = np.asarray(state.returns)[-h:, cols]
        return self.estimate_from_returns(R)
