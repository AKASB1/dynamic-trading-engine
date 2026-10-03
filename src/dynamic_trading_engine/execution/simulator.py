"""Execution simulator for one parent order over K intervals (vectorized across paths).

Mid price (currency units, arithmetic walk): ``S_k = S_{k-1} + sigma_abs sqrt(tau_k) xi_k +
gamma_ac q_k`` with ``sigma_abs = sigma_bar * S_0`` per bar, ``xi`` from ``exec.price`` and the
signed executed quantity ``q_k`` (permanent impact, an extension of the contract; default 0).
Realized volume ``V_k = v_k exp(s_v z_k - s_v^2 / 2)`` with ``z`` from ``exec.volume``.
Execution price ``p_k = S_{k-1} + side (half_spread + h_k)`` with the temporary impact
``h_k = S_0 y sigma_bar (|q_k| / (tau_k ADV))^beta`` (``sqrt``: beta 0.5, ``linear``: beta 1) or
``eta_ac |q_k| / tau_k`` (``direct``); commission of the contract at the reference ``S_{k-1}``.
Impact uses the decision-time ``sigma_bar`` and ``ADV``, never the realized volume.

Fills: the desired quantity of an interval plus the carried shortfall, up to
``participation_cap * V_k`` and the remaining quantity; the shortfall is carried (option); what
is left after the last interval is not executed (the opportunity term). Latency ``d``: a quantity
decided at the start of interval k reaches the market in interval k + d. A ``pov(rate)`` order
executes ``rate * V_k`` of the concurrent volume from interval d on until complete.

Implementation shortfall (positive = cost) and its exact decomposition:
``IS = side [sum |q_f| (p_f - S_0) + (Q - Q_filled)(S_end - S_0)] + commission``
``= timing + permanent + spread + temporary + opportunity + commission``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dynamic_trading_engine.execution.ac import ACParams, ac_closed_form
from dynamic_trading_engine.rng import stream


@dataclass(frozen=True)
class ExecConfig:
    impact: str = "sqrt"  # sqrt, linear, direct
    y: float = 0.5
    eta_ac: float = 0.0
    gamma_ac: float = 0.0
    half_spread_bps: float = 2.0
    commission_bps: float = 1.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    participation_cap: float | None = 0.1
    volume_noise_sd: float = 0.3
    carry: bool = True
    latency: int = 0
    # Tier-2 volume-forecast study (defaults leave the model above unchanged): a persistent
    # per-path volume level exp(s_l z_l - s_l^2 / 2) (z_l given to ``simulate``), and a realized
    # volume shape that differs from the expected profile the policies plan with
    volume_level_sd: float = 0.0
    true_profile: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ParentOrder:
    side: int  # +1 buy, -1 sell
    Q: float  # shares, > 0
    S0: float  # arrival mid
    sigma_bar: float  # per-bar sd of the simple return, decision time
    adv: float  # shares per bar, decision time
    tau: tuple[float, ...]  # interval lengths in bars
    profile: tuple[float, ...]  # expected volume per interval (shares), sums to adv * T

    @property
    def K(self) -> int:
        return len(self.tau)

    @property
    def T(self) -> float:
        return float(sum(self.tau))


def u_profile(K: int, adv: float, T: float, amp: float = 1.0) -> tuple[float, ...]:
    w = np.array([1.0 + amp * (2.0 * (k + 0.5) / K - 1.0) ** 2 for k in range(K)])
    return tuple(float(v) for v in adv * T * w / w.sum())


def flat_profile(K: int, adv: float, T: float) -> tuple[float, ...]:
    return tuple(float(adv * T / K) for _ in range(K))


# ---------------------------------------------------------------- policies


@dataclass
class ExecPolicy:
    """A static schedule (desired quantity per interval) or a rate order (``rate``)."""

    name: str
    schedule: np.ndarray | None = None
    rate: float | None = None
    params: dict = field(default_factory=dict)

    def desired(self, k: int, info: dict) -> np.ndarray:
        """Desired quantity decided at the start of interval k from what is known then."""
        return np.full(info["n_paths"], float(self.schedule[k]))


def market_policy(order: ParentOrder) -> ExecPolicy:
    s = np.zeros(order.K)
    s[0] = order.Q
    return ExecPolicy("market", s)


def twap_policy(order: ParentOrder) -> ExecPolicy:
    return ExecPolicy("twap", np.full(order.K, order.Q / order.K))


def vwap_policy(order: ParentOrder) -> ExecPolicy:
    v = np.asarray(order.profile, float)
    return ExecPolicy("vwap", order.Q * v / v.sum())


def pov_policy(rate: float) -> ExecPolicy:
    return ExecPolicy(f"pov_{rate:g}", None, rate=rate, params={"rate": rate})


def ac_policy(order: ParentOrder, lam: float, eta: float, gamma: float) -> ExecPolicy:
    """The static Almgren-Chriss schedule with sigma_ac = sigma_bar * S_0 (equal intervals)."""
    p = ACParams(order.Q, order.T, order.K, order.sigma_bar * order.S0, gamma, eta, lam)
    sch = ac_closed_form(p)
    return ExecPolicy(f"ac_{lam:g}", np.asarray(sch.n, float), params={"lambda_ac": lam})


# ---------------------------------------------------------------- simulation


def _impact(q, tau, order: ParentOrder, cfg: ExecConfig):
    if cfg.impact == "direct":
        return cfg.eta_ac * q / tau
    beta = 0.5 if cfg.impact == "sqrt" else 1.0
    return order.S0 * cfg.y * order.sigma_bar * (q / (tau * order.adv)) ** beta


def _commission(q, m, cfg: ExecConfig):
    c = q * m * cfg.commission_bps / 1e4 + q * cfg.commission_per_share
    return np.where(q > 0, np.maximum(cfg.min_commission, c), 0.0)


def draws(seed: int, n_paths: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    xi = stream(seed, "exec.price").standard_normal((n_paths, K))
    zv = stream(seed, "exec.volume").standard_normal((n_paths, K))
    return xi, zv


def simulate(
    order: ParentOrder,
    policy: ExecPolicy,
    cfg: ExecConfig,
    xi: np.ndarray,
    zv: np.ndarray,
    record: bool = False,
    zl: np.ndarray | None = None,
) -> dict:
    """Vectorized over paths (rows of ``xi`` and ``zv``; ``zl`` the per-path level draws of the
    volume-forecast study, used only when ``volume_level_sd`` > 0)."""
    P, K = xi.shape
    side = order.side
    hs = order.S0 * cfg.half_spread_bps / 1e4
    sig_abs = order.sigma_bar * order.S0
    sv = cfg.volume_noise_sd
    tau = np.asarray(order.tau, float)
    prof = np.asarray(cfg.true_profile if cfg.true_profile is not None else order.profile, float)
    sl = cfg.volume_level_sd
    level = np.exp(sl * zl - 0.5 * sl * sl) if (sl > 0 and zl is not None) else np.ones(P)
    realized = np.zeros((P, K))
    S = np.full(P, order.S0)
    filled = np.zeros(P)
    carry = np.zeros(P)
    arrivals = np.zeros((K, P))
    RW = np.zeros(P)
    cum_q = np.zeros(P)
    comp = {k: np.zeros(P) for k in ("timing", "permanent", "spread", "temporary", "commission")}
    exec_cost = np.zeros(P)
    notional = np.zeros(P)
    vw_num = np.zeros(P)
    vw_den = np.zeros(P)
    fills = np.zeros((P, K)) if record else None
    decided = np.zeros((P, K)) if record else None
    cap = cfg.participation_cap
    for k in range(K):
        info = {
            "n_paths": P,
            "k": k,
            "remaining": order.Q - filled - arrivals[k:].sum(axis=0) - carry,
            "past_volume": realized[:, :k],
        }
        if policy.rate is None:
            d = np.maximum(policy.desired(k, info), 0.0)
            if record:
                decided[:, k] = d
            if k + cfg.latency < K:
                arrivals[k + cfg.latency] += d
        V = prof[k] * np.exp(sv * zv[:, k] - 0.5 * sv * sv)
        if sl > 0:
            V = V * level
        realized[:, k] = V
        if policy.rate is not None:
            avail = policy.rate * V if k >= cfg.latency else np.zeros(P)
        else:
            avail = arrivals[k] + carry
        lim = cap * V if cap is not None else np.full(P, np.inf)
        q = np.minimum(np.minimum(avail, lim), order.Q - filled)
        q = np.maximum(q, 0.0)
        if policy.rate is None:
            carry = (avail - q) if cfg.carry else np.zeros(P)
        h = _impact(q, tau[k], order, cfg)
        p = S + side * (hs + h)
        comm = _commission(q, S, cfg)
        comp["timing"] += side * q * RW
        comp["permanent"] += cfg.gamma_ac * q * cum_q
        comp["spread"] += q * hs
        comp["temporary"] += q * h
        comp["commission"] += comm
        exec_cost += q * (p - order.S0)
        notional += q * p
        filled += q
        if record:
            fills[:, k] = q
        innov = sig_abs * math.sqrt(tau[k]) * xi[:, k]
        S_new = S + innov + cfg.gamma_ac * side * q
        vw_num += V * 0.5 * (S + S_new)
        vw_den += V
        S = S_new
        RW = RW + innov
        cum_q = cum_q + q
    unfilled = order.Q - filled
    comp["opportunity"] = side * unfilled * (S - order.S0)
    IS = side * (exec_cost + unfilled * (S - order.S0)) + comp["commission"]
    vwap = vw_num / vw_den
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = notional / filled
        slip = side * (avg - vwap) / vwap * 1e4
    out = {
        "IS": IS,
        "IS_bps": IS / (order.Q * order.S0) * 1e4,
        "completion": filled / order.Q,
        "slippage_vwap_bps": slip,
        "S_end": S,
        **comp,
    }
    if record:
        out["fills"] = fills
        out["decided"] = decided
    return out


def simulate_path_reference(
    order: ParentOrder, policy: ExecPolicy, cfg: ExecConfig, xi_row: np.ndarray, zv_row: np.ndarray
) -> dict:
    """One path with plain Python scalars: the reference of the vectorized simulator."""
    K = order.K
    side = order.side
    hs = order.S0 * cfg.half_spread_bps / 1e4
    sig_abs = order.sigma_bar * order.S0
    sv = cfg.volume_noise_sd
    S, filled, carry, RW, cum = order.S0, 0.0, 0.0, 0.0, 0.0
    arrivals = [0.0] * K
    c = dict.fromkeys(("timing", "permanent", "spread", "temporary", "commission"), 0.0)
    exec_cost = 0.0
    for k in range(K):
        if policy.rate is None:
            d = max(float(policy.schedule[k]), 0.0)
            if k + cfg.latency < K:
                arrivals[k + cfg.latency] += d
        V = order.profile[k] * math.exp(sv * zv_row[k] - 0.5 * sv * sv)
        if policy.rate is not None:
            avail = policy.rate * V if k >= cfg.latency else 0.0
        else:
            avail = arrivals[k] + carry
        lim = cfg.participation_cap * V if cfg.participation_cap is not None else math.inf
        q = max(min(avail, lim, order.Q - filled), 0.0)
        if policy.rate is None:
            carry = (avail - q) if cfg.carry else 0.0
        if cfg.impact == "direct":
            h = cfg.eta_ac * q / order.tau[k]
        else:
            beta = 0.5 if cfg.impact == "sqrt" else 1.0
            h = order.S0 * cfg.y * order.sigma_bar * (q / (order.tau[k] * order.adv)) ** beta
        p = S + side * (hs + h)
        comm = 0.0
        if q > 0:
            comm = max(
                cfg.min_commission, q * S * cfg.commission_bps / 1e4 + q * cfg.commission_per_share
            )
        c["timing"] += side * q * RW
        c["permanent"] += cfg.gamma_ac * q * cum
        c["spread"] += q * hs
        c["temporary"] += q * h
        c["commission"] += comm
        exec_cost += q * (p - order.S0)
        filled += q
        innov = sig_abs * math.sqrt(order.tau[k]) * xi_row[k]
        S = S + innov + cfg.gamma_ac * side * q
        RW += innov
        cum += q
    unfilled = order.Q - filled
    c["opportunity"] = side * unfilled * (S - order.S0)
    IS = side * (exec_cost + unfilled * (S - order.S0)) + c["commission"]
    return {"IS": IS, "completion": filled / order.Q, **c}
