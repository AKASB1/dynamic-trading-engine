"""Adaptive execution (Tier 2): re-optimize the remaining Almgren-Chriss schedule at the start of
every interval with what has been observed so far.

At interval k the policy knows the remaining quantity (net of fills and of orders in flight) and
the realized volumes of the earlier intervals. It estimates the volume level
``L = sum_{j<k} V_j / sum_{j<k} v_j`` (1 before any observation), rescales the expected profile of
the remaining intervals by ``L``, matches a linear temporary impact to the square-root model at
the remaining TWAP slice with that volume, and computes the static Almgren-Chriss schedule of the
remaining quantity over the remaining intervals; the first slice is the desired quantity, capped
at the participation cap of the expected (rescaled) volume of the interval. It never sees the
current or a later interval's volume or price innovation.
"""

from __future__ import annotations

import numpy as np

from dynamic_trading_engine.execution.ac import ACParams, ac_closed_form
from dynamic_trading_engine.execution.simulator import ExecConfig, ExecPolicy, ParentOrder


class AdaptiveACPolicy(ExecPolicy):
    def __init__(self, order: ParentOrder, cfg: ExecConfig, lam: float, adapt: bool = True):
        super().__init__(
            f"adaptive_ac_{lam:g}" if adapt else f"replan_ac_{lam:g}",
            None,
            params={"lambda_ac": lam},
        )
        self.order, self.cfg, self.lam, self.adapt = order, cfg, lam, adapt

    def desired(self, k: int, info: dict) -> np.ndarray:
        o, cfg = self.order, self.cfg
        rem = np.maximum(info["remaining"], 0.0)
        K = o.K
        prof = np.asarray(o.profile, float)
        past = info.get("past_volume")
        if self.adapt and past is not None and k > 0:
            level = past[:, :k].sum(axis=1) / prof[:k].sum()
        else:
            level = np.ones(len(rem))
        out = np.zeros(len(rem))
        n_left = K - k
        tau = o.tau[k]
        for i in range(len(rem)):
            if rem[i] <= 0:
                continue
            adv = o.adv * float(level[i])
            q = rem[i] / n_left
            if cfg.impact == "direct":
                eta = cfg.eta_ac  # the exact Almgren-Chriss world: the true linear impact
            else:
                beta = 0.5 if cfg.impact == "sqrt" else 1.0
                h = o.S0 * cfg.y * o.sigma_bar * (q / (tau * adv)) ** beta
                eta = max(h * tau / q, 1e-12)  # linear impact matched at the remaining TWAP slice
            p = ACParams(
                rem[i], tau * n_left, n_left, o.sigma_bar * o.S0, cfg.gamma_ac, eta, self.lam
            )
            sched = ac_closed_form(p).n
            want = float(sched[0])
            if cfg.participation_cap is not None:
                want = min(want, cfg.participation_cap * prof[k] * float(level[i]))
            out[i] = want
        return out
