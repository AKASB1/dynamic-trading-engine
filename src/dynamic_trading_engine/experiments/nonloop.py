"""Experiments that do not go through the rolling loop: S1 (exact references and solver
accuracy), S6(a) and S6(b) (multi-period control against the exact references), S7 (execution).
Each function returns (file name, rows, first columns) triples."""

from __future__ import annotations

import math
import time

import numpy as np

from dynamic_trading_engine.analytics.stats import mean_ci
from dynamic_trading_engine.contracts.costs import CostConfig, ImpactConfig, convex_coefficients
from dynamic_trading_engine.execution import ac as AC
from dynamic_trading_engine.execution import simulator as X
from dynamic_trading_engine.experiments.runner import run_parallel
from dynamic_trading_engine.forecasts.base import Forecast
from dynamic_trading_engine.optimization import problems as PR
from dynamic_trading_engine.optimization.optimizers import Optimizer, OptimizerConfig
from dynamic_trading_engine.optimization.testing import random_forecast, random_state
from dynamic_trading_engine.policies import dp as DP
from dynamic_trading_engine.policies import gp as GP
from dynamic_trading_engine.policies import single as SG
from dynamic_trading_engine.risk.models import RiskModel
from dynamic_trading_engine.rng import stream

# ---------------------------------------------------------------- S1


def _cvar_def(losses, alpha):
    S = len(losses)
    tail = (1.0 - alpha) * S
    srt = np.sort(losses)[::-1]
    k = int(math.floor(tail))
    return (srt[:k].sum() + (tail - k) * (srt[k] if k < S else 0.0)) / tail


def s1(reg, quick, workers):
    c1 = reg["S1"]
    n_prob = c1["n_problems_quick"] if quick else c1["n_problems"]
    rows = []
    g = stream(1, "experiment.S1")

    def ref_row(name, errs, statuses, note):
        st = {s: statuses.count(s) for s in sorted(set(statuses))}
        rows.append(
            {
                "reference": name,
                "n_problems": len(errs),
                "max_error": max(errs) if errs else math.nan,
                "statuses": " ".join(f"{k}:{v}" for k, v in st.items()),
                "note": note,
            }
        )

    costs = CostConfig()
    for name in ("R1", "R2", "R3"):
        errs, sts = [], []
        for _ in range(n_prob):
            n = int(g.integers(2, 25))
            w0 = g.uniform(-0.1, 0.1, n) if name == "R3" else np.zeros(n)
            st = random_state(g, n, w0=w0)
            cols = np.arange(n)
            risk = RiskModel("sample").estimate(st, cols)
            gamma = float(g.uniform(1, 50))
            fc = Forecast(g.standard_normal(n) * 5e-4, np.zeros(n), 0.0, False, "t")
            Si = np.linalg.inv(risk.sigma)
            one = np.ones(n)
            if name == "R1":
                sol = Optimizer(
                    OptimizerConfig("mean_variance", gamma=gamma, cost_scale=0.0),
                    "BUDGET",
                    costs,
                    1,
                ).solve(st, cols, fc, risk)
                nu = (one @ Si @ fc.mu - gamma) / (one @ Si @ one)
                want = Si @ (fc.mu - nu * one) / gamma
            elif name == "R2":
                sol = Optimizer(OptimizerConfig("min_variance"), "BUDGET", costs, 1).solve(
                    st, cols, None, risk
                )
                want = Si @ one / (one @ Si @ one)
            else:
                qc = CostConfig(
                    half_spread_bps=0.0, commission_bps=0.0, impact=ImpactConfig("linear", 0.5)
                )
                sol = Optimizer(OptimizerConfig("mean_variance", gamma=gamma), "UNC", qc, 1).solve(
                    st, cols, fc, risk
                )
                _a, l, _f = convex_coefficients(
                    np.asarray(st.last_close),
                    np.asarray(st.sigma_bar),
                    np.asarray(st.adv),
                    st.portfolio.equity,
                    qc,
                )
                want = np.linalg.solve(gamma * risk.sigma + np.diag(l), fc.mu + np.diag(l) @ w0)
            errs.append(float(np.max(np.abs(sol.weights - want)) / max(1.0, np.max(np.abs(want)))))
            sts.append(sol.status)
        ref_row(name, errs, sts, "max |w - closed form| / max(1, max|w|)")
    errs = []
    for _ in range(50):
        losses = g.standard_normal(int(g.integers(5, 200)))
        for a in (0.9, 0.95, 0.975):
            ru = min(z + np.maximum(losses - z, 0).sum() / ((1 - a) * len(losses)) for z in losses)
            errs.append(abs(ru - _cvar_def(losses, a)) / max(1.0, abs(ru)))
    ref_row(
        "R4", errs, ["exact"] * len(errs), "Rockafellar-Uryasev value vs the definition (no solve)"
    )
    errs, sts = [], []
    for _ in range(n_prob):
        st = random_state(g, 12)
        cols = np.arange(12)
        risk = RiskModel("lw").estimate(st, cols)
        fc = random_forecast(g, 12)
        a = Optimizer(
            OptimizerConfig("robust_box", gamma=5.0, kappa_rob=1e-300), "LS", costs, 5
        ).solve(st, cols, fc, risk)
        b = Optimizer(OptimizerConfig("mean_variance", gamma=5.0), "LS", costs, 5).solve(
            st, cols, fc, risk
        )
        errs.append(float(np.max(np.abs(a.weights - b.weights))))
        sts += [a.status, b.status]
    ref_row("R5", errs, sts, "robust object with kappa ~ 0 vs mean_variance (weights)")
    p = GP.GPParams()
    s = GP.gp_closed_form(p)
    known = [
        (s.A, 0.14129787310950273),
        (s.B, 1.1625476588156523),
        (s.a, 0.3532446827737568),
        (s.aim, 8.227637353852478),
    ]
    ref_row(
        "R6 closed form",
        [abs(x - y) / abs(y) for x, y in known],
        ["exact"] * 4,
        "A, B, a, aim vs the known answer",
    )
    errs = []
    for _ in range(n_prob):
        xp, mu = float(g.uniform(-0.3, 0.3)), float(g.normal(0, GP.stationary_sd(p)))
        want = (1 - s.a) * xp + s.a * s.aim * mu
        got = SG.mpc_first_action_cvxpy(xp, mu, p.g, p.Lam, p.phi_gp, 0.98, 60)
        errs.append(abs(got - want) / abs(want))
    ref_row(
        "R6 controller H=60",
        errs,
        ["optimal"] * len(errs),
        "first action vs the closed-form policy (relative)",
    )
    sd = GP.stationary_sd(p)
    res = DP.solve_dp(p.g, p.phi_gp, p.sigma_eps, p.delta, "quadratic", p.Lam, 0.8, 241, 5 * sd, 61)
    step = res.x_grid[1] - res.x_grid[0]
    errs = []
    for i in range(60, 181, 12):
        for j in range(20, 41, 4):
            want = (1 - s.a) * res.x_grid[i] + s.a * s.aim * res.mu_grid[j]
            errs.append(abs(res.x_grid[res.policy_idx[i, j]] - want) / step)
    ref_row("R6 grid DP", errs, ["exact"] * len(errs), "policy deviation in grid steps (bound: 1)")
    res7 = DP.solve_dp(
        p.g, p.phi_gp, p.sigma_eps, p.delta, "proportional", 1e-7, 0.8, 241, 5 * sd, 61
    )
    errs = [
        abs(res7.x_grid[res7.policy_idx[i, j]] - res7.mu_grid[j] / p.g) / step
        for i in range(60, 181, 12)
        for j in range(20, 41, 4)
    ]
    ref_row(
        "R7 grid DP (c -> 0)",
        errs,
        ["exact"] * len(errs),
        "policy vs mu/g in grid steps (bound: 1)",
    )
    for nm, prm, ans in (
        (
            "R8 known answer 1",
            AC.ACParams(1e6, 5, 5, 0.95, 2.5e-7, 2.5e-6, 1e-6),
            (0.607076163247063, 1212855.55836193),
        ),
        (
            "R8 known answer 2",
            AC.ACParams(1, 1, 4, 0.3, 0.1, 0.5, 2),
            (0.607060859211631, 0.575561755385669),
        ),
    ):
        sch = AC.ac_closed_form(prm)
        e = [abs(sch.kappa - ans[0]) / ans[0], abs(sch.E + prm.lam * sch.V - ans[1]) / ans[1]]
        for x in (AC.ac_qp(prm), AC.ac_linear_solve(prm)):
            E, V = AC.expected_cost_variance(prm, x)
            e.append(abs(E + prm.lam * V - ans[1]) / ans[1])
        ref_row(
            nm,
            e,
            ["exact", "exact", "optimal", "exact"],
            "kappa, objective; QP and linear solve objective",
        )
    # time per solve by problem type and size
    timing = []
    reps = 3 if quick else c1["timing_repetitions"]
    sizes = tuple(c1["timing_sizes_quick"] if quick else c1["timing_sizes"])
    for n in sizes:
        st = random_state(g, n, w0=np.zeros(n))
        cols = np.arange(n)
        risk = RiskModel("lw").estimate(st, cols)
        fc = random_forecast(g, n)
        for name in ("min_variance", "mean_variance", "cvar", "robust_box", "robust_ell", "mpc"):
            opt = Optimizer(OptimizerConfig(name, gamma=5.0, eta_cvar=0.03, H=5), "LS", costs, 5)
            opt.solve(st, cols, fc, risk)  # compile
            walls, cpus, stat = [], [], []
            for _ in range(reps):
                w0, c0 = time.perf_counter(), time.process_time()
                sol = opt.solve(st, cols, fc, risk)
                walls.append((time.perf_counter() - w0) * 1e3)
                cpus.append((time.process_time() - c0) * 1e3)
                stat.append(sol.status)
            timing.append(
                {
                    "optimizer": name,
                    "n": n,
                    "ladder": PR.ladder_size(n),
                    "status": stat[0],
                    "wall_ms_min": min(walls),
                    "wall_cpu_ms_min": min(cpus),
                    "repetitions": reps,
                }
            )
    return [
        ("exact_references", rows, ["reference", "n_problems", "max_error"]),
        ("s1_timing", timing, ["optimizer", "n"]),
    ]


# ---------------------------------------------------------------- S6


def _s6a_seed(args):
    phi, lam, seed, H_list, paths, steps = args
    p = GP.GPParams(phi_gp=phi, Lam=lam)
    s = GP.gp_closed_form(p)
    mu = SG.simulate_mu(paths, steps, phi, p.sigma_eps, stream(seed, f"experiment.S6a.{phi}.{lam}"))
    x_cf = SG.run_linear_policy(mu, 1 - s.a, s.a * s.aim)
    u_cf = SG.discounted_utility(x_cf, mu, p.g, p.delta, lam)
    out = []
    for H in H_list:
        a, b = SG.mpc_coefficients(p.g, lam, phi, p.delta, H)
        u = SG.discounted_utility(SG.run_linear_policy(mu, a, b), mu, p.g, p.delta, lam)
        out.append(
            (
                H,
                float(np.mean(u_cf)),
                float(np.mean(u)),
                float(100 * (np.mean(u_cf) - np.mean(u)) / abs(np.mean(u_cf))),
            )
        )
    return phi, lam, seed, out


def s6a(reg, quick, workers):
    c = reg["S6"]["a"]
    seeds = reg["seeds"]["dev_quick"] if quick else reg["seeds"]["S6"]
    paths = c["quick_paths"] if quick else c["paths"]
    steps = c["quick_steps"] if quick else c["steps"]
    args = [
        (phi, lam, s, c["H"], paths, steps)
        for phi in c["phi_gp"]
        for lam in c["Lam"]
        for s in seeds
    ]
    res = [_s6a_seed(a) for a in args]
    gaps = {}
    for phi, lam, _seed, out in res:
        for H, ucf, uh, gap in out:
            gaps.setdefault((phi, lam, H), []).append((ucf, uh, gap))
    rows = []
    for (phi, lam, H), v in sorted(gaps.items()):
        m, ci, n = mean_ci([x[2] for x in v])
        rows.append(
            {
                "phi_gp": phi,
                "Lam": lam,
                "H": H,
                "gap_pct_mean": m,
                "gap_pct_ci95": ci,
                "n_seeds": n,
                "utility_cf_mean": float(np.mean([x[0] for x in v])),
                "utility_mpc_mean": float(np.mean([x[1] for x in v])),
                "paths_per_seed": paths,
                "steps": steps,
            }
        )
    return [("s6a_gap", rows, ["phi_gp", "Lam", "H"])]


def _s6b_task(args):
    c_prop, H, paths, steps, seed = args
    p = GP.GPParams()
    sd = GP.stationary_sd(p)
    res = DP.solve_dp(
        p.g, p.phi_gp, p.sigma_eps, p.delta, "proportional", c_prop, 0.8, 241, 5 * sd, 61
    )
    mu = SG.simulate_mu(paths, steps, p.phi_gp, p.sigma_eps, stream(seed, "experiment.S6b"))
    # the DP policy on the same paths: trade to the nearest edge of the no-trade band, with the
    # band edges of the grid policy interpolated in mu (no discretization of the position)
    lo_e, hi_e = [], []
    for j in range(len(res.mu_grid)):
        lo, hi = DP.no_trade_band(res, j)
        lo_e.append(lo)
        hi_e.append(hi)
    lo_e, hi_e = np.array(lo_e), np.array(hi_e)
    x_dp = np.zeros_like(mu)
    prev = np.zeros(paths)
    for t in range(steps):
        lo = np.interp(mu[t], res.mu_grid, lo_e)
        hi = np.interp(mu[t], res.mu_grid, hi_e)
        prev = np.clip(prev, lo, hi)
        x_dp[t] = prev
    u_dp = SG.discounted_utility(x_dp, mu, p.g, p.delta, c_prop, "proportional")
    x_h = np.zeros_like(mu)
    for k in range(paths):
        prev = 0.0
        for t in range(steps):
            prev = SG.mpc_first_action_cvxpy(
                prev,
                float(mu[t, k]),
                p.g,
                c_prop,
                p.phi_gp,
                p.delta,
                H,
                cost_kind="proportional",
                opts=PR.CLARABEL_DEFAULT,
            )
            x_h[t, k] = prev
    u_h = SG.discounted_utility(x_h, mu, p.g, p.delta, c_prop, "proportional")
    v0 = float(np.mean([res.value_at(0.0, float(m0)) for m0 in mu[0]]))
    return (
        c_prop,
        H,
        float(np.mean(u_dp)),
        float(np.mean(u_h)),
        float(np.std(u_h - u_dp, ddof=1) / math.sqrt(paths)),
        v0,
    )


def s6b(reg, quick, workers):
    c = reg["S6"]["b"]
    paths = c["quick_paths"] if quick else c["paths"]
    steps = c["quick_steps"] if quick else c["steps"]
    seed = reg["seeds"]["dev_quick"][0] if quick else reg["seeds"]["S6b"][0]
    tasks = [(cp, H, paths, steps, seed) for cp in c["c_prop"] for H in c["H"]]
    res = run_parallel(_s6b_task, tasks, workers)
    rows = []
    for cp, H, u_dp, u_h, se, v0 in res:
        rows.append(
            {
                "c_prop": cp,
                "H": H,
                "utility_dp_policy": u_dp,
                "utility_mpc": u_h,
                "gap_pct": 100 * (u_dp - u_h) / abs(u_dp),
                "gap_se": se,
                "dp_value_at_start": v0,
                "paths": paths,
                "steps": steps,
                "seed": seed,
            }
        )
    return [("s6b_proportional", rows, ["c_prop", "H"])]


# ---------------------------------------------------------------- S7


def _ac_EV(order, x, eta, gamma):
    p = AC.ACParams(order.Q, order.T, order.K, order.sigma_bar * order.S0, gamma, eta, 0.0)
    return AC.expected_cost_variance(p, x)


def _holdings(order, sched):
    return order.Q - np.concatenate([[0.0], np.cumsum(sched)])


def s7(reg, quick, workers):
    c = reg["S7"]
    seeds = reg["seeds"]["dev_quick"] if quick else reg["seeds"]["S7"]
    P = c["quick_paths"] if quick else c["paths"]
    K, T, S0, sig, adv = c["K"], c["T_bars"], c["S0"], c["sigma_bar"], c["adv"]
    tau = tuple([T / K] * K)
    prof = X.u_profile(K, adv, T)
    a = c["a"]
    Qa = a["Q_frac_adv"] * adv * T
    order = X.ParentOrder(1, Qa, S0, sig, adv, tau, prof)
    cfg_a = X.ExecConfig(
        impact="direct",
        eta_ac=a["eta_ac"],
        gamma_ac=a["gamma_ac"],
        half_spread_bps=0.0,
        commission_bps=0.0,
        participation_cap=None,
        volume_noise_sd=0.0,
    )
    pols = [X.ac_policy(order, lam, a["eta_ac"], a["gamma_ac"]) for lam in a["lambda_grid"]]
    pols += [
        X.market_policy(order),
        X.twap_policy(order),
        X.vwap_policy(order),
        X.pov_policy(a["pov_rate"]),
    ]
    frontier = []
    per_seed = {}
    for seed in seeds:
        xi, zv = X.draws(seed, P, K)
        for pol in pols:
            out = X.simulate(order, pol, cfg_a, xi, zv, record=True)
            per_seed.setdefault(pol.name, []).append(out)
    for pol in pols:
        outs = per_seed[pol.name]
        sched = outs[0]["fills"][0]  # deterministic in the exact world
        x = _holdings(order, sched)
        E, V = _ac_EV(order, x, a["eta_ac"], a["gamma_ac"])
        IS = np.concatenate([o["IS"] for o in outs])
        means = [float(np.mean(o["IS"])) for o in outs]
        m, ci, n = mean_ci(means)
        row = {
            "policy": pol.name,
            "lambda_ac": pol.params.get("lambda_ac", ""),
            "E_cf": E,
            "V_cf": V,
            "sd_cf": math.sqrt(V),
            "E_mc": float(np.mean(IS)),
            "E_mc_se": float(np.std(IS, ddof=1) / math.sqrt(len(IS))),
            "V_mc": float(np.var(IS, ddof=1)),
            "completion": float(np.mean(np.concatenate([o["completion"] for o in outs]))),
            "paths": len(IS),
        }
        frontier.append(row)
    # no baseline beats ac(lambda) on E + lambda V (closed form; Monte Carlo within its error)
    checks = []
    for lam in a["lambda_grid"]:
        acr = next(r for r in frontier if r["policy"] == f"ac_{lam:g}")
        best_obj = acr["E_cf"] + lam * acr["V_cf"]
        for r in frontier:
            if r["policy"].startswith("ac_"):
                continue
            obj = r["E_cf"] + lam * r["V_cf"]
            obj_mc = r["E_mc"] + lam * r["V_mc"]
            ac_mc = acr["E_mc"] + lam * acr["V_mc"]
            se = math.sqrt(r["E_mc_se"] ** 2 + acr["E_mc_se"] ** 2)
            checks.append(
                {
                    "lambda_ac": lam,
                    "baseline": r["policy"],
                    "obj_ac_cf": best_obj,
                    "obj_baseline_cf": obj,
                    "baseline_not_better_cf": obj >= best_obj * (1 - 1e-12),
                    "obj_ac_mc": ac_mc,
                    "obj_baseline_mc": obj_mc,
                    "baseline_not_better_mc": obj_mc
                    >= ac_mc
                    - 3 * se
                    - 3 * lam * (r["V_mc"] + acr["V_mc"]) * math.sqrt(2.0 / r["paths"]),
                }
            )
    # (b) sqrt impact, U-shaped profile, volume noise, cap; (c) latency and cap; (d) decomposition
    b = c["b"]
    res_b, res_c, decomp = [], [], []

    def run_world(Q, cfg, label, keep_decomp=False):
        o = X.ParentOrder(1, Q, S0, sig, adv, tau, prof)
        q_twap = Q / K
        h_twap = S0 * cfg.y * sig * math.sqrt(q_twap / (tau[0] * adv))
        eta_eq = h_twap * tau[0] / q_twap  # linear impact matched at the TWAP slice
        plist = [
            X.market_policy(o),
            X.twap_policy(o),
            X.vwap_policy(o),
            X.pov_policy(b["pov_rate"]),
            X.ac_policy(o, b["ac_lambda"], eta_eq, 0.0),
        ]
        out = []
        for pol in plist:
            means, sds, slips, compl = [], [], [], []
            dsum = {}
            for seed in seeds:
                xi, zv = X.draws(seed, P, K)
                r = X.simulate(o, pol, cfg, xi, zv)
                means.append(float(np.mean(r["IS_bps"])))
                sds.append(float(np.std(r["IS_bps"], ddof=1)))
                compl.append(float(np.mean(r["completion"])))
                slips.append(float(np.nanmean(r["slippage_vwap_bps"])))
                for k in (
                    "timing",
                    "permanent",
                    "spread",
                    "temporary",
                    "opportunity",
                    "commission",
                ):
                    dsum.setdefault(k, []).append(float(np.mean(r[k])) / (Q * S0) * 1e4)
            m, ci, n = mean_ci(means)
            row = {
                "world": label,
                "Q_frac_adv": Q / (adv * T),
                "policy": pol.name,
                "IS_bps_mean": m,
                "IS_bps_ci95": ci,
                "IS_bps_sd_paths": float(np.mean(sds)),
                "completion": float(np.mean(compl)),
                "slippage_vwap_bps": float(np.mean(slips)),
                "seeds": n,
                "paths_per_seed": P,
            }
            out.append(row)
            if keep_decomp:
                decomp.append(
                    {
                        "policy": pol.name,
                        "Q_frac_adv": Q / (adv * T),
                        **{f"{k}_bps": float(np.mean(v)) for k, v in dsum.items()},
                        "IS_bps": m,
                    }
                )
        return out

    cfg_b = X.ExecConfig(
        impact="sqrt",
        volume_noise_sd=b["volume_noise_sd"],
        participation_cap=b["participation_cap"],
    )
    for qf in b["Q_frac_adv"]:
        res_b += run_world(
            qf * adv * T, cfg_b, "sqrt+U+noise+cap", keep_decomp=(qf == c["c"]["Q_frac_adv"])
        )
    cc = c["c"]
    for lat in cc["latency"]:
        for cap in cc["participation_cap"]:
            cfg = X.ExecConfig(
                impact="sqrt",
                volume_noise_sd=b["volume_noise_sd"],
                participation_cap=cap,
                latency=lat,
            )
            for r in run_world(cc["Q_frac_adv"] * adv * T, cfg, f"latency={lat},cap={cap:g}"):
                r["latency"], r["cap"] = lat, cap
                res_c.append(r)
    return [
        ("s7_frontier", frontier, ["policy", "lambda_ac"]),
        ("s7_frontier_checks", checks, ["lambda_ac", "baseline"]),
        ("s7_policies", res_b, ["world", "Q_frac_adv", "policy"]),
        ("s7_latency_cap", res_c, ["latency", "cap", "policy"]),
        ("s7_decomposition", decomp, ["policy", "Q_frac_adv"]),
    ]
