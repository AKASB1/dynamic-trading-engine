"""Tier-2 figures from the committed results in experiments/results/tier2 (they only read results).

python scripts/plot_tier2.py   # -> docs/figures/tier2_*.png
"""

from __future__ import annotations

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK2, SURF = "#52514e", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f5cd6", "#d64f8f"]
RES = os.path.join("experiments", "results", "tier2")
OUT = os.path.join("docs", "figures")


def read(name):
    with open(os.path.join(RES, name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=100, facecolor=SURF)
    plt.close(fig)
    print(p)


SHORT = {
    "CVaR, LS book, proportional cost (LP)": "CVaR (LP)",
    "MV, LS book, 1.5-power cost": "MV 1.5-power",
    "MV, LS book, quadratic cost form": "MV quadratic",
    "R1 mean-variance, budget only": "R1",
    "R2 minimum variance, budget only": "R2",
    "R3 quadratic costs, unconstrained": "R3",
}


def solvers():
    rows = read("solver_comparison.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURF)
    probs = sorted({r["problem"] for r in rows})
    solv = sorted({r["solver"] for r in rows})
    w = 0.8 / len(solv)
    for j, s in enumerate(solv):
        xs, ts, es = [], [], []
        for i, p in enumerate(probs):
            rs = [r for r in rows if r["problem"] == p and r["solver"] == s and r["n"] == "30"]
            if rs and int(rs[0]["failed"]) < int(
                rs[0]["solves"]
            ):  # a solver that failed every solve has no bar
                xs.append(i + j * w)
                ts.append(f(rs[0]["wall_ms_median"]))
                es.append(max(f(rs[0]["max_err_ref_obj"]), 1e-16))
        ax[0].bar(xs, ts, width=w, color=C[j % len(C)], label=s)
        ax[1].bar(xs, es, width=w, color=C[j % len(C)], label=s)
    for a in ax:
        a.set_xticks(
            [i + 0.4 - w / 2 for i in range(len(probs))],
            [SHORT.get(p, p) for p in probs],
            fontsize=7,
        )
        a.set_yscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=6)
    ax[0].set_title("median solve time, n = 30 (ms; shared machine)", fontsize=9)
    ax[1].set_title("largest objective difference vs Clarabel at 1e-10 (relative)", fontsize=9)
    fig.text(
        0.5,
        0.005,
        "no bar: the solver does not take the problem class or failed every solve (OSQP on the CVaR LPs)",
        ha="center",
        fontsize=7,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "tier2_solvers.png")


def capacity():
    rows = read("capacity.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    for k, s in enumerate(("MV", "RANK_LS")):
        rs = [r for r in rows if r["strategy"] == s]
        x = [f(r["capital"]) for r in rs]
        ax[0].errorbar(
            x,
            [f(r["ce_ann_mean"]) for r in rs],
            yerr=[f(r["ce_ann_ci95"]) for r in rs],
            fmt="o-",
            color=C[k],
            capsize=3,
            label=s,
        )
        ax[1].plot(x, [f(r["cap_binds_share"]) for r in rs], "o-", color=C[k], label=s)
    for a in ax:
        a.set_xscale("log")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        a.set_xlabel("capital (quote currency)", fontsize=8)
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_title("Tier 2 capacity: ce_ann against capital (simulated)", fontsize=9)
    ax[1].set_title("share of fills where the participation cap binds", fontsize=9)
    fig.tight_layout()
    save(fig, "tier2_capacity.png")


def adaptive():
    rows = read("adaptive_execution.csv")
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURF)
    w = 0.8 / len(pols)
    for j, p in enumerate(pols):
        rs = [next(r for r in rows if r["scenario"] == s and r["policy"] == p) for s in scen]
        ax.bar(
            [i + j * w for i in range(len(scen))],
            [f(r["IS_bps_mean"]) for r in rs],
            yerr=[f(r["IS_bps_ci95"]) for r in rs],
            width=w,
            color=C[j % len(C)],
            label=p,
            capsize=2,
        )
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(scen))], scen, fontsize=7)
    ax.set_ylabel("mean shortfall, bps of Q*S0", fontsize=8)
    ax.set_title("Tier 2 adaptive execution (5% ADV, sqrt impact, cap 0.1; simulated)", fontsize=9)
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=0.3)
    save(fig, "tier2_adaptive.png")


def main() -> int:
    for fn in (solvers, capacity, adaptive):
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
