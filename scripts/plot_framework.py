"""Framework figure: the decision loop and the information boundary.

python scripts/plot_framework.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"


def box(ax, x, y, w, h, title, body, color, fc=SURF):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=color, lw=1.6
        )
    )
    ax.text(x + 0.08, y + h - 0.1, title, fontsize=8.2, weight="bold", color=INK, va="top")
    ax.text(x + 0.08, y + h - 0.38, body, fontsize=6.6, color=INK2, va="top", linespacing=1.3)


def arrow(ax, a, b, text="", color=INK2, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.1,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
        )
    )
    if text:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.07,
            text,
            fontsize=6.2,
            color=color,
            ha="center",
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "figures"))
    a = ap.parse_args(argv)
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.95)
    ax.axis("off")
    # information boundary
    ax.add_patch(
        FancyBboxPatch(
            (2.45, 0.25),
            8.35,
            3.55,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc="#f2f6fc",
            ec=BLUE,
            lw=1.0,
            ls="--",
        )
    )
    ax.text(
        10.75,
        3.86,
        "decision pipeline: sees only rows with ts_avail <= t (never the truth)",
        fontsize=7,
        color=BLUE,
        style="italic",
        ha="right",
    )
    box(
        ax,
        0.2,
        3.95,
        2.0,
        1.4,
        "Synthetic market",
        "contract files (QC v1)\nbars, actions, series,\nreturns, liquidity\nmanifests + hashes",
        INK,
    )
    box(
        ax,
        0.2,
        2.05,
        2.0,
        1.4,
        "Truth (separate)",
        "latent signal s, mean m,\nregime, covariance\nonly: oracles (ORACLE),\nevaluators, audit",
        ORANGE,
        fc="#fdf3ee",
    )
    box(
        ax,
        2.7,
        2.2,
        1.9,
        1.3,
        "DecisionState(t)",
        "universe at t, known bars,\nPIT adjustment, returns,\nsignal_x as of t,\nportfolio after fills",
        BLUE,
    )
    box(ax, 4.85, 2.65, 1.75, 0.95, "Forecast", "none, plain;\noracles = bounds", BLUE)
    box(ax, 4.85, 1.55, 1.75, 0.95, "Risk model", "sample, ewma,\nLedoit-Wolf, PCA", BLUE)
    box(
        ax,
        6.85,
        1.85,
        1.9,
        1.5,
        "Optimizer / MPC",
        "CVXPY DPP, cached,\nwarm_start off; MV,\nmin-var, CVaR, robust,\nMPC; QC convex cost",
        AQUA,
    )
    box(
        ax,
        9.0,
        1.85,
        1.7,
        1.5,
        "Orders",
        "w E / P to lots,\nmarket, day,\nsubmitted at close t",
        AQUA,
    )
    box(
        ax,
        6.85,
        0.35,
        3.85,
        1.15,
        "Rolling loop (next bar)",
        "splits, dividends, next-open fills with QC costs, cap,\ndelisting exit, accruals, mark; identity asserted",
        INK,
    )
    box(
        ax,
        2.7,
        0.35,
        3.9,
        1.15,
        "Logs + independent validator",
        "orders, fills, positions, equity (QC 2.9), decisions.csv;\nrecomputes hold/trade PnL, cash, costs, identity",
        GREY,
    )
    box(
        ax,
        0.2,
        0.2,
        2.0,
        1.55,
        "Replay audit",
        "real vs poisoned world,\nmodule purge per world,\ngarbage truth, label;\ncanaries D1 D1b D2 D3",
        YELLOW,
        fc="#fdf8e8",
    )
    arrow(ax, (2.2, 4.4), (3.6, 3.52), "knowledge view")
    arrow(ax, (4.6, 3.1), (4.85, 3.1))
    arrow(ax, (4.6, 2.4), (4.85, 2.05))
    arrow(ax, (6.6, 3.1), (6.85, 2.95))
    arrow(ax, (6.6, 2.05), (6.85, 2.3))
    arrow(ax, (8.75, 2.6), (9.0, 2.6))
    arrow(ax, (9.85, 1.85), (9.6, 1.5))
    arrow(ax, (6.85, 0.92), (6.6, 0.92))
    arrow(ax, (7.3, 1.5), (3.9, 2.2), rad=-0.25)
    ax.text(8.05, 1.6, "next state", fontsize=6.2, color=INK2, ha="center")
    arrow(ax, (2.2, 3.1), (5.2, 3.6), color=ORANGE, rad=-0.45, ls="--")
    ax.text(4.3, 4.22, "truth: oracles only", fontsize=6.2, color=ORANGE)
    arrow(ax, (1.2, 1.75), (1.2, 2.05), color=YELLOW)
    ax.text(
        5.5,
        5.72,
        "dynamic-trading-engine: simulated decision layer (no live trading, no broker connectivity)",
        fontsize=9,
        weight="bold",
        color=INK,
        ha="center",
    )
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "framework.png")
    fig.savefig(p, dpi=110, facecolor=SURF)
    plt.close(fig)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
