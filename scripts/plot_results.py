"""Result figures from the committed result files (they only read results).

python scripts/plot_results.py                 # experiments/results -> docs/figures
python scripts/plot_results.py --quick         # quick results -> experiments/outputs/quick/figures

Every figure is labelled "simulated". The quick figures are for development only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f5cd6", "#d64f8f", "#52514e", "#3ab0c9"]


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, dpi=100, facecolor=SURF)
    plt.close(fig)
    print(p)


def ladder(res, out, tag, note):
    rows = read(os.path.join(res, f"s2b_ladder{tag}.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURF)
    for k, (suffix, lab) in enumerate(
        (("", "cost-aware"), ("_NC", "cost-ignorant (cost_scale 0)"))
    ):
        rs = [r for r in rows if r["variant"].endswith("_NC") == (suffix == "_NC")]
        x = [f(r["forecast_quality"]) for r in rs]
        y = [f(r["ce_ann_mean"]) for r in rs]
        e = [f(r["ce_ann_ci95"]) for r in rs]
        ax.errorbar(x, y, yerr=e, fmt="o-", color=C[k], capsize=3, lw=1.2, ms=4, label=lab)
        for r, xi, yi in zip(rs, x, y):
            ax.annotate(
                r["variant"].replace("MV_", "").replace("_NC", ""),
                (xi, yi),
                fontsize=6,
                color=INK2,
                xytext=(3, 3),
                textcoords="offset points",
            )
    style(
        ax,
        f"S2b information ladder (MV; every rung but plain is an ORACLE bound){note}",
        "measured correlation of the forecast with m (pooled, uncentred)",
        "ce_ann (mean, 95% CI)",
    )
    ax.legend(fontsize=7)
    save(fig, out, "s2b_ladder.png")


def cost_frontier(res, out, tag, note):
    rows = read(os.path.join(res, f"s3{tag}.csv"))
    cs = [r for r in rows if r["variant"].startswith("cost_scale=") or r["variant"] == "parity"]
    cs.sort(key=lambda r: 1.0 if r["variant"] == "parity" else f(r["variant"].split("=")[1]))
    xs = [1.0 if r["variant"] == "parity" else f(r["variant"].split("=")[1]) for r in cs]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURF)
    ax[0].plot([f(r["turnover"]) for r in cs], [f(r["cost_drag"]) for r in cs], "o-", color=C[0])
    for r, x in zip(cs, xs):
        ax[0].annotate(
            f"{x:g}",
            (f(r["turnover"]), f(r["cost_drag"])),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    style(
        ax[0],
        f"S3 cost frontier (labels: cost_scale){note}",
        "mean turnover per bar",
        "cost drag per year",
    )
    ax[1].errorbar(
        xs,
        [f(r["ce_ann_mean"]) for r in cs],
        yerr=[f(r["ce_ann_ci95"]) for r in cs],
        fmt="o-",
        color=C[1],
        capsize=3,
    )
    ax[1].set_xscale("symlog", linthresh=0.25)
    style(
        ax[1],
        "ce_ann by the optimizer's cost_scale (simulator: true costs)",
        "cost_scale (1 = parity)",
        "ce_ann (mean, 95% CI)",
    )
    fig.tight_layout()
    save(fig, out, "s3_cost_frontier.png")


def shift(res, out, tag, note):
    rows = read(os.path.join(res, f"s5_shift{tag}.csv"))
    main = [r for r in rows if not r["strategy"].endswith("_RETUNED")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURF)
    x = range(len(main))
    ax.bar(
        x,
        [f(r["effect_mean"]) for r in main],
        yerr=[f(r["effect_ci95"]) for r in main],
        color=C[0],
        capsize=3,
        alpha=0.85,
    )
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(list(x), [r["strategy"] for r in main], fontsize=7)
    style(
        ax,
        f"S5 effect of the shift: ce_ann(shift) - ce_ann(base), bars from the shift on{note}",
        "",
        "paired difference (mean, 95% CI)",
    )
    save(fig, out, "s5_shift.png")


def horizon(res, out, tag, note):
    rows = read(os.path.join(res, f"s6a_gap{tag}.csv"))
    fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor=SURF)
    combos = sorted({(f(r["phi_gp"]), f(r["Lam"])) for r in rows})
    for k, (phi, lam) in enumerate(combos):
        rs = sorted(
            [r for r in rows if f(r["phi_gp"]) == phi and f(r["Lam"]) == lam],
            key=lambda r: f(r["H"]),
        )
        ax.errorbar(
            [f(r["H"]) for r in rs],
            [f(r["gap_pct_mean"]) for r in rs],
            yerr=[f(r["gap_pct_ci95"]) for r in rs],
            fmt="o-",
            ms=3,
            lw=1,
            capsize=2,
            color=C[k % len(C)],
            label=f"phi {phi:g}, Lam {lam:g}",
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.001)
    ax.set_ylim(-0.001, 60)
    style(
        ax,
        f"S6(a) controller gap to the Garleanu-Pedersen policy{note}",
        "horizon H",
        "gap, % of closed-form utility",
    )
    ax.legend(fontsize=6, ncol=2)
    save(fig, out, "s6_horizon_gap.png")


def execution(res, out, tag, note):
    rows = read(os.path.join(res, f"s7_frontier{tag}.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURF)
    acr = [r for r in rows if r["policy"].startswith("ac_")]
    ax[0].plot(
        [math.sqrt(f(r["V_cf"])) for r in acr],
        [f(r["E_cf"]) for r in acr],
        "-",
        color=C[0],
        label="ac(lambda), closed form",
    )
    ax[0].plot(
        [math.sqrt(f(r["V_mc"])) for r in acr],
        [f(r["E_mc"]) for r in acr],
        "o",
        color=C[0],
        ms=3,
        label="ac(lambda), Monte Carlo",
    )
    for k, r in enumerate([r for r in rows if not r["policy"].startswith("ac_")]):
        ax[0].plot(
            math.sqrt(f(r["V_cf"])), f(r["E_cf"]), "s", color=C[(k + 1) % len(C)], label=r["policy"]
        )
    ax[0].set_yscale("log")
    style(
        ax[0],
        f"S7(a) cost-risk frontier, exact world{note}",
        "sd of shortfall (currency)",
        "expected shortfall (currency)",
    )
    ax[0].legend(fontsize=6)
    dec = read(os.path.join(res, f"s7_decomposition{tag}.csv"))
    comps = [
        "timing_bps",
        "permanent_bps",
        "spread_bps",
        "temporary_bps",
        "opportunity_bps",
        "commission_bps",
    ]
    bottom_pos = [0.0] * len(dec)
    bottom_neg = [0.0] * len(dec)
    for k, cpt in enumerate(comps):
        vals = [f(r[cpt]) for r in dec]
        b = [bp if v >= 0 else bn for v, bp, bn in zip(vals, bottom_pos, bottom_neg)]
        ax[1].bar(range(len(dec)), vals, bottom=b, color=C[k], label=cpt.replace("_bps", ""))
        bottom_pos = [bp + max(v, 0) for v, bp in zip(vals, bottom_pos)]
        bottom_neg = [bn + min(v, 0) for v, bn in zip(vals, bottom_neg)]
    ax[1].set_xticks(range(len(dec)), [r["policy"] for r in dec], fontsize=7)
    style(ax[1], "S7(d) shortfall decomposition (sqrt impact, 2% ADV)", "", "mean, bps of Q*S0")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, out, "s7_execution.png")


def exact_table(res, out, tag, note):
    rows = read(os.path.join(res, f"exact_references{tag}.csv"))
    fig, ax = plt.subplots(figsize=(9, 0.45 + 0.28 * len(rows)), facecolor=SURF)
    ax.axis("off")
    cells = [
        [r["reference"], r["n_problems"], f"{f(r['max_error']):.2e}", r["statuses"], r["note"]]
        for r in rows
    ]
    t = ax.table(
        cellText=cells,
        colLabels=["reference", "n", "max error", "statuses", "measure"],
        loc="center",
        cellLoc="left",
        colWidths=[0.17, 0.05, 0.1, 0.2, 0.48],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    ax.set_title(f"S1 exact references{note}", fontsize=9)
    save(fig, out, "exact_references.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        res, out, tag, note = (
            os.path.join("experiments", "outputs", "quick", "results"),
            os.path.join("experiments", "outputs", "quick", "figures"),
            "_quick",
            " [QUICK]",
        )
    else:
        res, out, tag, note = (
            os.path.join("experiments", "results"),
            os.path.join("docs", "figures"),
            "",
            "",
        )
    for fn in (ladder, cost_frontier, shift, horizon, execution, exact_table):
        fn(res, out, tag, note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
