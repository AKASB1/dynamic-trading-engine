"""Report for one run: Markdown and a PNG (equity, drawdown, exposure) from the run's logs."""

from __future__ import annotations

import os

import numpy as np

from dynamic_trading_engine.analytics.metrics import summarize
from dynamic_trading_engine.contracts import schemas as S
from dynamic_trading_engine.contracts.canonical import read_dataset_bytes, read_json
from dynamic_trading_engine.contracts.timeutil import format_ts


def make_report(run_dir: str, out_dir: str, ppy: int = 252) -> tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = read_json(os.path.join(run_dir, "run.json"))
    eq = S.load_bytes(S.EQUITY, read_dataset_bytes(os.path.join(run_dir, "equity_v1.csv"))).dicts()
    dec = S.load_bytes(
        __import__("dynamic_trading_engine.engine.logs", fromlist=["DECISIONS"]).DECISIONS,
        read_dataset_bytes(os.path.join(run_dir, "decisions.csv")),
    ).dicts()
    pos = S.load_bytes(
        S.POSITIONS, read_dataset_bytes(os.path.join(run_dir, "positions_v1.csv"))
    ).dicts()
    ts = [r["ts_event"] for r in eq]
    E = np.array([r["equity"] for r in eq])
    first = ts.index(dec[0]["ts_decision"]) if dec else 0
    flows = {k: np.array([r[k] for r in eq]) for k in S.EQUITY_FLOWS}
    costs = (
        flows["spread_cost"]
        + flows["impact_cost"]
        + flows["commission"]
        + flows["borrow"]
        + flows["financing"]
    )
    gross = {}
    for r in pos:
        gross[r["ts_event"]] = gross.get(r["ts_event"], 0.0) + abs(r["value"])
    gexp = np.array([gross.get(t, 0.0) for t in ts]) / E
    s = summarize(
        E,
        flows["hold_pnl"] + flows["trade_pnl"],
        costs,
        np.zeros(len(E)),
        gexp,
        np.zeros(len(E)),
        ppy,
        first,
    )
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    x = np.arange(len(E))
    ax[0].plot(x, E / E[0], lw=1.2, color="#2a78d6")
    ax[0].set_ylabel("equity / initial")
    peak = np.maximum.accumulate(E)
    ax[1].fill_between(x, -(peak - E) / peak, 0, color="#eb6834", alpha=0.6)
    ax[1].set_ylabel("drawdown")
    ax[2].plot(x, gexp, lw=1.0, color="#1baf7a")
    ax[2].set_ylabel("gross exposure")
    ax[2].set_xlabel("bar")
    for a in ax:
        a.axvline(first, color="#999", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    fig.suptitle(
        f"Simulated run: {meta.get('strategy_id')} on {meta.get('market')} seed {meta.get('seed')}"
    )
    fig.tight_layout()
    png = os.path.join(out_dir, "report.png")
    fig.savefig(png, dpi=90)
    plt.close(fig)
    lines = [
        f"# Run report: {meta.get('strategy_id')}",
        "",
        "Simulated on a synthetic market; not investment advice. "
        f"Market `{meta.get('market')}`, seed {meta.get('seed')}, "
        f"label `{meta.get('label') or '-'}`, "
        f"oracle: {meta.get('oracle')}. Bars {format_ts(ts[0])} to {format_ts(ts[-1])}; "
        f"metrics over the {s['n_bars']} bars after the first decision.",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in (
        "ce_ann",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sharpe_gross",
        "max_drawdown",
        "cvar95",
        "cost_drag",
        "gross_exposure",
    ):
        lines.append(f"| {k} | {s[k]:.4f} |")
    lines += [
        "",
        f"Decisions: {len(dec)}; holds: {sum(1 for d in dec if d['hold'])}.",
        "",
        "![report](report.png)",
        "",
    ]
    md = os.path.join(out_dir, "report.md")
    with open(md, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    return md, png
