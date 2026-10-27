"""Reliability against review on CLEF 2017-2019: fixed rules as labelled points, the flow network's penalty grid as a line.

The grid line is descriptive (choosing a point on it would use test labels);
the filled marker is the penalty chosen by inner validation.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FLOW, RLSTOP, RULE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"
# label and its offset in points, placed to keep neighbouring labels apart
LABELLED = {
    "IPHyperbolic (IP-H), tail='model'": ("IP-H (model)", (-4, 7)),
    "IPHyperbolic (IP-H)": ("IP-H", (6, -4)),
    "IPExponential (IP-E)": ("IP-E", (-26, -4)),
    "PoissonPoint (IP-P)": ("IP-P", (-4, 7)),
    "ConsecutiveIrrelevant": ("Consec. irr.", (6, -10)),
    "BatchPrecision": ("Batch prec.", (6, -10)),
    "Budget": ("Budget", (-14, -13)),
    "Oracle (offline bound)": ("Oracle", (6, -10)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, required=True, help="clef_compare.py output with all_results.csv")
    parser.add_argument("--grid", type=Path, required=True, help="clef_flow_experiment.py grid.csv")
    parser.add_argument("--name", required=True, help="the flow rule's name in all_results.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = pd.read_csv(args.compare / "all_results.csv")
    summary = results.groupby(["rule", "target"]).agg(reliability=("reliable", "mean"), review=("review_fraction", "mean")).reset_index()
    grid = pd.read_csv(args.grid).groupby(["target", "penalty"]).agg(reliability=("reliable", "mean"), review=("review_fraction", "mean")).reset_index()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, facecolor=SURFACE)
    for axis, target in zip(axes, (0.8, 0.9, 0.95)):
        axis.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(MUTED)
        axis.grid(color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        rows = summary.loc[summary.target == target]
        for rule, (label, offset) in LABELLED.items():
            row = rows.loc[rows.rule == rule]
            if row.empty:
                continue
            x, y = 100 * row.review.iloc[0], 100 * row.reliability.iloc[0]
            axis.plot(x, y, "o", color=RULE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            axis.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, color=MUTED)
        curve = grid.loc[grid.target == target].sort_values("penalty")
        axis.plot(100 * curve.review, 100 * curve.reliability, "-", color=FLOW, linewidth=2, zorder=2, label=f"{args.name}, penalty grid")
        chosen = rows.loc[rows.rule == args.name]
        axis.plot(100 * chosen.review, 100 * chosen.reliability, "o", color=FLOW, markersize=10, markeredgecolor=SURFACE, markeredgewidth=2,
                  zorder=4, label=f"{args.name}, selected penalty")
        rl = rows.loc[rows.rule == "RLStop"]
        if len(rl):
            axis.plot(100 * rl.review, 100 * rl.reliability, "s", color=RLSTOP, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                      label="RLStop (same training topics)")
        axis.set_title(f"Target recall {target}", color=INK, fontsize=11, pad=12)
        axis.set_xlabel("Mean review (% of pool)", color=MUTED)
        axis.set_xlim(0, 45)
        axis.set_ylim(60, 104)
        axis.tick_params(colors=MUTED)
    axes[0].set_ylabel("Topics reaching target (%)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
