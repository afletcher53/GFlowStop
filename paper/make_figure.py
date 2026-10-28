"""Figure 1 of the paper: CLEF reliability against review at target recall 0.9, from the committed results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "flow" / "r5"
FLOW, RLSTOP, RULE, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
LABELS = {
    "IPHyperbolic (IP-H), tail='model'": ("IP-H (model)", (16, -20)),
    "IPHyperbolic (IP-H)": ("IP-H", (-16, -3)),
    "IPExponential (IP-E)": ("IP-E", (-21, -9)),
    "IPHyperbolic (CX-H), process='cox'": ("CX-H", (5, -6)),
    "PoissonPoint (IP-P)": ("IP-P", (-8, -11)),
    "ConsecutiveIrrelevant": ("ConsecIrr", (4, 0)),
    "Oracle (offline bound)": ("Oracle", (4, -10)),
}


def main() -> None:
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    summary = pd.read_csv(ROOT / "clef_compare" / "summary.csv")
    rows = summary.loc[summary.target == 0.9].set_index("rule")
    grid = pd.read_csv(ROOT / "C3" / "grid.csv")
    curve = (grid.loc[(grid.target == 0.9) & (grid.aggregation == "mean")]
             .groupby("penalty").agg(reliability=("reliable", "mean"), review=("review_fraction", "mean")).sort_index())
    figure, axis = plt.subplots(figsize=(3.33, 2.05))
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.grid(color="#e4e3df", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.plot(100 * curve.review, 100 * curve.reliability, "-", color=FLOW, linewidth=1.5, label="GFlowStop, penalty grid (descriptive)")
    axis.plot(100 * rows.loc["C3 IPH", "review"], 100 * rows.loc["C3 IPH", "reliability"], "o", color=FLOW, markersize=6,
              markeredgecolor="white", markeredgewidth=1, label="GFlowStop, selected penalty", zorder=4)
    axis.plot(100 * rows.loc["RLStop", "review"], 100 * rows.loc["RLStop", "reliability"], "s", color=RLSTOP, markersize=5,
              markeredgecolor="white", markeredgewidth=1, label="RLStop", zorder=3)
    for rule, (label, offset) in LABELS.items():
        x, y = 100 * rows.loc[rule, "review"], 100 * rows.loc[rule, "reliability"]
        axis.plot(x, y, "o", color=RULE, markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.4, "shrinkA": 0, "shrinkB": 2} if label == "IP-H (model)" else None
        axis.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=6, color=MUTED, arrowprops=arrow)
    axis.set_xlim(0, 32)
    axis.set_ylim(60, 102)
    axis.set_xlabel("Mean review (% of pool)", color=MUTED)
    axis.set_ylabel("Topics reaching recall 0.9 (%)", color=MUTED)
    axis.tick_params(colors=MUTED, length=2)
    axis.legend(loc="lower right", frameon=False, fontsize=5.8)
    figure.tight_layout(pad=0.3)
    figure.savefig(Path(__file__).with_name("frontier.pdf"))


if __name__ == "__main__":
    main()
