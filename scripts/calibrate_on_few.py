"""Choose the operating point from a few labelled topics of the target collection.

The deployed network is unchanged; only its penalty (and ensemble aggregation)
is chosen differently. For each draw, ``k`` topics of each collection are taken
as a calibration set whose labels are assumed known (for example, reviews a team
has already finished). On those topics, the stable rule picks, per target, the
smallest penalty from which reliability stays at or above ``--goal``, and the
aggregation with the lower calibration review among those meeting it. The
remaining topics are scored at that choice. The pyterrier-tar baselines are
scored on the same held-out topics of the same draw, so each draw gives a paired
comparison. Results are averaged over draws, with 2.5–97.5% ranges.

Inputs: a ``grid.csv`` from ``clef_flow_experiment.py`` (stops at every penalty)
and the ``results.csv`` of ``clef_baselines.py`` on the same trajectories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clef_flow_experiment import TARGETS, select_penalties  # noqa: E402

RULES = ("IPHyperbolic (IP-H), tail='model'", "IPHyperbolic (IP-H)", "IPExponential (IP-E)", "PoissonPoint (IP-P)",
         "ConsecutiveIrrelevant", "CMHHeuristic", "Budget")


def choose(calibration: pd.DataFrame, target: float, goal: float) -> tuple[str, float]:
    best = None
    for aggregation, frame in calibration.loc[calibration.target == target].groupby("aggregation"):
        penalty = select_penalties(frame, goal, "stable")[target]
        rows = frame.loc[frame.penalty == penalty]
        score = (rows.reliable.mean() >= goal, -rows.review_fraction.mean(), rows.reliable.mean())
        if best is None or score > best[0]:
            best = (score, aggregation, penalty)
    return best[1], best[2]


def simulate(grid: pd.DataFrame, baselines: pd.DataFrame, k: int, draws: int, goal: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if "aggregation" not in grid:
        grid = grid.assign(aggregation="mean")
    grid = grid.assign(key=list(zip(grid.collection, grid.qid)))
    baselines = baselines.assign(key=list(zip(baselines.dataset, baselines.qid)))
    topics = {name: sorted(frame.qid.unique()) for name, frame in grid.groupby("collection")}
    rows = []
    for draw in range(draws):
        calibration, held = set(), set()
        for name, qids in topics.items():
            chosen = set(rng.choice(qids, size=min(k, len(qids) - 1), replace=False))
            calibration |= {(name, qid) for qid in chosen}
            held |= {(name, qid) for qid in qids if qid not in chosen}
        for target in TARGETS:
            picks = []
            for name in topics:
                cal = grid.loc[(grid.collection == name) & grid.key.isin(calibration)]
                aggregation, penalty = choose(cal, target, goal)
                picks.append(grid.loc[(grid.collection == name) & (grid.target == target) & (grid.aggregation == aggregation)
                                      & (grid.penalty == penalty) & grid.key.isin(held)])
            ours = pd.concat(picks)
            row = {"draw": draw, "k": k, "target": target, "topics": len(ours),
                   "reliability": ours.reliable.mean(), "review": ours.review_fraction.mean()}
            for rule in RULES:
                theirs = baselines.loc[(baselines.rule == rule) & (baselines.target == target) & baselines.key.isin(held)]
                row[f"{rule}|reliability"] = theirs.reliable.mean()
                row[f"{rule}|review"] = theirs.review_fraction.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, target), frame in result.groupby(["k", "target"]):
        row = {"k": k, "target": target, "reliability": frame.reliability.mean(), "review": frame.review.mean()}
        for rule in RULES:
            d_review = frame.review - frame[f"{rule}|review"]
            d_reliability = frame.reliability - frame[f"{rule}|reliability"]
            row[f"{rule}|review_difference"] = d_review.mean()
            row[f"{rule}|review_range"] = (d_review.quantile(0.025), d_review.quantile(0.975))
            row[f"{rule}|reliability_difference"] = d_reliability.mean()
            row[f"{rule}|share_cheaper_and_as_reliable"] = float(((d_review < 0) & (d_reliability >= 0)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("baselines", type=Path, help="clef_baselines.py results.csv on the same trajectories")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--goal", type=float, default=0.975)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid = pd.read_csv(args.grid)
    baselines = pd.read_csv(args.baselines)
    baselines = baselines.loc[baselines.seed == 0] if "seed" in baselines else baselines
    result = pd.concat([simulate(grid, baselines, k, args.draws, args.goal, args.seed + k) for k in args.k], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = summarise(result)
    summary.to_json(args.output.with_suffix(".summary.json"), orient="records", indent=1)
    for _, row in summary.iterrows():
        print(f"k={row.k} target={row.target}: ours {100 * row.reliability:.1f}% / {100 * row.review:.1f}%")
        for rule in RULES:
            low, high = row[f"{rule}|review_range"]
            print(f"    vs {rule:36s} review {100 * row[f'{rule}|review_difference']:+5.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"
                  f"  reliability {100 * row[f'{rule}|reliability_difference']:+5.1f}"
                  f"  cheaper-and-as-reliable in {100 * row[f'{rule}|share_cheaper_and_as_reliable']:.0f}% of draws")


if __name__ == "__main__":
    main()
