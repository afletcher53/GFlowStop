"""Replay pyterrier-tar's CLEF baselines on the trajectories built by build_clef_trajectories.py.

The rule configurations are those of pyterrier-tar's examples/results_table.py
(estimators, heuristics, and the two certificates with 20 control draws), run
through its public ``run_experiment`` so every row uses its stopping_metrics.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from math import ceil, log
from pathlib import Path

import numpy as np
import pandas as pd
import pyterrier_tar as tar

TARGETS = (0.8, 0.9, 0.95)
CONFIDENCE = 0.95
DRAWS = 20
YEARS = ("clef2017", "clef2018", "clef2019")


def estimators(target, size):
    return {
        "CMHHeuristic": lambda: tar.CMHHeuristic(target, size),
        "AnytimeCMH": lambda: tar.AnytimeCMH(target, size),
        "PoissonPoint (IP-P)": lambda: tar.PoissonPoint(target, size),
        "IPHyperbolic (IP-H)": lambda: tar.IPHyperbolic(target, size),
        "IPHyperbolic (IP-H), tail='model'": lambda: tar.IPHyperbolic(target, size, tail="model"),
        "IPExponential (IP-E)": lambda: tar.IPExponential(target, size),
        "IPAPPrior (IP-A), tail='model'": lambda: tar.IPAPPrior(target, size, tail="model"),
        "PoissonPoint (CX-P), process='cox'": lambda: tar.PoissonPoint(target, size, process="cox"),
        "IPHyperbolic (CX-H), process='cox'": lambda: tar.IPHyperbolic(target, size, process="cox"),
        "BetaBinomial": lambda: tar.BetaBinomial(target, collection_size=size),
        "SWIFTRecall": lambda: tar.SWIFTRecall(target, size),
        "Oracle (offline bound)": lambda: tar.Oracle(target),
    }


def heuristics(size):
    return {
        "Kneedle": lambda: tar.Kneedle(),
        "Budget": lambda: tar.Budget(size),
        "Rule2399": lambda: tar.Rule2399(),
        "ReviewHalf": lambda: tar.ReviewHalf(size),
        "BatchPrecision": lambda: tar.BatchPrecision(),
        "ConsecutiveIrrelevant": lambda: tar.ConsecutiveIrrelevant(),
    }


@lru_cache(maxsize=None)
def smallest_certifying_sample(target):
    return next(r for r in range(1, 1000) if tar.QBCB.required_control_rank(r, target, CONFIDENCE) <= r)


def control_sample(trajectory, positives, rng):
    order = rng.permutation(len(trajectory))
    found = np.cumsum(trajectory.label.to_numpy()[order] > 0)
    reached = np.flatnonzero(found >= positives)
    taken = order[: reached[0] + 1] if len(reached) else order
    control = trajectory.iloc[taken][["qid", "docno", "label"]].copy()
    control["sample_order"] = range(len(control))
    return control


def certificate_factory(name):
    def factory(context):
        positives = smallest_certifying_sample(context.target) if name == "QBCB" else ceil(-log(1 - CONFIDENCE) / (1 - context.target))
        control = control_sample(context.trajectory.sort_values("docno"), positives, context.rng())
        constructor = tar.QBCB if name == "QBCB" else tar.TargetRecapture
        return constructor(control, context.target, CONFIDENCE)
    return factory


def datasets(data: Path, years=YEARS):
    return [tar.ReplayDataset(year, "autotar", pd.read_parquet(data / f"{year}.parquet"), metadata={"source": "build_clef_trajectories.py"})
            for year in years]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/clef"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--certificates", action="store_true")
    args = parser.parse_args()
    if args.certificates:
        rules = [tar.RuleSpec(name, certificate_factory(name), {"confidence": CONFIDENCE, "sample": "minimum positive count"})
                 for name in ("QBCB", "TargetRecapture")]
        seeds = range(DRAWS)
    else:
        rules = [tar.RuleSpec(name, lambda c, name=name: heuristics(c.collection_size)[name](), {"defaults": True}) for name in heuristics(1)]
        rules += [tar.RuleSpec(name, lambda c, name=name: estimators(c.target, c.collection_size)[name](), {"defaults": True})
                  for name in estimators(0.9, 1)]
        seeds = (0,)
    result = tar.run_experiment(datasets(args.data), rules, targets=TARGETS, seeds=seeds, output=args.output, n_jobs=args.jobs)
    print(result.groupby(["rule", "target"])[["reliable", "review_fraction" if "review_fraction" in result else "cost"]].mean())


if __name__ == "__main__":
    main()
