"""Score trained stopping flow networks and RLStop through pyterrier-tar, next to its baseline replay.

Every learned rule is replayed by pyterrier-tar's ``run_experiment`` on the same
trajectories as ``clef_baselines.py``, so all rows share its stopping_metrics.
Each topic is scored with the network trained without its collection
(``clef_flow_experiment.py`` output), at the penalty chosen by inner validation.
RLStop (pyterrier-tar's PPO implementation, default settings) is trained per
target on the same AutoTAR training topics; a topic shorter than its 100
windows is reviewed in full and marked as not fired.

Paired comparisons report, per target, the mean difference in review fraction
and in reliability against each named baseline over the same topics, with
topic-bootstrap 95% intervals.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyterrier_tar as tar
from pyterrier_tar.rules._base import _TrajectoryStoppingRule

from gflowstop.tar_flow import FlowStopPolicy, RecallFlowEnsemble

COLLECTIONS = ("clef2017", "clef2018", "clef2019")
TARGETS = (0.8, 0.9, 0.95)


class FlowStopRule(_TrajectoryStoppingRule):
    """pyterrier-tar replay of a stopping flow network: stop when ``P_F(stop) >= threshold``."""

    _equation = "P_F(stop | x) = F(x, stop) / (F(x, stop) + F(x, continue)) >= threshold"

    def __init__(self, policy: FlowStopPolicy):
        self.policy = policy

    def _iter_checkpoints(self, ranked, labels):
        labels = np.asarray(labels) > 0
        depths, probabilities = self.policy.stop_probabilities(labels)  # each row reads only its reviewed prefix
        for depth, probability in zip(depths, probabilities):
            fired = bool(probability >= self.policy.threshold)
            yield {"index": int(depth), "fired": fired, "stop_probability": float(probability),
                   "reason": "flow policy chose stop" if fired else "flow policy chose continue"}
            if fired:
                return


class RLStopOrFullReview(tar.RLStop):
    """RLStop, reviewing in full any topic too short for its windows."""

    def _iter_checkpoints(self, ranked, labels):
        if len(labels) < self.n_windows:
            yield {"index": len(labels), "fired": False, "reason": "fewer documents than RLStop windows; full review"}
            return
        yield from super()._iter_checkpoints(ranked, labels)


def datasets(data: Path) -> list:
    return [tar.ReplayDataset(name, "autotar", pd.read_parquet(data / f"{name}.parquet")) for name in COLLECTIONS]


def train_rlstop(data: Path, output: Path, timesteps: int) -> dict:
    """One RLStop per held-out collection and target, trained on the other collections' AutoTAR topics."""
    frames = {name: pd.read_parquet(data / f"{name}.parquet") for name in (*COLLECTIONS, "clef2017train")}
    policies = {}
    for test in COLLECTIONS:
        parts, seen = [], set(frames[test].qid)
        for name, frame in frames.items():
            if name != test:
                parts.append(frame.loc[~frame.qid.isin(seen)])  # a topic judged in two collections is used once
                seen |= set(frame.qid)
        training = pd.concat(parts, ignore_index=True)
        sizes = training.groupby("qid").size()
        training = training.loc[training.qid.isin(sizes.index[sizes >= 100])]
        for target in TARGETS:
            path = output / f"rlstop-{test}-{target}"
            rule = RLStopOrFullReview(target, total_timesteps=timesteps)
            if Path(str(path) + ".json").exists() or (output / f"rlstop-{test}-{target}.json").exists():
                try:
                    rule.load(path)
                except ValueError:
                    rule.fit(training)
                    rule.save(path)
            else:
                rule.fit(training)
                rule.save(path)
            policies[test, target] = rule
            print(f"RLStop {test} {target}: trained on {training.qid.nunique()} topics", flush=True)
    return policies


def bootstrap(values: np.ndarray, replicates: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired(results: pd.DataFrame, rule: str, baselines: list[str]) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        ours = results.loc[(results.rule == rule) & (results.target == target)].set_index(["dataset", "qid"])
        for baseline in baselines:
            theirs = results.loc[(results.rule == baseline) & (results.target == target)].set_index(["dataset", "qid"])
            common = ours.index.intersection(theirs.index)
            review = (ours.loc[common, "review_fraction"] - theirs.loc[common, "review_fraction"]).to_numpy(float)
            reliable = (ours.loc[common, "reliable"].astype(float) - theirs.loc[common, "reliable"].astype(float)).to_numpy()
            r_mean, r_low, r_high = bootstrap(review)
            l_mean, l_low, l_high = bootstrap(reliable)
            rows.append({"rule": rule, "baseline": baseline, "target": target, "topics": len(common),
                         "review_difference": r_mean, "review_ci": [r_low, r_high],
                         "reliability_difference": l_mean, "reliability_ci": [l_low, l_high],
                         "only_ours_reliable": int((reliable > 0).sum()), "only_baseline_reliable": int((reliable < 0).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/clef"))
    parser.add_argument("--flow", type=Path, nargs="+", required=True, help="clef_flow_experiment.py output folders")
    parser.add_argument("--names", nargs="+", help="rule names for the flow folders")
    parser.add_argument("--baselines", type=Path, required=True, help="clef_baselines.py output folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rlstop-timesteps", type=int, default=0, help="train RLStop with this many timesteps (0 skips it)")
    parser.add_argument("--grid", action="store_true", help="also replay every grid penalty of the first flow folder")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.names or [folder.name for folder in args.flow]
    replays = datasets(args.data)
    # CD011686 is judged in both 2018 and 2019 with different qrels; its relevant count tells them apart.
    keys = [(dataset.name, qid, int(frame.label.sum())) for dataset in replays for qid, frame in dataset.trajectory.groupby("qid")]
    lookup = {(qid, relevant): name for name, qid, relevant in keys}
    if len(lookup) != len(keys):
        raise ValueError("a topic cannot be assigned to one collection")

    def collection(trajectory: pd.DataFrame) -> str:
        return lookup[str(trajectory.qid.iloc[0]), int(trajectory.label.sum())]

    rules = []
    for name, folder in zip(names, args.flow):
        networks = {test: pickle.loads((folder / f"network-{test}.pkl").read_bytes()) for test in COLLECTIONS}
        penalties = json.loads((folder / "penalties.json").read_text())

        def factory(context, networks=networks, penalties=penalties):
            test = collection(context.trajectory)
            choice = penalties[test][str(context.target)]
            if isinstance(choice, dict):  # bagged runs also choose the ensemble aggregation
                network = RecallFlowEnsemble(networks[test].members, choice["aggregation"])
                return FlowStopRule(FlowStopPolicy(network, context.target, choice["penalty"]))
            return FlowStopRule(FlowStopPolicy(networks[test], context.target, choice))

        rules.append(tar.RuleSpec(name, factory, {"folder": str(folder), "penalties": penalties}, promise="policy"))
        if args.grid and folder == args.flow[0]:
            for penalty in json.loads((folder / "config.json").read_text()).get("penalties", [0.5, 1, 2, 4, 8, 16, 32]):
                def grid_factory(context, networks=networks, penalty=penalty):
                    return FlowStopRule(FlowStopPolicy(networks[collection(context.trajectory)], context.target, penalty))
                rules.append(tar.RuleSpec(f"{name} penalty={penalty:g}", grid_factory, {"folder": str(folder), "penalty": penalty}, promise="policy"))
    if args.rlstop_timesteps:
        policies = train_rlstop(args.data, args.output / "rlstop", args.rlstop_timesteps)
        rules.append(tar.RuleSpec(
            "RLStop", lambda context: policies[collection(context.trajectory), context.target],
            {"timesteps": args.rlstop_timesteps, "training": "other collections' AutoTAR topics with >= 100 documents"}, promise="policy",
        ))
    learned = tar.run_experiment(replays, rules, targets=TARGETS, output=args.output / "replay", n_jobs=args.jobs)
    baselines = pd.read_csv(args.baselines / "results.csv")
    baselines = baselines.loc[baselines.seed == 0] if "seed" in baselines else baselines
    results = pd.concat([baselines, learned], ignore_index=True)
    results.to_csv(args.output / "all_results.csv", index=False)
    summary = results.groupby(["rule", "target"]).agg(reliability=("reliable", "mean"), review=("review_fraction", "mean"),
                                                      recall=("recall", "mean"), topics=("qid", "count")).reset_index()
    summary.to_csv(args.output / "summary.csv", index=False)
    by_collection = results.groupby(["rule", "target", "dataset"]).agg(reliability=("reliable", "mean"), review=("review_fraction", "mean")).reset_index()
    by_collection.to_csv(args.output / "by_collection.csv", index=False)
    references = ["IPHyperbolic (IP-H), tail='model'", "IPHyperbolic (IP-H)", "IPExponential (IP-E)", "PoissonPoint (IP-P)",
                  "ConsecutiveIrrelevant", "CMHHeuristic", "Kneedle", "Budget"]
    if args.rlstop_timesteps:
        references.append("RLStop")
    comparisons = pd.concat([paired(results, name, [r for r in references + [n for n in names if n != name]]) for name in names], ignore_index=True)
    comparisons.to_json(args.output / "paired.json", orient="records", indent=1)
    table = summary.pivot(index="rule", columns="target", values=["reliability", "review"])
    print((100 * table).round(1).sort_values(("review", 0.9)).to_string())


if __name__ == "__main__":
    main()
