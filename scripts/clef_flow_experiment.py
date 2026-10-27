"""Leave-one-collection-out evaluation of the stopping flow network on CLEF TAR 2017--2019.

For each test collection, a network is trained on the AutoTAR rankings of the
other two collections, the twelve extra 2017 AutoTAR topics, and (optionally)
the participant runs of years other than the test year. Any training ranking
whose topic id is in the test collection is dropped. The miss penalty for
each target is chosen by inner leave-one-collection-out over the two training
collections: the cheapest penalty whose pooled validation reliability reaches
``--reliability-goal``. Test labels are used only to score stops.

Writes, per test topic, the stop at every grid penalty (``grid.csv``, a
descriptive frontier) and at the selected penalty (``selected.csv``), plus the
selected penalties and the configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gflowstop import tar_flow
from gflowstop.flow import FlowStoppingNetwork
from gflowstop.tar_flow import FlowStopPolicy, RecallChains, RecallFlowEnsemble, RecallFlowNetwork, training_chains

COLLECTIONS = ("clef2017", "clef2018", "clef2019")
RUN_YEAR = {"clef2017": "2017", "clef2018": "2018", "clef2019": None}
TARGETS = (0.8, 0.9, 0.95)
PENALTIES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


@dataclass(frozen=True)
class Config:
    model: str = "structured"
    aggregator: str = "geometric"
    solver: str = "policy"
    beta: float = 1e4
    hidden: tuple[int, ...] = (64, 64)
    rounds: int = 120
    steps: int = 25
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    copies: int = 3
    runs: bool = True
    run_weight: float = 0.25
    reliability_goal: float = 0.95
    selection: str = "stable"
    max_penalty: float = 256.0
    power_features: bool = True
    seed: int = 0
    ensemble: int = 3


def load(data: Path) -> tuple[dict, dict]:
    def rankings(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {qid: group.sort_values("rank").label.to_numpy() for qid, group in frame.groupby("qid", sort=False)}

    collections = {name: rankings(pd.read_parquet(data / f"{name}.parquet")) for name in (*COLLECTIONS, "clef2017train")}
    runs: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for path in sorted((data / "runs").glob("*/*.parquet")):
        for qid, labels in rankings(pd.read_parquet(path)).items():
            runs.setdefault(path.parent.name, []).append((path.stem, qid, labels))
    return collections, runs


def training_set(collections: dict, runs: dict, held_out: set[str], config: Config) -> tuple[list[np.ndarray], list[float]]:
    """AutoTAR rankings of the non-held-out collections, the extra 2017 topics, and allowed runs."""
    excluded = {qid for name in held_out for qid in collections[name]}
    rankings, weights = [], []
    for name in (*COLLECTIONS, "clef2017train"):
        if name in held_out:
            continue
        for qid, labels in collections[name].items():
            if qid not in excluded:
                rankings.append(labels)
                weights.append(1.0)
    if config.runs:
        blocked = {RUN_YEAR[name] for name in held_out}
        for year, items in runs.items():
            if year in blocked:
                continue
            for _run, qid, labels in items:
                if qid not in excluded:
                    rankings.append(labels)
                    weights.append(config.run_weight)
    return rankings, weights


def fit(rankings: list[np.ndarray], weights: list[float], config: Config):
    if config.model == "structured":
        members = []
        for seed in range(config.seed, config.seed + config.ensemble):
            data = RecallChains.build(rankings, TARGETS, (0.25, config.max_penalty), copies=config.copies, seed=seed, weights=weights)
            members.append(RecallFlowNetwork(
                beta=config.beta, hidden=config.hidden, rounds=config.rounds, steps=config.steps, batch_size=config.batch_size,
                learning_rate=config.learning_rate, weight_decay=config.weight_decay, seed=seed, aggregator=config.aggregator,
            ).fit(data))
        return RecallFlowEnsemble(members)
    data = training_chains(rankings, TARGETS, copies=config.copies, seed=config.seed, weights=weights)
    network = FlowStoppingNetwork(
        beta=config.beta, aggregator=config.aggregator, hidden=config.hidden, rounds=config.rounds, steps=config.steps,
        batch_size=config.batch_size, learning_rate=config.learning_rate, weight_decay=config.weight_decay, seed=config.seed,
        solver=config.solver,
    )
    return network.fit(data)


def stops(network, collection: str, rankings: dict, penalties=PENALTIES) -> pd.DataFrame:
    rows = []
    for qid, labels in rankings.items():
        relevant = int(np.sum(labels > 0))
        cumulative = np.cumsum(labels > 0)
        for target in TARGETS:
            for penalty in penalties:
                stop = FlowStopPolicy(network, target, penalty).stop(labels)
                recall = cumulative[stop - 1] / relevant if relevant else 1.0
                rows.append({"collection": collection, "qid": qid, "target": target, "penalty": penalty, "stop": stop,
                             "collection_size": len(labels), "relevant": relevant, "recall": recall,
                             "reliable": recall >= target, "review_fraction": stop / len(labels)})
    return pd.DataFrame(rows)


def select_penalties(validation: pd.DataFrame, goal: float, rule: str = "stable") -> dict[float, float]:
    """Per target, the penalty to deploy, from pooled validation reliability.

    ``rule='cheapest'`` (the first evaluation): the cheapest penalty whose
    reliability reaches ``goal``. ``rule='stable'``: the smallest penalty from
    which reliability stays at or above ``goal`` for every larger penalty, so
    one lucky dip on about sixty topics cannot be chosen. Either falls back to
    the largest penalty.
    """
    chosen = {}
    for target, frame in validation.groupby("target"):
        summary = frame.groupby("penalty").agg(reliable=("reliable", "mean"), review=("review_fraction", "mean")).sort_index()
        if rule == "cheapest":
            passing = summary.loc[summary.reliable >= goal]
            chosen[float(target)] = float(passing.review.idxmin() if len(passing) else summary.index.max())
        else:
            failing = summary.index[summary.reliable < goal]
            above = summary.index[summary.index > failing.max()] if len(failing) else summary.index
            chosen[float(target)] = float(above.min() if len(above) else summary.index.max())
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/clef"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", nargs="+", default=list(COLLECTIONS))
    for field, default in asdict(Config()).items():
        kind = type(default)
        if kind is tuple:
            parser.add_argument(f"--{field.replace('_', '-')}", type=int, nargs="+", default=list(default))
        elif kind is bool:
            parser.add_argument(f"--{field.replace('_', '-')}", type=lambda value: value.lower() in ("1", "true", "yes"), default=default)
        else:
            parser.add_argument(f"--{field.replace('_', '-')}", type=kind, default=default)
    args = parser.parse_args()
    config = Config(**{field: tuple(getattr(args, field)) if isinstance(default, tuple) else getattr(args, field)
                       for field, default in asdict(Config()).items()})
    import pickle

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    tar_flow.POWER_FEATURES = config.power_features
    penalties = tuple(penalty for penalty in PENALTIES if penalty <= config.max_penalty)
    collections, runs = load(args.data)
    grid, selected, choices = [], [], {}
    for test in args.test:
        started = time.time()
        validation = []
        for inner in COLLECTIONS:
            if inner == test:
                continue
            rankings, weights = training_set(collections, runs, {test, inner}, config)
            network = fit(rankings, weights, config)
            validation.append(stops(network, inner, collections[inner], penalties))
        validation = pd.concat(validation, ignore_index=True)
        validation.assign(test=test).to_csv(args.output / f"validation-{test}.csv", index=False)
        chosen = select_penalties(validation, config.reliability_goal, config.selection)
        rankings, weights = training_set(collections, runs, {test}, config)
        network = fit(rankings, weights, config)
        with open(args.output / f"network-{test}.pkl", "wb") as handle:
            pickle.dump(network, handle)
        frame = stops(network, test, collections[test], penalties)
        grid.append(frame)
        picks = frame.loc[[chosen[target] == penalty for target, penalty in zip(frame.target, frame.penalty)]]
        selected.append(picks)
        choices[test] = {str(target): penalty for target, penalty in chosen.items()}
        print(f"{test}: penalties {chosen}; training rankings {len(rankings)}; {time.time() - started:.0f}s", flush=True)
        print(picks.groupby("target")[["reliable", "review_fraction"]].mean().round(4).to_string(), flush=True)
    grid, selected = pd.concat(grid, ignore_index=True), pd.concat(selected, ignore_index=True)
    import pickle

    grid.to_csv(args.output / "grid.csv", index=False)
    selected.to_csv(args.output / "selected.csv", index=False)
    (args.output / "penalties.json").write_text(json.dumps(choices, indent=2) + "\n")
    print("pooled selected:")
    print(selected.groupby("target")[["reliable", "review_fraction"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
