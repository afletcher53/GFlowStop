"""Small K=1 utility screen for a fixed external ranking split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT
from .toys import Episode, RankLatentBelief, k1_rank_bellman, utility, zero_inflated_k1_rank_bellman


@dataclass(frozen=True)
class ScoreMeanPrior:
    center: float
    scale: float
    coefficients: np.ndarray

    def predict(self, scores: np.ndarray) -> float:
        feature = (float(np.mean(scores)) - self.center) / self.scale
        return float(expit(self.coefficients[0] + self.coefficients[1] * feature))


def load_fixed_rankings(path: Path) -> list[Episode]:
    frame = pd.read_csv(path, usecols=["Topic", "Rank", "Score", "label"]).sort_values(["Topic", "Rank"], kind="stable")
    return [
        Episode(group.label.to_numpy(int), group.Score.to_numpy(float), {"topic": str(topic)}, "external_k1", index)
        for index, (topic, group) in enumerate(frame.groupby("Topic", sort=True))
    ]


def fit_score_mean_prior(episodes: list[Episode], ridge: float = 1e-2) -> ScoreMeanPrior:
    means = np.asarray([episode.scores.mean() for episode in episodes])
    worlds = np.asarray([episode.labels.any() for episode in episodes], dtype=float)
    if not (0.0 < worlds.mean() < 1.0):
        raise ValueError("world-prior fit requires both positive and zero topics")
    center, scale = float(means.mean()), max(float(means.std()), 1e-9)
    features = (means - center) / scale

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = expit(coefficients[0] + coefficients[1] * features)
        loss = -np.mean(worlds * np.log(np.clip(probabilities, 1e-12, 1.0)) + (1.0 - worlds) * np.log(np.clip(1.0 - probabilities, 1e-12, 1.0)))
        loss += ridge * coefficients[1] ** 2 / 2.0
        gradient = np.array((np.mean(probabilities - worlds), np.mean((probabilities - worlds) * features) + ridge * coefficients[1]))
        return float(loss), gradient

    result = minimize(lambda value: objective(value), np.zeros(2), jac=True, method="BFGS")
    if not result.success:
        raise RuntimeError(f"score-mean prior did not converge: {result.message}")
    return ScoreMeanPrior(center, scale, np.asarray(result.x, dtype=float))


def _rollout(episode: Episode, model: RankLatentBelief, zero_mass: float, q_star: float, cost: float, method: str) -> dict[str, object]:
    plan = zero_inflated_k1_rank_bellman(model, episode.observation(0, 0, 1), zero_mass, q_star, cost)
    depth = 0
    while plan.actions[depth] == ACTION_CONTINUE:
        if episode.labels[depth]:
            depth += 1
            action = ACTION_ACCEPT
            break
        depth += 1
    else:
        action = int(plan.actions[depth])
    return {
        "method": method,
        "topic": str(episode.latent.get("topic", episode.seed)),
        "q_star": q_star,
        "cost": cost,
        "world": int(episode.labels.any()),
        "stop_depth": depth,
        "action": action,
        "root_world_probability": float(plan.beliefs[0]),
        "net_utility": utility(episode.labels, 1, depth, action, q_star, cost),
    }


def evaluate_k1_split(train: list[Episode], test: list[Episode], q_stars: tuple[float, ...], costs: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = [episode for episode in train if episode.labels.any()]
    zero_mass = 1.0 - len(positives) / len(train)
    model = RankLatentBelief(use_scores=False, use_score_mean=False, use_local_scores=False).fit(positives)
    prior = fit_score_mean_prior(train)
    rows: list[dict[str, object]] = []
    for q_star in q_stars:
        for cost in costs:
            for episode in test:
                rows.append(_rollout(episode, model, zero_mass, q_star, cost, "zero_inflated_rank"))
                rows.append(_rollout(episode, model, 1.0 - prior.predict(episode.scores), q_star, cost, "zero_inflated_score_mean"))
    per_topic = pd.DataFrame(rows)
    summary = per_topic.groupby(["method", "q_star", "cost"], as_index=False).agg(
        topics=("topic", "count"), mean_net_utility=("net_utility", "mean"),
        se_net_utility=("net_utility", lambda values: float(values.std(ddof=1) / np.sqrt(len(values)))),
        mean_stop_depth=("stop_depth", "mean"),
        decision_accuracy=("action", lambda action: float((action.to_numpy() == np.where(
            per_topic.loc[action.index, "world"].to_numpy(), ACTION_ACCEPT, ACTION_REJECT,
        )).mean())),
        mean_root_world_probability=("root_world_probability", "mean"),
    )
    summary["train_zero_mass"] = zero_mass
    summary["score_mean_prior_coefficients"] = json.dumps(prior.coefficients.tolist())
    return per_topic, summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_external_greedy(path: Path, script: Path, q_stars: tuple[float, ...], costs: tuple[float, ...], probability_column: str) -> pd.DataFrame:
    """Call the supplied Greedy implementation directly on the fixed test rows."""
    spec = importlib.util.spec_from_file_location("external_greedy", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import external Greedy script {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frame = pd.read_csv(path, usecols=["Topic", "Rank", "label", probability_column])
    rows: list[dict[str, object]] = []
    for q_star in q_stars:
        for cost in costs:
            for topic, group in frame.groupby("Topic", sort=True):
                result = module.simulate_topic(group, cost, q_star, probability_column, "greedy")
                rows.append({
                    "topic": str(topic), "q_star": q_star, "cost": cost,
                    "net_utility": result.net_utility, "stop_depth": result.stop_depth,
                })
    return pd.DataFrame(rows)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/clefip_k1"))
    parser.add_argument("--q-stars", type=_floats, default=(0.5,))
    parser.add_argument("--costs", type=_floats, default=(0.01,))
    parser.add_argument("--greedy-script", type=Path)
    parser.add_argument("--greedy-probability-column", default="prob_iso_z")
    args = parser.parse_args(argv)
    revision = subprocess.run(["git", "-C", args.source_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if revision != args.source_revision:
        raise ValueError(f"source revision mismatch: expected {args.source_revision}, got {revision}")
    train, test = load_fixed_rankings(args.train_csv), load_fixed_rankings(args.test_csv)
    per_topic, summary = evaluate_k1_split(train, test, args.q_stars, args.costs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_topic.to_csv(args.output_dir / "per_topic.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if args.greedy_script:
        greedy = evaluate_external_greedy(args.test_csv, args.greedy_script, args.q_stars, args.costs, args.greedy_probability_column)
        greedy.to_csv(args.output_dir / "greedy_reference.csv", index=False)
        comparisons = []
        for method, group in per_topic.groupby("method", sort=False):
            paired = group.merge(greedy, on=["topic", "q_star", "cost"], suffixes=("", "_greedy"), validate="one_to_one")
            delta = paired.net_utility - paired.net_utility_greedy
            comparisons.append({
                "method": method, "q_star": paired.q_star.iloc[0], "cost": paired.cost.iloc[0],
                "topics": len(paired), "mean_utility_delta_vs_greedy": float(delta.mean()),
                "se_utility_delta_vs_greedy": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                "win_rate_vs_greedy": float((delta > 0.0).mean()),
                "mean_depth_delta_vs_greedy": float((paired.stop_depth - paired.stop_depth_greedy).mean()),
            })
        pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_vs_greedy.csv", index=False)
    metadata = {
        "gflowstop_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "gflowstop_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
        "source_root": str(args.source_root.resolve()), "source_revision": revision,
        "source_dirty": bool(subprocess.run(["git", "-C", args.source_root, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
        "train_csv": str(args.train_csv.resolve()), "train_sha256": _sha256(args.train_csv),
        "test_csv": str(args.test_csv.resolve()), "test_sha256": _sha256(args.test_csv),
        "q_stars": args.q_stars, "costs": args.costs,
        "score_mean_prior": "one-feature logistic prior fit on training worlds",
        "zero_inflated_model": "rank-latent relevance fit on positive training topics plus structural zero mass",
        "greedy_script": None if not args.greedy_script else str(args.greedy_script.resolve()),
        "greedy_script_sha256": None if not args.greedy_script else _sha256(args.greedy_script),
        "greedy_probability_column": args.greedy_probability_column if args.greedy_script else None,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
