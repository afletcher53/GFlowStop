"""Imposed-world horizon screen for fixed, fully labelled TAR rankings.

The decision world is explicit: ``W = 1[sum(y) >= K]``.  ``K`` is selected
from the fitted rank-belief distribution before test labels are read, making
this a controlled decision task rather than a natural downstream outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT
from .external_utility import load_fixed_rankings
from .toys import Episode, RankLatentBelief, utility


def _moments(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(probabilities), probabilities.shape[1] + 1), dtype=float)
    variances = means.copy()
    means[:, :-1] = np.cumsum(probabilities[:, ::-1], axis=1)[:, ::-1]
    variances[:, :-1] = np.cumsum((probabilities * (1.0 - probabilities))[:, ::-1], axis=1)[:, ::-1]
    return means, variances


@dataclass
class _TopicPlanner:
    """Capped Bellman planner using a normal-mixture remaining-count tail."""

    episode: Episode
    model: RankLatentBelief

    def __post_init__(self) -> None:
        observation = self.episode.observation(0, 0, 1)
        self.probabilities = self.model._probability_matrix(observation)
        self.means, self.variances = _moments(self.probabilities)
        self.initial_weights = self.model.initial_state(observation).weights

    @property
    def length(self) -> int:
        return len(self.episode.labels)

    def world_probability(self, weights: np.ndarray, depth: int, hits: int, threshold: int) -> float:
        needed = threshold - hits
        remaining = self.length - depth
        if needed <= 0:
            return 1.0
        if needed > remaining:
            return 0.0
        mean, variance = self.means[:, depth], self.variances[:, depth]
        standard_deviation = np.sqrt(variance)
        cdf = np.where(
            standard_deviation > 1e-12,
            ndtr((needed - 0.5 - mean) / np.maximum(standard_deviation, 1e-12)),
            (mean < needed).astype(float),
        )
        return float(np.clip(weights @ (1.0 - cdf), 0.0, 1.0))

    def threshold_for(self, target_world_probability: float) -> tuple[int, float]:
        """Choose the closest root probability without consulting labels."""
        if not 0.0 < target_world_probability < 1.0:
            raise ValueError("target world probability must be strictly between zero and one")
        low, high = 1, self.length
        while low < high:
            middle = (low + high) // 2
            if self.world_probability(self.initial_weights, 0, 0, middle) > target_world_probability:
                low = middle + 1
            else:
                high = middle
        candidates = [low]
        if low > 1:
            candidates.append(low - 1)
        threshold = min(candidates, key=lambda value: (abs(self.world_probability(self.initial_weights, 0, 0, value) - target_world_probability), value))
        return threshold, self.world_probability(self.initial_weights, 0, 0, threshold)

    def rollout(self, threshold: int, q_star: float, cost: float, horizon: int, root_only: bool = False) -> tuple[int, int, float, int, float, float]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not 1 <= threshold <= self.length or not 0.0 < q_star < 1.0 or cost < 0.0:
            raise ValueError("invalid imposed-world utility parameters")

        def terminal_values(weights: np.ndarray, depth: int, hits: int) -> tuple[np.ndarray, float]:
            values = np.full(3, -np.inf)
            if hits >= threshold:
                values[ACTION_ACCEPT] = (1.0 - q_star) / 2.0
                return values, 1.0
            if hits + self.length - depth < threshold:
                values[ACTION_REJECT] = q_star / 2.0
                return values, 0.0
            belief = self.world_probability(weights, depth, hits, threshold)
            values[ACTION_ACCEPT] = (belief - q_star) / 2.0
            values[ACTION_REJECT] = (q_star - belief) / 2.0
            return values, belief

        def action_values(weights: np.ndarray, depth: int, hits: int, remaining_horizon: int) -> tuple[np.ndarray, float]:
            values, belief = terminal_values(weights, depth, hits)
            undecided = hits < threshold and hits + self.length - depth >= threshold
            if remaining_horizon and undecided and depth < self.length:
                probability = float(weights @ self.probabilities[:, depth])
                continuation = -cost
                if probability:
                    hit_weights = weights * self.probabilities[:, depth]
                    continuation += probability * action_values(hit_weights / hit_weights.sum(), depth + 1, hits + 1, remaining_horizon - 1)[0].max()
                if probability < 1.0:
                    miss_weights = weights * (1.0 - self.probabilities[:, depth])
                    continuation += (1.0 - probability) * action_values(miss_weights / miss_weights.sum(), depth + 1, hits, remaining_horizon - 1)[0].max()
                values[ACTION_CONTINUE] = continuation
            return values, belief

        weights = self.initial_weights.copy()
        hits = 0
        root_action = ACTION_REJECT
        root_information_value = 0.0
        root_belief = 0.0
        for depth in range(self.length + 1):
            values, belief = action_values(weights, depth, hits, horizon)
            action = int(np.argmax(values))
            if depth == 0:
                root_action = action
                root_belief = belief
                root_information_value = float(values.max() - max(values[ACTION_ACCEPT], values[ACTION_REJECT]))
                if root_only:
                    return depth, action, belief, root_action, root_belief, root_information_value
            if action != ACTION_CONTINUE:
                return depth, action, belief, root_action, root_belief, root_information_value
            label = int(self.episode.labels[depth])
            likelihood = self.probabilities[:, depth] if label else 1.0 - self.probabilities[:, depth]
            weights *= likelihood
            weights /= weights.sum()
            hits += label
        raise AssertionError("horizon planner did not terminate")


def _next_hit_log_loss(episodes: list[Episode], model: RankLatentBelief) -> float:
    losses: list[float] = []
    for episode in episodes:
        observation = episode.observation(0, 0, 1)
        probabilities = model._probability_matrix(observation)
        weights = model.initial_state(observation).weights
        for depth, label in enumerate(episode.labels):
            probability = np.clip(weights @ probabilities[:, depth], 1e-9, 1.0 - 1e-9)
            losses.append(float(-np.log(probability if label else 1.0 - probability)))
            weights *= probabilities[:, depth] if label else 1.0 - probabilities[:, depth]
            weights /= weights.sum()
    return float(np.mean(losses))


def evaluate_horizon_sweep(
    train: list[Episode],
    test: list[Episode],
    train_sizes: tuple[int, ...] = (10, 25, 50, 94),
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    q_stars: tuple[float, ...] = (0.3, 0.5, 0.7),
    kappas: tuple[float, ...] = (0.05, 0.20, 0.50),
    seed: int = 20260725,
    root_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit only on the supplied train topics and report every held-out cell."""
    if not train or not test or not train_sizes or not horizons:
        raise ValueError("train, test, train sizes, and horizons must be non-empty")
    if max(train_sizes) > len(train) or min(train_sizes) < 2:
        raise ValueError("training sizes must be between two and the train-topic count")
    if min(horizons) < 1 or any(not 0.0 < value < 1.0 for value in q_stars) or min(kappas) < 0.0:
        raise ValueError("invalid horizon, q-star, or kappa grid")
    ordering = np.random.default_rng(seed).permutation(len(train))
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for training_topics in sorted(set(train_sizes)):
        selected = [train[index] for index in ordering[:training_topics]]
        # No raw topic-score mean: the fitted local score features are z-scores and slopes within each ranking.
        model = RankLatentBelief(use_scores=True, use_score_mean=False, use_local_scores=True).fit(selected)
        calibration_rows.append({"training_topics": training_topics, "next_hit_log_loss": _next_hit_log_loss(test, model), "test_topics": len(test)})
        if root_only:
            for episode in test:
                topic, planner = str(episode.latent.get("topic", episode.seed)), _TopicPlanner(episode, model)
                for q_star in q_stars:
                    threshold, selected_root_probability = planner.threshold_for(q_star)
                    for kappa in kappas:
                        cost = kappa / planner.length
                        _depth, _action, _belief, root_action, root_belief, root_information_value = planner.rollout(
                            threshold, q_star, cost, max(horizons), root_only=True,
                        )
                        rows.append({
                            "topic": topic, "training_topics": training_topics, "horizon": max(horizons),
                            "q_star": q_star, "kappa": kappa, "threshold": threshold,
                            "selected_root_world_probability": selected_root_probability,
                            "root_world_probability": root_belief, "root_action": root_action,
                            "root_continues": int(root_action == ACTION_CONTINUE),
                            "root_information_value": root_information_value,
                        })
            continue
        for episode in test:
            topic, planner = str(episode.latent.get("topic", episode.seed)), _TopicPlanner(episode, model)
            for q_star in q_stars:
                threshold, selected_root_probability = planner.threshold_for(q_star)
                for kappa in kappas:
                    cost = kappa / planner.length
                    for horizon in sorted(set(horizons)):
                        depth, action, belief, root_action, root_belief, root_information_value = planner.rollout(threshold, q_star, cost, horizon)
                        world = int(planner.episode.labels.sum() >= threshold)
                        rows.append({
                            "topic": topic, "training_topics": training_topics, "horizon": horizon,
                            "q_star": q_star, "kappa": kappa, "cost": cost, "threshold": threshold,
                            "selected_root_world_probability": selected_root_probability,
                            "root_world_probability": root_belief, "root_action": root_action,
                            "root_continues": int(root_action == ACTION_CONTINUE),
                            "root_information_value": root_information_value,
                            "stop_depth": depth, "action": action, "belief": belief, "world": world,
                            "utility": utility(planner.episode.labels, threshold, depth, action, q_star, cost),
                        })
    per_topic = pd.DataFrame(rows)
    if root_only:
        root_diagnostics = per_topic.groupby(["training_topics", "horizon", "q_star", "kappa"], as_index=False).agg(
            topics=("topic", "size"), mean_threshold=("threshold", "mean"),
            mean_selected_root_world_probability=("selected_root_world_probability", "mean"),
            root_continue_fraction=("root_continues", "mean"),
            mean_root_information_value=("root_information_value", "mean"),
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(calibration_rows), root_diagnostics
    grouped = per_topic.groupby(["training_topics", "horizon", "q_star", "kappa"], as_index=False).agg(
        topics=("topic", "size"), mean_utility=("utility", "mean"), mean_stop_depth=("stop_depth", "mean"),
        decision_accuracy=("action", lambda values: float((values.to_numpy() == np.where(per_topic.loc[values.index, "world"].to_numpy(), ACTION_ACCEPT, ACTION_REJECT)).mean())),
        world_prevalence=("world", "mean"), mean_threshold=("threshold", "mean"),
        root_continue_fraction=("root_continues", "mean"), mean_root_information_value=("root_information_value", "mean"),
    )
    best = grouped.loc[grouped.groupby(["training_topics", "q_star", "kappa"])["mean_utility"].idxmax(), ["training_topics", "q_star", "kappa", "horizon"]].rename(columns={"horizon": "h_star"})
    summary = grouped.merge(best, on=["training_topics", "q_star", "kappa"], validate="many_to_one")
    root_diagnostics = summary[["training_topics", "horizon", "q_star", "kappa", "topics", "world_prevalence", "mean_threshold", "root_continue_fraction", "mean_root_information_value"]].copy()
    return per_topic, summary, pd.DataFrame(calibration_rows), root_diagnostics


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision(root: Path | None, revision: str | None) -> str | None:
    if (root is None) != (revision is None):
        raise ValueError("source root and source revision must be supplied together")
    if root is None:
        return None
    actual = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"source revision mismatch: expected {revision}, got {actual}")
    return actual


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path, help="Fixed training ranking CSV with Topic, Rank, Score, and label columns.")
    parser.add_argument("test_csv", type=Path, help="Fixed held-out ranking CSV with Topic, Rank, Score, and label columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/rcv1_horizon"))
    parser.add_argument("--train-sizes", type=_integers, default=(10, 25, 50, 94))
    parser.add_argument("--horizons", type=_integers, default=(1, 2, 4, 8))
    parser.add_argument("--q-stars", type=_floats, default=(0.3, 0.5, 0.7))
    parser.add_argument("--kappas", type=_floats, default=(0.05, 0.20, 0.50))
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--root-only", action="store_true", help="Write root preflight diagnostics without running the utility horizon sweep.")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--expected-train-topics", type=int)
    parser.add_argument("--expected-test-topics", type=int)
    parser.add_argument("--require-common-length", action="store_true", help="Fail unless every ranking uses the same document-universe length.")
    args = parser.parse_args(argv)
    source_revision = _source_revision(args.source_root, args.source_revision)
    train, test = load_fixed_rankings(args.train_csv), load_fixed_rankings(args.test_csv)
    if args.expected_train_topics is not None and len(train) != args.expected_train_topics:
        raise ValueError(f"expected {args.expected_train_topics} training topics, found {len(train)}")
    if args.expected_test_topics is not None and len(test) != args.expected_test_topics:
        raise ValueError(f"expected {args.expected_test_topics} test topics, found {len(test)}")
    train_topics = {str(episode.latent.get("topic", episode.seed)) for episode in train}
    test_topics = {str(episode.latent.get("topic", episode.seed)) for episode in test}
    if train_topics & test_topics:
        raise ValueError("training and test topic identifiers must be disjoint")
    if args.require_common_length and len({len(episode.labels) for episode in train + test}) != 1:
        raise ValueError("all rankings must share a common document-universe length")
    per_topic, summary, calibration, root_diagnostics = evaluate_horizon_sweep(
        train, test, args.train_sizes, args.horizons, args.q_stars, args.kappas, args.seed, args.root_only,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.root_only:
        per_topic.to_csv(args.output_dir / "per_topic.csv", index=False)
        summary.to_csv(args.output_dir / "utility_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration.csv", index=False)
    root_diagnostics.to_csv(args.output_dir / "root_diagnostics.csv", index=False)
    gates = {"root_diagnostics": {"status": "preflight_complete" if args.root_only else "inspect_before_interpreting_horizon_curve", "file": "root_diagnostics.csv", "note": "These are fitted-belief diagnostics, not an oracle for real rankings."}}
    (args.output_dir / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    metadata = {
        "gflowstop_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "gflowstop_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
        "train_csv": str(args.train_csv.resolve()), "train_sha256": _sha256(args.train_csv),
        "test_csv": str(args.test_csv.resolve()), "test_sha256": _sha256(args.test_csv),
        "source_root": None if args.source_root is None else str(args.source_root.resolve()), "source_revision": source_revision,
        "train_sizes": args.train_sizes, "horizons": args.horizons, "q_stars": args.q_stars, "kappas": args.kappas, "seed": args.seed, "root_only": args.root_only,
        "train_topics": len(train), "test_topics": len(test), "expected_train_topics": args.expected_train_topics, "expected_test_topics": args.expected_test_topics, "require_common_length": args.require_common_length,
        "decision_world": "W = 1[sum(labels) >= K]; K is selected from the train-fitted root belief without test labels",
        "belief_features": "rank plus within-topic score z-scores and slopes; no raw topic-score mean",
        "tail_approximation": "normal mixture over latent rank-intensity states",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    count = "root diagnostics" if args.root_only else f"{len(per_topic)} held-out horizon rows"
    print(f"Wrote {count} to {args.output_dir}")


if __name__ == "__main__":
    main()
