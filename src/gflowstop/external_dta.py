"""Audit whether the count-threshold world can represent DTA decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topic_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Topic", "label"])
    counts = frame.groupby("Topic", as_index=False)["label"].sum()
    counts["topic"] = counts["Topic"].astype(str).str.removesuffix(".csv")
    return counts[["topic", "label"]].rename(columns={"label": "total_relevant"})


def _final_pi(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["topic", "depth", "pi"])
    final = frame.sort_values(["topic", "depth"], kind="stable").groupby("topic", as_index=False).tail(1)
    if final.topic.duplicated().any() or not np.isfinite(final.pi).all():
        raise ValueError("decision timeseries must have one finite final pi per topic")
    return final[["topic", "pi"]].rename(columns={"pi": "final_pi"})


def load_dta_split(rankings: Path, timeseries: Path) -> pd.DataFrame:
    """Join a fixed ranking's full relevant count to its final DTA decision."""
    counts = _topic_counts(rankings)
    merged = counts.merge(_final_pi(timeseries), on="topic", validate="one_to_one")
    if len(merged) != len(counts):
        raise ValueError("some ranking topics have no final DTA decision")
    return merged.sort_values("topic", kind="stable").reset_index(drop=True)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_count_threshold(train: pd.DataFrame, world: np.ndarray) -> int:
    """Fit the original increasing ``sum(y) >= K`` world on training topics."""
    counts = train.total_relevant.to_numpy()
    candidates = range(int(counts.max()) + 2)
    return max(candidates, key=lambda threshold: (np.mean((counts >= threshold) == world), -threshold))


def evaluate_count_world_linkage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    decision_thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the best full-information count threshold selected on training only."""
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for pi_threshold in decision_thresholds:
        if not 0.0 < pi_threshold < 1.0:
            raise ValueError("DTA decision thresholds must be strictly between zero and one")
        train_world = (train.final_pi >= pi_threshold).to_numpy()
        count_threshold = _best_count_threshold(train, train_world)
        majority_action = int(train_world.mean() >= 0.5)
        for split, frame in (("train", train), ("test", test)):
            world = (frame.final_pi >= pi_threshold).to_numpy(dtype=int)
            methods = {
                "full_count_threshold": (frame.total_relevant.to_numpy() >= count_threshold).astype(int),
                "train_majority": np.full(len(frame), majority_action, dtype=int),
            }
            for method, predicted in methods.items():
                correct = predicted == world
                for row, target, prediction, is_correct in zip(frame.itertuples(index=False), world, predicted, correct):
                    rows.append({
                        "split": split, "method": method, "pi_threshold": pi_threshold,
                        "topic": row.topic, "total_relevant": row.total_relevant,
                        "final_pi": row.final_pi, "world": int(target), "prediction": int(prediction),
                        "correct": bool(is_correct), "selected_count_threshold": count_threshold,
                        "train_majority_action": majority_action,
                    })
                summaries.append({
                    "split": split, "method": method, "pi_threshold": pi_threshold,
                    "topics": len(frame), "accuracy": float(correct.mean()),
                    "selected_count_threshold": count_threshold,
                    "train_majority_action": majority_action,
                    "world_prevalence": float(world.mean()),
                    "count_final_pi_pearson": _pearson(frame.total_relevant, frame.final_pi),
                })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("test_csv", type=Path)
    parser.add_argument("decision_timeseries", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/dta_count_linkage"))
    parser.add_argument("--pi-thresholds", type=_thresholds, default=(0.2, 0.5, 0.8))
    args = parser.parse_args(argv)
    train = load_dta_split(args.train_csv, args.decision_timeseries)
    test = load_dta_split(args.test_csv, args.decision_timeseries)
    per_topic, summary = evaluate_count_world_linkage(train, test, args.pi_thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_topic.to_csv(args.output_dir / "per_topic.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "gflowstop_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "gflowstop_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
        "train_csv": str(args.train_csv.resolve()), "train_sha256": _sha256(args.train_csv),
        "test_csv": str(args.test_csv.resolve()), "test_sha256": _sha256(args.test_csv),
        "decision_timeseries": str(args.decision_timeseries.resolve()), "decision_timeseries_sha256": _sha256(args.decision_timeseries),
        "pi_thresholds": args.pi_thresholds,
        "interpretation": "full-information count-threshold upper bound; not a stopping-policy evaluation",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
