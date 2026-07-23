"""Adapter for fixed public ranking simulations used in external recall tests.

The Repke et al. framework stores each ranking as matching JSON metadata and
Feather data files.  This module preserves that ranking order and emits rows
that can be concatenated with its stopping-method output.  It evaluates a
posterior statement about target recall; it does not claim a frequentist
recall guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import beta

from .toys import Episode, RankLatentBelief


REPKE_FRAMEWORK_URL = "https://github.com/destiny-evidence/stopping-methods"
POSTERIOR_APPROXIMATION = "normal-mixture"


@dataclass(frozen=True)
class ExternalRanking:
    """One fully labelled evaluation ranking from an external framework."""

    dataset: str
    simulation: str
    key: str
    labels: np.ndarray
    scores: np.ndarray
    info_path: Path
    ranking_path: Path

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or self.scores.ndim != 1 or len(self.labels) != len(self.scores):
            raise ValueError("external labels and scores must be same-length vectors")
        if not len(self.labels) or not np.isin(self.labels, (0, 1)).all():
            raise ValueError("external rankings must have non-empty binary labels")

    def episode(self) -> Episode:
        digest = hashlib.sha256(self.key.encode()).digest()
        return Episode(
            labels=self.labels,
            scores=self.scores,
            latent={"dataset": self.dataset, "simulation": self.simulation},
            environment="repke_fixed_ranking",
            seed=int.from_bytes(digest[:8], "big"),
        )


def _finite_scores(values: Iterable[object]) -> np.ndarray:
    scores = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(scores)
    fill = float(np.median(scores[finite])) if finite.any() else 0.0
    return np.where(finite, scores, fill)


def repke_ranking_from_frame(info: dict, frame: pd.DataFrame, info_path: Path, ranking_path: Path) -> ExternalRanking:
    """Build an episode from one Repke ranking frame without changing its order."""
    required = {"label", "score", "batch", "order"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Repke ranking {ranking_path} is missing columns: {sorted(missing)}")
    ordered = frame.sort_values(["batch", "order"], kind="stable")
    labels = pd.to_numeric(ordered["label"], errors="raise").to_numpy(dtype=int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Repke ranking {ranking_path} has non-binary labels")
    params = (info.get("batches") or [{}])[0].get("params", {})
    dataset = str(params.get("dataset", info.get("dataset", info_path.stem)))
    simulation = str(info.get("repeat", info.get("simulation", "0")))
    return ExternalRanking(
        dataset=dataset,
        simulation=simulation,
        key=info_path.stem,
        labels=labels,
        scores=_finite_scores(ordered["score"]),
        info_path=info_path,
        ranking_path=ranking_path,
    )


def load_repke_rankings(ranking_dir: Path) -> list[ExternalRanking]:
    """Read all matching ``*.json``/``*.feather`` simulation outputs."""
    ranking_dir = Path(ranking_dir)
    rankings: list[ExternalRanking] = []
    for info_path in sorted(ranking_dir.glob("*.json")):
        ranking_path = info_path.with_suffix(".feather")
        if not ranking_path.exists():
            continue
        with info_path.open() as handle:
            info = json.load(handle)
        try:
            frame = pd.read_feather(ranking_path)
        except ImportError as error:  # pragma: no cover - depends on optional reader at runtime
            raise RuntimeError("Repke Feather inputs require pyarrow; install the project dependencies") from error
        rankings.append(repke_ranking_from_frame(info, frame, info_path, ranking_path))
    if not rankings:
        raise FileNotFoundError(f"no paired Repke JSON/Feather rankings found in {ranking_dir}")
    return rankings


def _checkpoints(length: int, interval: int) -> tuple[int, ...]:
    if interval < 1:
        raise ValueError("check interval must be positive")
    points = list(range(interval, length + 1, interval))
    if not points or points[-1] != length:
        points.append(length)
    return tuple(points)


def _folds(rankings: list[ExternalRanking], folds: int) -> dict[str, int]:
    datasets = sorted({ranking.dataset for ranking in rankings})
    if len(datasets) < 2:
        raise ValueError("external cross-fitting requires rankings from at least two datasets")
    if folds < 2:
        raise ValueError("external cross-fitting requires at least two folds")
    folds = min(folds, len(datasets))
    return {dataset: index % folds for index, dataset in enumerate(datasets)}


def _suffix_moments(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(probabilities), probabilities.shape[1] + 1), dtype=float)
    variances = means.copy()
    means[:, :-1] = np.cumsum(probabilities[:, ::-1], axis=1)[:, ::-1]
    variances[:, :-1] = np.cumsum((probabilities * (1.0 - probabilities))[:, ::-1], axis=1)[:, ::-1]
    return means, variances


def _recall_probability(weights: np.ndarray, means: np.ndarray, variances: np.ndarray, depth: int, hits: int, target: float) -> float:
    if not 0.0 < target < 1.0:
        raise ValueError("target recall must be strictly between zero and one")
    limit = int(np.floor(hits * (1.0 - target) / target + 1e-12))
    mean, variance = means[:, depth], variances[:, depth]
    standard_deviation = np.sqrt(variance)
    cdf = np.where(
        standard_deviation > 1e-12,
        ndtr((limit + 0.5 - mean) / np.maximum(standard_deviation, 1e-12)),
        (mean <= limit).astype(float),
    )
    return float(np.clip(weights @ cdf, 0.0, 1.0))


def _failure_ucb(failures: int, total: int, confidence: float = 0.95) -> float:
    if not 0 <= failures <= total or total < 1:
        raise ValueError("invalid binomial counts")
    return 1.0 if failures == total else float(beta.ppf(confidence, failures + 1, total - failures))


def _trace_ranking(
    ranking: ExternalRanking,
    model: RankLatentBelief,
    method: str,
    fold: int,
    target: float,
    alpha: float,
    check_interval: int,
) -> list[dict[str, object]]:
    """Evaluate every permitted stopping check without exposing future labels."""
    episode = ranking.episode()
    observation = episode.observation(0, 0, threshold=1)
    state = model.initial_state(observation)
    probabilities = model._probability_matrix(observation).copy()
    means, variances = _suffix_moments(probabilities)
    confidence = 1.0 - alpha
    depths = _checkpoints(len(episode.labels), check_interval)
    oracle_depth = int(np.flatnonzero(np.cumsum(episode.labels) >= np.ceil(target * episode.labels.sum()))[0] + 1) if episode.labels.sum() else 0
    weights = state.weights.copy()
    hits, depth, previous = 0, 0, 0
    method_hash = hashlib.sha256(json.dumps({"method": method, "target": target, "alpha": alpha}, sort_keys=True).encode()).hexdigest()
    rows: list[dict[str, object]] = []
    for batch_i, checkpoint in enumerate(depths):
        while depth < checkpoint:
            label = int(episode.labels[depth])
            likelihood = probabilities[:, depth] if label else 1.0 - probabilities[:, depth]
            weights *= likelihood
            weights /= weights.sum()
            hits += label
            depth += 1
        posterior = _recall_probability(weights, means, variances, depth, hits, target)
        recall = 1.0 if not episode.labels.sum() else float(hits / episode.labels.sum())
        rows.append({
            "dataset": ranking.dataset,
            "sim-rep": ranking.simulation,
            "sim_key": ranking.key,
            "fold": fold,
            "batch_i": batch_i,
            "n_total": len(episode.labels),
            "n_seen": depth,
            "n_unseen": len(episode.labels) - depth,
            "n_incl": int(episode.labels.sum()),
            "n_incl_seen": hits,
            "n_records_batch": depth - previous,
            "method": method,
            "method-KEY": method,
            "method-hash": method_hash,
            "method-recall_target": target,
            "method-confidence_level": confidence,
            "method-posterior_approximation": POSTERIOR_APPROXIMATION,
            "method-posterior_recall_probability": posterior,
            "safe_to_stop": bool(posterior >= confidence),
            "realised_recall": recall,
            "recall_oracle_depth": oracle_depth,
            "info_path": str(ranking.info_path),
            "ranking_path": str(ranking.ranking_path),
        })
        previous = depth
    return rows


def _first_stops(trace: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "sim-rep", "sim_key", "fold", "method", "method-hash", "method-recall_target", "method-confidence_level"]
    rows: list[dict[str, object]] = []
    for _key, group in trace.groupby(keys, sort=False):
        group = group.sort_values("n_seen", kind="stable")
        stopping = group[group["safe_to_stop"]]
        row = (stopping.iloc[0] if len(stopping) else group.iloc[-1]).to_dict()
        row["stopped_by_posterior"] = bool(len(stopping))
        row["review_fraction"] = float(row["n_seen"] / row["n_total"])
        row["failed"] = bool(row["realised_recall"] < row["method-recall_target"])
        row["excess_effort"] = int(row["n_seen"] - row["recall_oracle_depth"])
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(first_stops: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method-recall_target", "method-confidence_level"]
    rows: list[dict[str, object]] = []
    for key, group in first_stops.groupby(keys, sort=False):
        by_dataset = []
        for _dataset, cell in group.groupby("dataset", sort=False):
            by_dataset.append(_failure_ucb(int(cell["failed"].sum()), len(cell)))
        failures = int(group["failed"].sum())
        rows.append({
            "method": key[0],
            "target_recall": key[1],
            "confidence_level": key[2],
            "runs": len(group),
            "failure_rate": float(group["failed"].mean()),
            "failure_ucb_95": _failure_ucb(failures, len(group)),
            "worst_dataset_failure_ucb_95": float(max(by_dataset)),
            "mean_review_fraction": float(group["review_fraction"].mean()),
            "mean_excess_effort": float(group["excess_effort"].mean()),
            "mean_realised_recall": float(group["realised_recall"].mean()),
        })
    return pd.DataFrame(rows)


def evaluate_repke_rankings(
    rankings: list[ExternalRanking],
    targets: tuple[float, ...] = (0.8, 0.9, 0.95),
    alphas: tuple[float, ...] = (0.05, 0.10),
    check_interval: int = 100,
    folds: int = 5,
    include_local_scores: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-fit rank beliefs and return Repke-compatible traces and summaries.

    Dataset-level folds ensure that a ranking's labels never fit its belief
    model.  The primary method intentionally uses rank only: score means were
    the synthetic gain but are not comparable across real queries.  A local
    score ablation is opt-in and normalises within each ranking.
    """
    fold_by_dataset = _folds(rankings, folds)
    rows: list[dict[str, object]] = []
    method_options = [("GFlowStop-R-rank-normal-posterior", False)]
    if include_local_scores:
        method_options.append(("GFlowStop-R-rank-local-normal-posterior", True))
    for fold in sorted(set(fold_by_dataset.values())):
        train = [ranking.episode() for ranking in rankings if fold_by_dataset[ranking.dataset] != fold]
        test = [ranking for ranking in rankings if fold_by_dataset[ranking.dataset] == fold]
        for method, local_scores in method_options:
            model = RankLatentBelief(use_scores=local_scores, use_score_mean=False, use_local_scores=local_scores).fit(train)
            for ranking in test:
                for target in targets:
                    for alpha in alphas:
                        rows.extend(_trace_ranking(ranking, model, method, fold, target, alpha, check_interval))
    trace = pd.DataFrame(rows)
    first_stops = _first_stops(trace)
    return trace, first_stops, _summary(first_stops)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_repke(
    ranking_dir: Path,
    output_dir: Path,
    targets: tuple[float, ...],
    alphas: tuple[float, ...],
    check_interval: int,
    folds: int,
    include_local_scores: bool,
    framework_revision: str,
) -> None:
    rankings = load_repke_rankings(ranking_dir)
    trace, first_stops, summary = evaluate_repke_rankings(rankings, targets, alphas, check_interval, folds, include_local_scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_dir / "repke_trace.csv", index=False)
    first_stops.to_csv(output_dir / "repke_first_stops.csv", index=False)
    summary.to_csv(output_dir / "repke_summary.csv", index=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
    metadata = {
        "source_framework": REPKE_FRAMEWORK_URL,
        "framework_revision": framework_revision,
        "gflowstop_revision": revision,
        "gflowstop_dirty": dirty,
        "external_benchmark_sha256": _sha256(Path(__file__)),
        "ranking_dir": str(Path(ranking_dir).resolve()),
        "ranking_files": {str(ranking.ranking_path): _sha256(ranking.ranking_path) for ranking in rankings},
        "info_files": {str(ranking.info_path): _sha256(ranking.info_path) for ranking in rankings},
        "targets": targets,
        "alphas": alphas,
        "check_interval": check_interval,
        "folds": folds,
        "include_local_scores": include_local_scores,
        "posterior_approximation": POSTERIOR_APPROXIMATION,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-fitted GFlowStop-R on Repke fixed rankings")
    parser.add_argument("ranking_dir", type=Path, help="directory containing paired Repke *.json and *.feather rankings")
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/repke"))
    parser.add_argument("--targets", type=_float_tuple, default=(0.8, 0.9, 0.95))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.05, 0.10))
    parser.add_argument("--check-interval", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-local-scores", action="store_true")
    parser.add_argument("--framework-revision", required=True, help="commit that generated or evaluated the supplied rankings")
    args = parser.parse_args(argv)
    run_repke(args.ranking_dir, args.output_dir, args.targets, args.alphas, args.check_interval, args.folds, args.include_local_scores, args.framework_revision)


if __name__ == "__main__":
    main()
