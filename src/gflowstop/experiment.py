from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from math import ceil
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .core import (
    ACTION_ACCEPT,
    ACTION_REJECT,
    CountBellmanTeacher,
    CountModel,
    ConditionalStoppingPolicy,
    Topic,
    certificate_action,
    count_bellman_teacher,
    count_state,
    decide_amortized,
    decide_bellman,
    decide_consecutive_misses,
    decide_evpi_bound,
    decide_fixed,
    decide_hit_probability_cutoff,
    decide_kneedle,
    decide_one_step_evsi,
    decide_qbcb,
    decide_score_cutoff,
    features,
    fit_count_model,
    full_information_utility,
    realised_utility,
)


Q_STARS = (0.3, 0.5, 0.7)
COSTS = (0.0005, 0.002, 0.005)


def load_pyterrier_rankings(cache: Path, max_depth: int, max_topics: int) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, dtype={"qid": str, "docno": str})
    import pyterrier as pt

    dataset = pt.datasets.get_dataset("vaswani")
    topics = dataset.get_topics().head(max_topics)
    retriever = pt.terrier.Retriever.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
    ranking = retriever.transform(topics)
    ranking = ranking.sort_values(["qid", "rank"]).groupby("qid", as_index=False).head(max_depth)
    qrels = dataset.get_qrels()[["qid", "docno", "label"]]
    ranking = ranking.merge(qrels, on=["qid", "docno"], how="left")
    ranking["label"] = ranking["label"].fillna(0).astype(int).gt(0).astype(int)
    keep = ["qid", "docno", "rank", "score", "label"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    ranking[keep].to_csv(cache, index=False)
    return ranking[keep]


def ranked_topics(ranking: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    records = []
    for qid, group in ranking.groupby("qid", sort=False):
        group = group.sort_values("rank")
        records.append((str(qid), group["label"].to_numpy(dtype=int), group["score"].to_numpy(dtype=float)))
    return records


def labelled_topics(records: list[tuple[str, np.ndarray, np.ndarray]], threshold: int) -> list[Topic]:
    return [Topic(qid, labels, int(labels.sum() >= threshold), scores) for qid, labels, scores in records]


def _stratified_indices(topics: list[Topic] | list[tuple[str, np.ndarray, np.ndarray]], folds: int, seed: int) -> list[np.ndarray]:
    if folds < 2 or folds > len(topics):
        raise ValueError("folds must be between two and the number of topics")
    counts = np.asarray([int(item.labels.sum()) if isinstance(item, Topic) else int(item[1].sum()) for item in topics])
    rng = np.random.default_rng(seed)
    bins = np.array_split(np.argsort(counts, kind="stable"), min(5, max(1, len(topics) // folds)))
    assigned = [[] for _ in range(folds)]
    for bin_indices in bins:
        for offset, index in enumerate(rng.permutation(bin_indices)):
            assigned[offset % folds].append(int(index))
    return [np.asarray(indices, dtype=int) for indices in assigned]


def outer_splits(records: list[tuple[str, np.ndarray, np.ndarray]], folds: int, repeats: int, seed: int):
    all_indices = np.arange(len(records))
    for repeat in range(repeats):
        for fold, test in enumerate(_stratified_indices(records, folds, seed + repeat)):
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            yield repeat, fold, train, test


def _teacher_cache(
    model: CountModel, topics: list[Topic], threshold: int, q_star: float, cost: float, temperature: float, states=None,
) -> dict[int, CountBellmanTeacher]:
    return {
        length: count_bellman_teacher(model, length, threshold, q_star, cost, temperature, (states or {}).get(length) or count_state(model, length, threshold))
        for length in {len(topic.labels) for topic in topics}
    }


def training_examples(
    topics: list[Topic], model: CountModel, threshold: int, temperature: float, policy: ConditionalStoppingPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect teacher states, then student-induced states in DAgger rounds."""
    xs, ys = [], []
    states = {length: count_state(model, length, threshold) for length in {len(topic.labels) for topic in topics}}
    caches = {
        (q_star, cost): _teacher_cache(model, topics, threshold, q_star, cost, temperature, states)
        for q_star in Q_STARS for cost in COSTS
    }
    for q_star in Q_STARS:
        for cost in COSTS:
            for topic in topics:
                teacher = caches[q_star, cost][len(topic.labels)]
                hits = 0
                for depth in range(len(topic.labels) + 1):
                    target = teacher.target(depth, hits)
                    xs.append(features(topic, model, threshold, depth, hits, q_star, cost, teacher.state))
                    ys.append(target)
                    forced = certificate_action(hits, depth, len(topic.labels), threshold)
                    action = int(np.argmax(target)) if policy is None or forced is not None or depth == len(topic.labels) else int(np.argmax(policy.probabilities(xs[-1])))
                    if action != 0:
                        break
                    hits += int(topic.labels[depth])
    return np.asarray(xs), np.asarray(ys)


def fit_policy(
    topics: list[Topic], model: CountModel, threshold: int, temperature: float, steps: int, dagger_rounds: int,
) -> ConditionalStoppingPolicy:
    x, target = training_examples(topics, model, threshold, temperature)
    policy = ConditionalStoppingPolicy().fit(x, target, steps=steps)
    for _ in range(dagger_rounds):
        student_x, student_target = training_examples(topics, model, threshold, temperature, policy)
        x, target = np.vstack((x, student_x)), np.vstack((target, student_target))
        policy = ConditionalStoppingPolicy().fit(x, target, steps=steps)
    return policy


def tune_parameter(
    topics: list[Topic], threshold: int, q_star: float, cost: float, candidates, rule, inner_folds: int, seed: int,
):
    scores = []
    folds = _stratified_indices(topics, min(inner_folds, len(topics)), seed)
    all_indices = np.arange(len(topics))
    for candidate in candidates:
        utilities = []
        for validation in folds:
            train = [topics[index] for index in np.setdiff1d(all_indices, validation, assume_unique=True)]
            model = fit_count_model(train)
            states = {
                length: count_state(model, length, threshold)
                for length in {len(topic.labels) for topic in train + [topics[index] for index in validation]}
            }
            for index in validation:
                topic = topics[index]
                depth, action, _ = rule(topic, model, states[len(topic.labels)], candidate)
                utilities.append(realised_utility(topic, depth, action, q_star, cost))
        scores.append(float(np.mean(utilities)))
    return candidates[int(np.argmax(scores))]


def evaluate_split(
    train: list[Topic], test: list[Topic], repeat: int, fold: int, threshold: int, temperature: float, steps: int,
    dagger_rounds: int, inner_folds: int, seed: int, include_adapted_qbcb: bool, qbcb_seeds: int,
) -> tuple[list[dict], CountModel]:
    model = fit_count_model(train)
    policy = fit_policy(train, model, threshold, temperature, steps, dagger_rounds)
    rows = []
    for q_star in Q_STARS:
        for cost in COSTS:
            states = {
                length: count_state(model, length, threshold)
                for length in {len(topic.labels) for topic in train + test}
            }
            teachers = _teacher_cache(model, test, threshold, q_star, cost, temperature, states)
            batch = tune_parameter(train, threshold, q_star, cost, (5, 10, 20, 40), lambda t, m, s, k: decide_evpi_bound(t, m, threshold, q_star, cost, batch=k, state=s), inner_folds, seed)
            budget = tune_parameter(train, threshold, q_star, cost, (10, 25, 50, 100, 200), lambda t, m, s, k: decide_fixed(t, m, threshold, q_star, k, state=s), inner_folds, seed)
            misses = tune_parameter(train, threshold, q_star, cost, (5, 10, 20, 40), lambda t, m, s, k: decide_consecutive_misses(t, m, threshold, q_star, k, state=s), inner_folds, seed)
            hit_cutoff = tune_parameter(train, threshold, q_star, cost, (0.002, 0.005, 0.01, 0.02, 0.05), lambda t, m, s, p: decide_hit_probability_cutoff(t, m, threshold, q_star, p, state=s), inner_folds, seed)
            score_cutoff = tune_parameter(train, threshold, q_star, cost, (0.25, 0.35, 0.45, 0.55, 0.65), lambda t, m, s, p: decide_score_cutoff(t, m, threshold, q_star, p, state=s), inner_folds, seed)
            kneedle = tune_parameter(train, threshold, q_star, cost, ((10, 1.5), (10, 2.0), (20, 1.5), (20, 2.0), (20, 3.0), (40, 2.0)), lambda t, m, s, p: decide_kneedle(t, m, threshold, q_star, *p, state=s), inner_folds, seed)
            methods = {
                "Amortized-count": lambda topic: decide_amortized(topic, model, policy, threshold, q_star, cost, states[len(topic.labels)]),
                "Count-Bellman": lambda topic: decide_bellman(topic, teachers[len(topic.labels)]),
                "One-step-EVSI": lambda topic: decide_one_step_evsi(topic, model, threshold, q_star, cost, states[len(topic.labels)]),
                "EVPI-bound": lambda topic: decide_evpi_bound(topic, model, threshold, q_star, cost, state=states[len(topic.labels)]),
                "Batch-EVPI-bound": lambda topic: decide_evpi_bound(topic, model, threshold, q_star, cost, batch=batch, state=states[len(topic.labels)]),
                "Fixed-budget": lambda topic: decide_fixed(topic, model, threshold, q_star, budget, state=states[len(topic.labels)]),
                "Zero-screen": lambda topic: decide_fixed(topic, model, threshold, q_star, 0, state=states[len(topic.labels)]),
                "Full-screen": lambda topic: decide_fixed(topic, model, threshold, q_star, len(topic.labels), certify=False, state=states[len(topic.labels)]),
                "Consecutive-miss": lambda topic: decide_consecutive_misses(topic, model, threshold, q_star, misses, state=states[len(topic.labels)]),
                "Hit-probability cutoff": lambda topic: decide_hit_probability_cutoff(topic, model, threshold, q_star, hit_cutoff, state=states[len(topic.labels)]),
                "BM25-score cutoff": lambda topic: decide_score_cutoff(topic, model, threshold, q_star, score_cutoff, state=states[len(topic.labels)]),
                "Kneedle": lambda topic: decide_kneedle(topic, model, threshold, q_star, *kneedle, state=states[len(topic.labels)]),
            }
            for name, rule in methods.items():
                for topic in test:
                    depth, action, belief = rule(topic)
                    utility = realised_utility(topic, depth, action, q_star, cost)
                    full_information = full_information_utility(topic, q_star)
                    rows.append({
                        "repeat": repeat, "fold": fold, "topic": topic.qid, "method": name, "q_star": q_star, "cost": cost,
                        "yield_threshold": threshold, "random_seed": None,
                        "tuned_batch": batch if name == "Batch-EVPI-bound" else None,
                        "tuned_budget": budget if name == "Fixed-budget" else None,
                        "tuned_misses": misses if name == "Consecutive-miss" else None,
                        "tuned_hit_cutoff": hit_cutoff if name == "Hit-probability cutoff" else None,
                        "tuned_score_cutoff": score_cutoff if name == "BM25-score cutoff" else None,
                        "tuned_kneedle": str(kneedle) if name == "Kneedle" else None,
                        "stop_depth": depth, "action": "accept" if action == ACTION_ACCEPT else "reject", "belief": belief,
                        "world": topic.world, "net_utility": utility, "full_information_utility": full_information,
                        "full_information_gap": full_information - utility,
                        "decision_correct": int(action == (ACTION_ACCEPT if topic.world else ACTION_REJECT)),
                    })
            if include_adapted_qbcb:
                for random_seed in range(qbcb_seeds):
                    for topic in test:
                        depth, action, belief = decide_qbcb(topic, model, threshold, q_star, 20, 0.8, 0.05, seed + random_seed)
                        utility = realised_utility(topic, depth, action, q_star, cost)
                        full_information = full_information_utility(topic, q_star)
                        rows.append({
                            "repeat": repeat, "fold": fold, "topic": topic.qid, "method": "QBCB-adapted", "q_star": q_star, "cost": cost,
                            "yield_threshold": threshold, "random_seed": random_seed,
                            "tuned_batch": None, "tuned_budget": None, "tuned_misses": None, "tuned_hit_cutoff": None,
                            "tuned_score_cutoff": None, "tuned_kneedle": None,
                            "stop_depth": depth, "action": "accept" if action == ACTION_ACCEPT else "reject", "belief": belief,
                            "world": topic.world, "net_utility": utility, "full_information_utility": full_information,
                            "full_information_gap": full_information - utility,
                            "decision_correct": int(action == (ACTION_ACCEPT if topic.world else ACTION_REJECT)),
                        })
    return rows, model


def bootstrap_interval(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def summarise(per_topic: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    rows = []
    grouped = per_topic.groupby(["method", "q_star", "cost"], sort=True)
    for number, ((method, q_star, cost), group) in enumerate(grouped):
        by_topic = group.groupby("topic", as_index=False)[["net_utility", "full_information_gap", "stop_depth", "decision_correct"]].mean()
        utility_ci = bootstrap_interval(by_topic.net_utility.to_numpy(), seed + number, replicates)
        depth_ci = bootstrap_interval(by_topic.stop_depth.to_numpy(), seed + 10_000 + number, replicates)
        accuracy_ci = bootstrap_interval(by_topic.decision_correct.to_numpy(), seed + 20_000 + number, replicates)
        rows.append({
            "method": method, "q_star": q_star, "cost": cost, "topics": len(by_topic),
            "mean_net_utility": by_topic.net_utility.mean(), "utility_ci_low": utility_ci[0], "utility_ci_high": utility_ci[1],
            "mean_full_information_gap": by_topic.full_information_gap.mean(), "mean_stop_depth": by_topic.stop_depth.mean(),
            "depth_ci_low": depth_ci[0], "depth_ci_high": depth_ci[1], "decision_accuracy": by_topic.decision_correct.mean(),
            "accuracy_ci_low": accuracy_ci[0], "accuracy_ci_high": accuracy_ci[1],
        })
    return pd.DataFrame(rows).sort_values(["q_star", "cost", "mean_net_utility"], ascending=[True, True, False])


def paired_comparison(per_topic: pd.DataFrame, method: str, comparator: str, seed: int, replicates: int) -> dict:
    selected = per_topic[per_topic.method.isin((method, comparator))]
    averages = selected.groupby(["topic", "method"], as_index=False).net_utility.mean().pivot(index="topic", columns="method", values="net_utility").dropna()
    difference = (averages[method] - averages[comparator]).to_numpy()
    ci = bootstrap_interval(difference, seed, replicates)
    rng = np.random.default_rng(seed + 1)
    null = (difference * rng.choice((-1.0, 1.0), size=(replicates, len(difference)))).mean(axis=1)
    return {
        "method": method, "comparator": comparator, "topics": len(difference), "mean_difference": float(difference.mean()),
        "bootstrap_95_ci": list(ci), "paired_sign_flip_p": float((np.count_nonzero(np.abs(null) >= abs(difference.mean())) + 1) / (replicates + 1)),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated held-out amortized Bayesian stopping on PyTerrier Vaswani rankings.")
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--max-topics", type=int, default=93)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--include-adapted-qbcb", action="store_true")
    parser.add_argument("--qbcb-seeds", type=int, default=50)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.inner_folds < 2 or args.bootstrap_replicates < 1 or args.qbcb_seeds < 1:
        raise SystemExit("--inner-folds, --bootstrap-replicates, and --qbcb-seeds must be positive")
    cache = args.data_dir / f"vaswani_bm25_top{args.max_depth}_{args.max_topics}.csv"
    records = ranked_topics(load_pyterrier_rankings(cache, args.max_depth, args.max_topics))
    rows, split_details = [], []
    for repeat, fold, train_indices, test_indices in outer_splits(records, args.folds, args.repeats, args.seed):
        train_records = [records[index] for index in train_indices]
        test_records = [records[index] for index in test_indices]
        threshold = ceil(np.median([labels.sum() for _, labels, _ in train_records]))
        train, test = labelled_topics(train_records, threshold), labelled_topics(test_records, threshold)
        split_rows, model = evaluate_split(
            train, test, repeat, fold, threshold, args.temperature, args.steps, args.dagger_rounds, args.inner_folds,
            args.seed + repeat * args.folds + fold, args.include_adapted_qbcb, args.qbcb_seeds,
        )
        rows.extend(split_rows)
        split_details.append({
            "repeat": repeat, "fold": fold, "threshold": threshold, "model": {"alpha": model.alpha, "beta": model.beta},
            "train_topic_ids": [records[index][0] for index in train_indices], "test_topic_ids": [records[index][0] for index in test_indices],
        })
    per_topic = pd.DataFrame(rows)
    summary = summarise(per_topic, args.seed, args.bootstrap_replicates)
    comparison = paired_comparison(per_topic, "Amortized-count", "Count-Bellman", args.seed, args.bootstrap_replicates)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    per_topic.to_csv(args.results_dir / "per_topic.csv", index=False)
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    (args.results_dir / "splits.json").write_text(json.dumps(split_details, indent=2) + "\n")
    (args.results_dir / "statistics.json").write_text(json.dumps({"amortized_vs_count_bellman": comparison}, indent=2) + "\n")
    metadata = {
        "method": "Amortized Bayesian Stopping: log-linear policy distilled from an exact beta-binomial count-state Bellman teacher",
        "not_a_gflownet": True,
        "dataset": "PyTerrier vaswani", "retrieval": "PyTerrier ready-made terrier_stemmed BM25",
        "unjudged_documents": "treated as non-relevant", "max_depth": args.max_depth, "topics": len(records),
        "evaluation": {"folds": args.folds, "repeats": args.repeats, "inner_folds": args.inner_folds, "seed": args.seed, "split_file": "splits.json"},
        "teacher": {"model": "beta-binomial remaining-count", "hard_certificates": True, "temperature": args.temperature},
        "student": {"features": 11, "steps": args.steps, "dagger_rounds": args.dagger_rounds},
        "conditions": {"q_star": Q_STARS, "cost": COSTS}, "bootstrap_replicates": args.bootstrap_replicates,
        "versions": {"python": platform.python_version(), "numpy": _package_version("numpy"), "pandas": _package_version("pandas"), "pyterrier": _package_version("python-terrier")},
        "git_revision": _git_revision(),
    }
    (args.results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.results_dir / "report.md").write_text(
        "# Held-out amortized Bayesian stopping results\n\n"
        "This experiment is cost-sensitive Bayesian finite-horizon stopping with policy distillation, not a GFlowNet. "
        "The teacher solves the fitted beta-binomial remaining-count model exactly and hard certificates force acceptance once the yield threshold is reached and rejection once it is unattainable. "
        "The fitted model can still be wrong for real rankings.\n\n"
        "Nested repeated stratified folds are stored in `splits.json`; intervals resample topics after averaging repeated folds, stochastic seeds, and the nine decision conditions. "
        "Direct net utility is primary. The full-information gap is only a labelled upper-bound comparison, not a hindsight stopping oracle.\n\n"
        f"The observed Amortized-count minus Count-Bellman mean utility is {comparison['mean_difference']:.5f}, with bootstrap 95% CI "
        f"[{comparison['bootstrap_95_ci'][0]:.5f}, {comparison['bootstrap_95_ci'][1]:.5f}] and paired sign-flip p={comparison['paired_sign_flip_p']:.3f}.\n\n"
        "| Method | q* | cost | topics | mean net utility (95% CI) | mean full-information gap | mean stop depth (95% CI) | decision accuracy (95% CI) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(
            f"| {row.method} | {row.q_star:.1f} | {row.cost:.4f} | {row.topics} | {row.mean_net_utility:.4f} [{row.utility_ci_low:.4f}, {row.utility_ci_high:.4f}] | "
            f"{row.mean_full_information_gap:.4f} | {row.mean_stop_depth:.1f} [{row.depth_ci_low:.1f}, {row.depth_ci_high:.1f}] | "
            f"{row.decision_accuracy:.3f} [{row.accuracy_ci_low:.3f}, {row.accuracy_ci_high:.3f}] |"
            for row in summary.itertuples()
        ) + "\n"
    )
    print(f"Wrote {len(per_topic)} evaluation rows and {len(summary)} summary rows to {args.results_dir}")


if __name__ == "__main__":
    main()
