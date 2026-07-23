import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import Table, ipc

from gflowstop.external_benchmark import _checkpoints, _published_repke_run, _recall_probability, evaluate_repke_rankings, load_repke_rankings, repke_ranking_from_frame
from gflowstop.external_utility import evaluate_external_greedy, evaluate_k1_split, fit_score_mean_prior
from gflowstop.toys import Episode


def _ranking(dataset: str, repeat: int, labels: list[int]):
    frame = pd.DataFrame({
        "label": labels,
        "score": [np.nan, *np.linspace(0.2, 0.8, len(labels) - 1)],
        "batch": [1] * len(labels),
        "order": list(reversed(range(len(labels)))),
    })
    return repke_ranking_from_frame(
        {"repeat": repeat, "batches": [{"params": {"dataset": dataset}}]},
        frame,
        Path(f"{dataset}-{repeat}.json"),
        Path(f"{dataset}-{repeat}.feather"),
    )


def test_repke_adapter_preserves_ranking_and_uses_dataset_level_crossfit():
    rankings = [
        _ranking("a", 0, [1, 0, 0, 0]),
        _ranking("b", 0, [0, 1, 0, 0]),
        _ranking("c", 0, [0, 0, 1, 0]),
    ]
    assert rankings[0].labels.tolist() == [0, 0, 0, 1]
    assert np.isfinite(rankings[0].scores).all()

    trace, first_stops, summary = evaluate_repke_rankings(
        rankings, targets=(0.8,), alphas=(0.1,), check_interval=2, folds=3,
    )

    assert len(trace) == 6
    assert set(trace["method"]) == {"GFlowStop-R-rank-normal-posterior"}
    assert set(trace["method-posterior_approximation"]) == {"normal-mixture"}
    assert set(first_stops["fold"]) == {0, 1, 2}
    assert (first_stops["n_seen"] <= first_stops["n_total"]).all()
    assert 0.0 <= float(summary.iloc[0]["worst_dataset_failure_ucb_95"]) <= 1.0


def test_repke_loader_reads_the_framework_json_feather_pair(tmp_path):
    info_path = tmp_path / "run.json"
    info_path.write_text(json.dumps({"repeat": 2, "batches": [{"params": {"dataset": "demo"}}]}))
    frame = pd.DataFrame({
        "label": [1, 0], "score": [0.9, 0.1], "batch": [1, 1], "order": [1, 0],
    })
    with (tmp_path / "run.feather").open("wb") as handle:
        with ipc.new_file(handle, Table.from_pandas(frame).schema) as writer:
            writer.write_table(Table.from_pandas(frame))

    rankings = load_repke_rankings(tmp_path, published_only=False)

    assert len(rankings) == 1
    assert rankings[0].dataset == "demo"
    assert rankings[0].labels.tolist() == [0, 1]


def test_recall_cap_is_not_rounded_down_by_binary_floating_point():
    probability = _recall_probability(
        np.array([1.0]), np.array([[1.0]]), np.array([[0.0]]), depth=0, hits=4, target=0.8,
    )
    assert probability == 1.0


def test_published_repke_selection_rejects_old_batch_protocols():
    info = {"batches": [{"params": {"batch": {
        "num_random_init": 500, "dyn_min_batch_incl": 2, "dyn_min_batch_size": 25,
        "dyn_max_batch_size": 200, "dyn_growth_rate": 0.1,
    }}}]}
    assert _published_repke_run(info)
    info["batches"][0]["params"]["batch"]["dyn_max_batch_size"] = 2000
    assert not _published_repke_run(info)


def test_repke_checkpoints_match_framework_batch_boundaries():
    # RankedDataset.bounds(batch_size=15) rounds an evenly spaced partition.
    # Literal 15-document checks would add an illegal checkpoint at 30 here.
    assert _checkpoints(31, 15) == (15, 31)


def test_score_mean_prior_is_fitted_only_from_training_worlds():
    train = [
        Episode(np.array([1]), np.array([10.0]), {}, "test", 1),
        Episode(np.array([1]), np.array([9.0]), {}, "test", 2),
        Episode(np.array([0]), np.array([-9.0]), {}, "test", 3),
        Episode(np.array([0]), np.array([-10.0]), {}, "test", 4),
    ]
    prior = fit_score_mean_prior(train)
    assert prior.predict(np.array([8.0])) > prior.predict(np.array([-8.0]))


def test_k1_utility_summary_scores_rejection_as_correct_for_zero_worlds():
    train = [
        Episode(np.array([1]), np.array([10.0]), {}, "test", 1),
        Episode(np.array([1]), np.array([9.0]), {}, "test", 2),
        Episode(np.array([0]), np.array([-9.0]), {}, "test", 3),
        Episode(np.array([0]), np.array([-10.0]), {}, "test", 4),
    ]
    per_topic, summary = evaluate_k1_split(train, train, (0.5,), (0.1,))
    expected = per_topic.groupby("method").apply(
        lambda rows: (rows.action == np.where(rows.world, 1, 2)).mean(), include_groups=False,
    )
    assert np.allclose(summary.set_index("method").decision_accuracy.sort_index(), expected.sort_index())


def test_external_greedy_runner_calls_the_supplied_implementation(tmp_path):
    rankings = tmp_path / "rankings.csv"
    pd.DataFrame({
        "Topic": ["a", "a", "b"], "Rank": [1, 2, 1], "label": [1, 0, 0],
        "prob_iso_z": [0.9, 0.1, 0.2],
    }).to_csv(rankings, index=False)
    implementation = tmp_path / "greedy.py"
    implementation.write_text(
        "from types import SimpleNamespace\n"
        "def simulate_topic(rows, cost, q_star, probability_column, policy):\n"
        "    assert policy == 'greedy' and probability_column == 'prob_iso_z'\n"
        "    return SimpleNamespace(net_utility=float(rows[probability_column].sum() - cost), stop_depth=len(rows))\n"
    )

    result = evaluate_external_greedy(rankings, implementation, (0.5,), (0.1,), "prob_iso_z")

    assert result.to_dict("records") == [
        {"topic": "a", "q_star": 0.5, "cost": 0.1, "net_utility": 0.9, "stop_depth": 2},
        {"topic": "b", "q_star": 0.5, "cost": 0.1, "net_utility": 0.1, "stop_depth": 1},
    ]
