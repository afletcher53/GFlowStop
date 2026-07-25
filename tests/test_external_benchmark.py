import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow import Table, ipc

from gflowstop.external_benchmark import _checkpoints, _published_repke_run, _recall_probability, evaluate_repke_rankings, load_repke_rankings, repke_ranking_from_frame
from gflowstop.external_dta import evaluate_count_world_linkage, load_dta_split
from gflowstop.external_horizon import _next_hit_log_loss, evaluate_horizon_sweep, main as horizon_main
from gflowstop.external_utility import evaluate_external_greedy, evaluate_k1_split, fit_score_mean_prior
from gflowstop.toys import Episode, RankLatentBelief


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


def test_imposed_world_horizon_sweep_uses_train_fit_and_emits_root_diagnostics():
    train = [
        Episode(np.array(labels), np.linspace(1.0, 0.1, 4), {"topic": f"train-{index}"}, "test", index)
        for index, labels in enumerate(([1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]))
    ]
    test = [
        Episode(np.array(labels), np.linspace(0.9, 0.2, 4), {"topic": f"test-{index}"}, "test", index)
        for index, labels in enumerate(([1, 0, 0, 0], [0, 0, 1, 0]))
    ]

    per_topic, summary, calibration, roots = evaluate_horizon_sweep(
        train, test, train_sizes=(2, 4), horizons=(1, 2), q_stars=(0.5,), kappas=(0.1,), seed=7,
    )

    assert len(per_topic) == 8
    assert per_topic.threshold.between(1, 4).all()
    assert set(summary.h_star) <= {1, 2}
    assert np.isfinite(calibration.next_hit_log_loss).all()
    assert roots.root_continue_fraction.between(0.0, 1.0).all()


def test_next_hit_log_loss_reuses_the_root_observation_without_changing_the_score():
    episode = Episode(np.array([1, 0, 1, 0]), np.linspace(0.9, 0.2, 4), {}, "test", 0)
    model = RankLatentBelief(use_scores=True, use_score_mean=False, use_local_scores=True).fit([episode])

    expected, state, hits = [], model.initial_state(episode.observation(0, 0, 1)), 0
    for depth, label in enumerate(episode.labels):
        observation = episode.observation(depth, hits, 1)
        probability = np.clip(model.next_hit_probability(state, observation), 1e-9, 1.0 - 1e-9)
        expected.append(-np.log(probability if label else 1.0 - probability))
        state = model.update(state, observation, int(label))
        hits += int(label)

    assert _next_hit_log_loss([episode], model) == np.mean(expected)


def test_imposed_world_horizon_respects_reached_threshold_certificate():
    episode = Episode(np.array([1, 0]), np.array([1.0, 0.0]), {"topic": "test"}, "test", 0)
    per_topic, _summary, _calibration, _roots = evaluate_horizon_sweep(
        [episode, Episode(np.array([0, 0]), np.array([1.0, 0.0]), {"topic": "train"}, "test", 1)], [episode],
        train_sizes=(2,), horizons=(2,), q_stars=(0.5,), kappas=(0.0,), seed=3,
    )
    assert per_topic.action.item() == 1


def test_imposed_world_horizon_command_writes_auditable_artifacts(tmp_path):
    train_csv, test_csv, output_dir = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "results"
    pd.DataFrame({
        "Topic": ["a"] * 4 + ["b"] * 4,
        "Rank": list(range(1, 5)) * 2,
        "Score": list(np.linspace(1.0, 0.1, 4)) * 2,
        "label": [1, 0, 0, 0, 0, 1, 0, 0],
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "Topic": ["c"] * 4 + ["d"] * 4,
        "Rank": list(range(1, 5)) * 2,
        "Score": list(np.linspace(1.0, 0.1, 4)) * 2,
        "label": [0, 0, 1, 0, 0, 0, 0, 0],
    }).to_csv(test_csv, index=False)

    horizon_main([str(train_csv), str(test_csv), "--output-dir", str(output_dir), "--train-sizes", "2", "--horizons", "1", "--q-stars", "0.5", "--kappas", "0.1", "--expected-train-topics", "2", "--expected-test-topics", "2", "--require-common-length"])

    assert {path.name for path in output_dir.iterdir()} == {"per_topic.csv", "utility_summary.csv", "calibration.csv", "root_diagnostics.csv", "gates.json", "metadata.json"}
    assert json.loads((output_dir / "gates.json").read_text())["root_diagnostics"]["status"] == "inspect_before_interpreting_horizon_curve"


def test_imposed_world_preflight_does_not_write_a_utility_curve(tmp_path):
    train_csv, test_csv, output_dir = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "preflight"
    frame = pd.DataFrame({"Topic": ["a"] * 4 + ["b"] * 4, "Rank": list(range(1, 5)) * 2, "Score": [1.0, 0.7, 0.4, 0.1] * 2, "label": [1, 0, 0, 0, 0, 1, 0, 0]})
    frame.to_csv(train_csv, index=False)
    frame.assign(Topic=["c"] * 4 + ["d"] * 4).to_csv(test_csv, index=False)

    horizon_main([str(train_csv), str(test_csv), "--output-dir", str(output_dir), "--train-sizes", "2", "--horizons", "1,2", "--q-stars", "0.5", "--kappas", "0.1", "--root-only", "--expected-train-topics", "2", "--expected-test-topics", "2", "--require-common-length"])

    assert not (output_dir / "utility_summary.csv").exists()
    assert json.loads((output_dir / "gates.json").read_text())["root_diagnostics"]["status"] == "preflight_complete"


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


def test_dta_linkage_selects_the_count_world_from_training_only(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    timeseries = tmp_path / "timeseries.csv"
    pd.DataFrame({
        "Topic": ["a.csv"] * 3 + ["b.csv"] * 3,
        "label": [1, 1, 1, 0, 0, 0],
    }).to_csv(train_csv, index=False)
    pd.DataFrame({
        "Topic": ["c.csv"] * 3 + ["d.csv"] * 3,
        "label": [1, 1, 1, 0, 0, 0],
    }).to_csv(test_csv, index=False)
    pd.DataFrame({
        "topic": ["a", "a", "b", "b", "c", "d"],
        "depth": [1, 2, 1, 2, 1, 1],
        "pi": [0.4, 0.9, 0.6, 0.1, 0.1, 0.9],
    }).to_csv(timeseries, index=False)

    train = load_dta_split(train_csv, timeseries)
    test = load_dta_split(test_csv, timeseries)
    per_topic, summary = evaluate_count_world_linkage(train, test, (0.5,))

    count_test = per_topic[(per_topic.split == "test") & (per_topic.method == "full_count_threshold")]
    assert set(count_test.selected_count_threshold) == {1}
    assert count_test.correct.tolist() == [False, False]
    assert summary[(summary.split == "test") & (summary.method == "full_count_threshold")].accuracy.item() == 0.0
