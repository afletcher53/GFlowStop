import itertools

import numpy as np

from gflowstop.core import (
    ACTION_ACCEPT,
    ACTION_CONTINUE,
    ACTION_REJECT,
    CountModel,
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
    fit_count_model,
    terminal_values,
)


class ContinuePolicy:
    def probabilities(self, _):
        return np.array([1.0, 0.0, 0.0])


def brute_value(model, length, threshold, q_star, cost, depth=0, hits=0):
    forced = certificate_action(hits, depth, length, threshold)
    if forced is not None:
        return terminal_values(model, length, threshold, depth, hits, q_star)[forced]
    stop = terminal_values(model, length, threshold, depth, hits, q_star).max()
    if depth == length:
        return stop
    p_hit = model.p_hit(depth, hits)
    return max(stop, -cost + p_hit * brute_value(model, length, threshold, q_star, cost, depth + 1, hits + 1) + (1.0 - p_hit) * brute_value(model, length, threshold, q_star, cost, depth + 1, hits))


def test_count_bellman_matches_brute_force_on_small_problem():
    model = CountModel(alpha=1.5, beta=3.0)
    teacher = count_bellman_teacher(model, length=6, threshold=3, q_star=0.4, cost=0.03, temperature=0.01)
    assert np.isclose(teacher.values[0, 0], brute_value(model, 6, 3, 0.4, 0.03))
    assert np.allclose(teacher.targets[0, 0].sum(), 1.0)


def test_certificates_force_actions_and_save_screening():
    model = CountModel(1.0, 1.0)
    positive = Topic("positive", np.array([1, 1, 0, 0]), world=1, scores=np.ones(4))
    negative = Topic("negative", np.array([0, 0, 0, 0]), world=0, scores=np.ones(4))
    assert decide_amortized(positive, model, ContinuePolicy(), 2, 0.5, 0.01)[:2] == (2, ACTION_ACCEPT)
    assert decide_amortized(negative, model, ContinuePolicy(), 3, 0.5, 0.01)[:2] == (2, ACTION_REJECT)


def test_full_observation_is_always_correct():
    model = CountModel(1.0, 1.0)
    for labels in itertools.product((0, 1), repeat=5):
        topic = Topic("all", np.asarray(labels), int(sum(labels) >= 3), scores=np.ones(5))
        depth, action, _ = decide_fixed(topic, model, 3, 0.5, budget=5, certify=False)
        assert depth == 5
        assert action == (ACTION_ACCEPT if topic.world else ACTION_REJECT)


def test_cost_and_probability_boundaries_are_finite_and_sane():
    model = CountModel(1e-6, 1e-6)
    teacher = count_bellman_teacher(model, length=5, threshold=3, q_star=0.5, cost=10.0, temperature=0.01)
    assert np.isfinite(teacher.values[0, 0])
    assert np.argmax(teacher.action_values(0, 0)) != ACTION_CONTINUE
    cheap = count_bellman_teacher(CountModel(1.0, 1.0), length=5, threshold=3, q_star=0.5, cost=0.0, temperature=0.01)
    assert np.argmax(cheap.action_values(0, 0)) == ACTION_CONTINUE


def test_variable_lengths_and_count_monotonicity_work():
    model = CountModel(2.0, 5.0)
    short = Topic("short", np.array([0, 0, 0]), world=0, scores=np.ones(3))
    impossible = count_bellman_teacher(model, length=3, threshold=4, q_star=0.5, cost=0.01, temperature=0.01)
    assert decide_bellman(short, impossible)[:2] == (0, ACTION_REJECT)
    assert model.high_yield_probability(10, 4, 4, 2) >= model.high_yield_probability(10, 4, 4, 1)


def test_count_state_table_matches_the_beta_binomial_predictive_tail():
    model = CountModel(1.5, 3.0)
    state = count_state(model, 8, 4)
    for depth in range(9):
        for hits in range(depth + 1):
            assert np.isclose(state.belief[depth, hits], model.high_yield_probability(8, 4, depth, hits))


def test_baselines_return_valid_certified_outcomes():
    model = CountModel(2.0, 8.0)
    topic = Topic("fixed", np.array([1, 0, 0, 1, 0, 0, 0, 0]), world=1, scores=np.linspace(1.0, 0.1, 8))
    outcomes = [
        decide_bellman(topic, count_bellman_teacher(model, 8, 2, 0.5, 0.01, 0.01)),
        decide_one_step_evsi(topic, model, 2, 0.5, 0.01),
        decide_evpi_bound(topic, model, 2, 0.5, 0.01),
        decide_consecutive_misses(topic, model, 2, 0.5, misses=2),
        decide_hit_probability_cutoff(topic, model, 2, 0.5, cutoff=0.01),
        decide_score_cutoff(topic, model, 2, 0.5, cutoff=0.2),
        decide_kneedle(topic, model, 2, 0.5, first_batch=2, rho_threshold=1.5),
        decide_qbcb(topic, model, 2, 0.5, sample_positives=1, target_recall=0.8, alpha=0.05, seed=7),
    ]
    for screened, action, belief in outcomes:
        assert 0 <= screened <= len(topic.labels)
        assert action in (ACTION_ACCEPT, ACTION_REJECT)
        assert 0.0 <= belief <= 1.0


def test_model_fit_and_teacher_are_reproducible():
    topics = [
        Topic("a", np.array([1, 0, 0, 0]), 0),
        Topic("b", np.array([1, 1, 0, 0]), 1),
        Topic("c", np.array([1, 1, 1, 0]), 1),
    ]
    left, right = fit_count_model(topics), fit_count_model(topics)
    assert left == right
    assert np.array_equal(
        count_bellman_teacher(left, 4, 2, 0.5, 0.01, 0.01).q_values,
        count_bellman_teacher(right, 4, 2, 0.5, 0.01, 0.01).q_values,
    )
