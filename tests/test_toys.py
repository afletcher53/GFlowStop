from math import comb

import numpy as np

from gflowstop.core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT, CountModel, count_bellman_teacher
from gflowstop.toys import (
    ExactOracle,
    PerfectInformationEnvironment,
    RankDecayEnvironment,
    RankLatentBelief,
    PolicyObservation,
    beta_binomial_sequence_distribution,
    delayed_sequence_distribution,
    hindsight_flow_action,
    independent_sequence_distribution,
    mixture_sequence_distribution,
    rank_belief_action,
    sample_episodes,
)
from gflowstop.toy_experiment import _failure_ucb, _qbcb_j


def test_hindsight_flow_target_is_not_bayes_optimal():
    sequences, probabilities = independent_sequence_distribution(np.array([0.5]))
    oracle = ExactOracle(sequences, probabilities, threshold=1, q_star=0.5, cost=0.01)
    assert oracle.action(()) == ACTION_CONTINUE
    assert oracle.values[()] > 0.0
    assert hindsight_flow_action(0, 0.5) != ACTION_CONTINUE
    assert hindsight_flow_action(1, 0.5) != ACTION_CONTINUE


def test_beta_binomial_enumeration_matches_count_bellman():
    model = CountModel(1.5, 3.0)
    sequences, probabilities = beta_binomial_sequence_distribution(model.alpha, model.beta, length=7)
    oracle = ExactOracle(sequences, probabilities, threshold=3, q_star=0.4, cost=0.03)
    teacher = count_bellman_teacher(model, length=7, threshold=3, q_star=0.4, cost=0.03, temperature=0.01)
    assert np.isclose(oracle.values[()], teacher.values[0, 0], atol=1e-12)


def test_episode_observation_excludes_evaluation_latent_data():
    episode = PerfectInformationEnvironment().sample(np.random.default_rng(7))
    observation = episode.observation(0, 0, 1)
    assert not hasattr(observation, "latent")
    assert not hasattr(observation, "labels")


def test_delayed_information_requires_non_myopic_planning():
    sequences, probabilities = delayed_sequence_distribution(10, early_length=3)
    oracle = ExactOracle(sequences, probabilities, threshold=2, q_star=0.5, cost=0.01)
    assert oracle.action_with_lookahead((), 1) != ACTION_CONTINUE
    assert oracle.action(()) == ACTION_CONTINUE


def test_qbcb_order_statistic_has_the_requested_binomial_failure_bound():
    target, alpha, positives = 0.9, 0.05, 30
    order = _qbcb_j(positives, target, alpha)
    failure = sum(comb(positives, count) * target**count * (1.0 - target) ** (positives - count) for count in range(order, positives + 1))
    assert order == 30
    assert failure <= alpha


def test_zero_failure_ucb_is_not_zero():
    assert np.isclose(_failure_ucb(0, 10_000), 1.0 - 0.05 ** (1.0 / 10_000))


def test_rank_planner_reuses_full_horizon_child_values():
    episodes = sample_episodes(RankDecayEnvironment(1.5, 1.0, 0.5, mu=1.0), 4, 4, seed=7)
    model = RankLatentBelief().fit(episodes[:3])
    observation = episodes[3].observation(0, 0, threshold=2)
    state = model.initial_state(observation)
    cache = {}
    _action, values = rank_belief_action(model, state, observation, q_star=0.5, cost=0.02, lookahead=4, cache=cache)
    belief = model.world_probability(state, observation)
    assert np.isclose(values[ACTION_ACCEPT], (belief - 0.5) / 2.0)
    assert np.isclose(values[ACTION_REJECT], (0.5 - belief) / 2.0)
    size = len(cache)
    child = episodes[3].observation(1, int(episodes[3].labels[0]), threshold=2)
    child_state = model.update(state, observation, int(episodes[3].labels[0]))
    rank_belief_action(model, child_state, child, q_star=0.5, cost=0.02, lookahead=4, cache=cache)
    assert len(cache) == size


def test_rank_belief_full_horizon_matches_exact_oracle():
    length, threshold, q_star, cost = 8, 3, 0.4, 0.03
    model = RankLatentBelief(rank_edges=tuple(np.linspace(0.0, 1.0, length + 1)), use_scores=False, eta_points=3)
    model.rank_logits = np.linspace(-0.8, -2.2, length)
    model.etas = np.array((-0.7, 0.0, 0.8))
    model.prior = np.array((0.2, 0.5, 0.3))
    model._probability_cache = model._tail_cache = None
    scores = np.zeros(length)
    root = PolicyObservation(length, 0, 0, threshold, scores, ())
    sequences, probabilities = mixture_sequence_distribution(model._probability_matrix(root), model.prior)
    oracle = ExactOracle(sequences, probabilities, threshold, q_star, cost)
    for history, exact_values in oracle.q_values.items():
        state = model.initial_state(root)
        observation = root
        for label in history:
            state = model.update(state, observation, label)
            observation = PolicyObservation(length, observation.depth + 1, observation.hits + label, threshold, scores, observation.history + (label,))
        _action, values = rank_belief_action(model, state, observation, q_star, cost, lookahead=length)
        assert np.allclose(values, exact_values, atol=1e-10)
