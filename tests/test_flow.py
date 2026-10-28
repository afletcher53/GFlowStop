from functools import lru_cache

import numpy as np
import pytest

from gflowstop.core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT, certificate_action
from gflowstop.flow import ChainData, ExactFlowNetwork, FlowStoppingNetwork
from gflowstop.tar_flow import (
    RecallChains,
    base_features,
    checkpoints,
    prefix_features,
    ranking_features,
    stop_utilities,
)
from gflowstop.toys import (
    ExactOracle,
    beta_binomial_sequence_distribution,
    delayed_sequence_distribution,
    independent_sequence_distribution,
)


@pytest.mark.parametrize(
    "law, threshold, q_star, cost",
    [
        (beta_binomial_sequence_distribution(1.5, 3.0, 8), 3, 0.4, 0.02),
        (delayed_sequence_distribution(10, 3), 2, 0.5, 0.01),
    ],
)
def test_geometric_flow_network_recovers_the_bayes_value(law, threshold, q_star, cost):
    sequences, probabilities = law
    oracle = ExactOracle(sequences, probabilities, threshold, q_star, cost)
    network = ExactFlowNetwork(sequences, probabilities, threshold, q_star, cost, beta=1e5, aggregator="geometric")
    assert np.isclose(network.expected_utility(greedy=True), oracle.values[()], atol=1e-9)
    assert np.isclose(network.q_values[()].max(), oracle.values[()], atol=1e-4)


def test_known_world_flow_network_samples_terminals_in_proportion_to_reward():
    """With no chance left (a point-mass law) every aggregator is an ordinary tree GFlowNet."""
    sequence = np.array([[1, 0, 1, 0]])
    beta, threshold, q_star, cost = 7.0, 2, 0.4, 0.03
    rewards = {}
    hits = 0
    for depth in range(5):
        forced = certificate_action(hits, depth, 4, threshold)
        for action in (ACTION_ACCEPT, ACTION_REJECT):
            if forced is None or forced == action:
                payoff = (1 - q_star) / 2 if action == ACTION_ACCEPT else -(1 - q_star) / 2
                rewards[(tuple(sequence[0, :depth]), action)] = np.exp(beta * (payoff - cost * depth))
        if forced is not None:
            break
        hits += sequence[0, depth]
    expected = {key: value / sum(rewards.values()) for key, value in rewards.items()}
    for aggregator in ("geometric", "arithmetic", "hindsight"):
        network = ExactFlowNetwork(sequence, np.array([1.0]), threshold, q_star, cost, beta, aggregator)
        distribution = network.terminal_distribution()
        assert distribution.keys() == expected.keys()
        assert all(np.isclose(distribution[key], expected[key]) for key in expected)


def test_arithmetic_expected_flow_is_risk_seeking_at_large_beta():
    sequences, probabilities = independent_sequence_distribution(np.array([0.5]))
    oracle = ExactOracle(sequences, probabilities, threshold=1, q_star=0.5, cost=0.01)
    network = ExactFlowNetwork(sequences, probabilities, 1, 0.5, 0.01, beta=1e3, aggregator="arithmetic")
    assert oracle.action(()) == ACTION_CONTINUE
    assert network.greedy_action(()) != ACTION_CONTINUE
    assert network.expected_utility() < oracle.values[()] - 0.2


def test_hindsight_continue_flow_is_never_below_the_geometric_one():
    sequences, probabilities = beta_binomial_sequence_distribution(1.5, 3.0, 7)
    geometric = ExactFlowNetwork(sequences, probabilities, 3, 0.4, 0.02, beta=30.0, aggregator="geometric")
    hindsight = ExactFlowNetwork(sequences, probabilities, 3, 0.4, 0.02, beta=30.0, aggregator="hindsight")
    continuing = [history for history, values in geometric.q_values.items() if np.isfinite(values[ACTION_CONTINUE])]
    assert continuing
    assert all(hindsight.q_values[h][ACTION_CONTINUE] >= geometric.q_values[h][ACTION_CONTINUE] - 1e-12 for h in continuing)
    assert any(hindsight.q_values[h][ACTION_CONTINUE] > geometric.q_values[h][ACTION_CONTINUE] + 1e-6 for h in continuing)


def test_policy_returns_average_only_the_policys_choices():
    chain = ChainData(np.zeros((3, 1)), np.array([[-0.1], [-0.2], [-0.3]]), np.array([1, 2, -1]))
    probabilities = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(chain.heights(), [2, 1, 0])
    assert np.allclose(chain.policy_returns(probabilities, np.inf), [-0.2, -0.3, -0.3])


def _toy_chains(sequences, threshold, q_star, cost):
    features, utilities, next_row = [], [], []
    for sequence in sequences:
        hits, length = 0, len(sequence)
        for depth in range(length + 1):
            forced = certificate_action(hits, depth, length, threshold)
            world = sequence.sum() >= threshold
            row = np.full(2, np.nan)
            for column, action in enumerate((ACTION_ACCEPT, ACTION_REJECT)):
                if forced is None or forced == action:
                    accepted = action == ACTION_ACCEPT
                    row[column] = ((1 - q_star) / 2 if world else -q_star / 2) if accepted else (-(1 - q_star) / 2 if world else q_star / 2)
                    row[column] -= cost * depth
            features.append([depth / length, hits / length])
            utilities.append(row)
            done = forced is not None or depth == length
            next_row.append(-1 if done else len(features))
            if done:
                break
            hits += sequence[depth]
    return ChainData(np.array(features), np.array(utilities), np.array(next_row))


def test_learned_geometric_network_reaches_the_bayes_value_on_an_exchangeable_toy():
    sequences, probabilities = beta_binomial_sequence_distribution(1.5, 3.0, 6)
    threshold, q_star, cost = 2, 0.4, 0.02
    oracle = ExactOracle(sequences, probabilities, threshold, q_star, cost)
    rng = np.random.default_rng(0)
    data = _toy_chains(sequences[rng.choice(len(sequences), size=6000, p=probabilities)], threshold, q_star, cost)
    network = FlowStoppingNetwork(beta=1e4, rounds=60, steps=20, batch_size=2048, hidden=(32, 32)).fit(data)

    @lru_cache(maxsize=None)
    def value(history):
        depth, hits = len(history), sum(history)
        mask = np.all(sequences[:, :depth] == np.array(history, dtype=int), axis=1) if history else np.ones(len(sequences), bool)
        weights = probabilities[mask] / probabilities[mask].sum()
        world = sequences[mask].sum(axis=1) >= threshold
        accept = float(weights @ np.where(world, (1 - q_star) / 2, -q_star / 2)) - cost * depth
        reject = float(weights @ np.where(world, -(1 - q_star) / 2, q_star / 2)) - cost * depth
        forced = certificate_action(hits, depth, 6, threshold)
        if forced is not None:
            return accept if forced == ACTION_ACCEPT else reject
        q = network.action_values(np.array([[depth / 6, hits / 6]]))[0]
        q[0] = q[0] if depth < 6 else -np.inf
        action = int(np.argmax(q))
        if action:
            return accept if action == 1 else reject
        p_hit = float(weights @ sequences[mask][:, depth])
        return p_hit * value(history + (1,)) + (1 - p_hit) * value(history + (0,))

    assert value(()) >= oracle.values[()] - 2e-3


def test_recall_features_read_only_the_reviewed_prefix():
    rng = np.random.default_rng(3)
    labels = (rng.random(700) < np.linspace(0.4, 0.01, 700)).astype(int)
    depths, features = ranking_features(labels, 0.9, 4.0)
    for index in (0, 7, 60, len(depths) - 2):
        depth = depths[index]
        assert np.allclose(features[index], prefix_features(labels[:depth], len(labels), 0.9, 4.0), atol=1e-6)
        flipped = labels.copy()
        flipped[depth:] = 1 - flipped[depth:]
        assert np.allclose(features[index], ranking_features(flipped, 0.9, 4.0)[1][index], atol=1e-6)
    with pytest.raises(ValueError):
        base_features(labels[:10], len(labels), np.array([11]))


def test_stop_utility_charges_review_and_a_scaled_miss_penalty():
    labels = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    depths = checkpoints(len(labels))
    utilities = stop_utilities(labels, depths, target=0.9, penalty=3.0)
    assert np.allclose(utilities[:3], -(depths[:3] / 10 + 3.0) / 4.0)
    assert np.allclose(utilities[3:], -(depths[3:] / 10) / 4.0)


def test_recall_chains_link_each_ranking_copy_in_review_order():
    rankings = [np.array([1, 0, 1, 0, 0]), np.array([0, 0, 0, 1])]
    data = RecallChains.build(rankings, copies=2, count=10)
    ends = np.flatnonzero(data.next_row < 0)
    assert len(ends) == 4
    assert np.all(data.next_row[data.next_row >= 0] == np.flatnonzero(data.next_row >= 0) + 1)
    assert np.allclose(data.review[ends], 1.0) and np.allclose(data.miss[ends], 0.0)


def test_iph_features_read_only_the_reviewed_prefix():
    pytest.importorskip("pyterrier_tar")
    rng = np.random.default_rng(5)
    labels = (rng.random(900) < np.linspace(0.3, 0.005, 900)).astype(int)
    depths, features = ranking_features(labels, 0.9, 4.0, power=True, iph=True)
    assert features.shape[1] == 50
    for index in (3, 40, 90, len(depths) - 2):
        depth = depths[index]
        flipped = labels.copy()
        flipped[depth:] = 1 - flipped[depth:]
        assert np.allclose(features[index], ranking_features(flipped, 0.9, 4.0, power=True, iph=True)[1][index], atol=1e-6)
