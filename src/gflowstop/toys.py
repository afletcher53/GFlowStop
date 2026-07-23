"""Fixed synthetic stopping tasks and small exact Bayes oracles.

The module is deliberately self-contained: policy code receives a
``PolicyObservation`` only, while ``Episode.latent`` is evaluation metadata.
That separation makes accidental future-label or latent-variable leakage fail
at the API boundary rather than relying on reviewer vigilance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import lgamma, log
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT, certificate_action


def _expit(value: np.ndarray | float) -> np.ndarray | float:
    value = np.asarray(value)
    positive = value >= 0.0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result if result.ndim else float(result)


def _logit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class Episode:
    """One fixed ranking; ``latent`` is explicitly evaluation-only."""

    labels: np.ndarray
    scores: np.ndarray
    latent: Mapping[str, float | int | str]
    environment: str
    seed: int

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or self.scores.ndim != 1 or len(self.labels) != len(self.scores):
            raise ValueError("labels and scores must be same-length one-dimensional arrays")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be binary")

    def observation(self, depth: int, hits: int, threshold: int) -> "PolicyObservation":
        if not 0 <= depth <= len(self.labels) or not 0 <= hits <= depth:
            raise ValueError("invalid observable prefix")
        return PolicyObservation(
            length=len(self.labels), depth=depth, hits=hits, threshold=threshold,
            scores=self.scores.copy(), history=tuple(int(value) for value in self.labels[:depth]),
        )


@dataclass(frozen=True)
class PolicyObservation:
    """The complete policy input. It contains neither latent variables nor future labels."""

    length: int
    depth: int
    hits: int
    threshold: int
    scores: np.ndarray
    history: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.history) != self.depth or self.hits != sum(self.history):
            raise ValueError("history and observable count disagree")
        if hasattr(self, "latent") or hasattr(self, "labels"):
            raise AssertionError("stopping rules must not receive latent data or unrevealed labels")


class Environment(Protocol):
    name: str

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        """Return one fully-labelled episode for offline evaluation."""


class BeliefModel(Protocol):
    def fit(self, episodes: list[Episode]) -> "BeliefModel": ...

    def initial_state(self, observation: PolicyObservation) -> object: ...

    def update(self, state: object, observation: PolicyObservation, observed_label: int) -> object: ...

    def next_hit_probability(self, state: object, observation: PolicyObservation) -> float: ...

    def world_probability(self, state: object, observation: PolicyObservation) -> float: ...


class StoppingRule(Protocol):
    def act(self, observation: PolicyObservation) -> int: ...


@dataclass(frozen=True)
class PerfectInformationEnvironment:
    name: str = "perfect_information"

    def sample(self, rng: np.random.Generator, length: int = 1) -> Episode:
        if length != 1:
            raise ValueError("the perfect-information counterexample has horizon one")
        world = int(rng.integers(0, 2))
        return Episode(np.array([world], dtype=int), np.zeros(1), {"world": world}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class BetaBinomialEnvironment:
    alpha: float
    beta: float
    name: str = "beta_binomial"

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        theta = float(rng.beta(self.alpha, self.beta))
        labels = rng.binomial(1, theta, size=length).astype(int)
        scores = rng.normal(size=length)
        return Episode(labels, scores, {"theta": theta, "alpha": self.alpha, "beta": self.beta}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class RankDecayEnvironment:
    gamma: float
    sigma_eta: float
    sigma_score: float
    mu: float = -2.0
    name: str = "rank_decay"

    def probabilities(self, eta: float, length: int) -> np.ndarray:
        ranks = np.arange(1, length + 1, dtype=float)
        return np.asarray(_expit(eta - self.gamma * np.log1p(ranks)), dtype=float)

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        eta = float(rng.normal(self.mu, self.sigma_eta))
        probabilities = self.probabilities(eta, length)
        labels = rng.binomial(1, probabilities).astype(int)
        scores = _logit(probabilities) + rng.normal(0.0, self.sigma_score, size=length)
        return Episode(labels, np.asarray(scores), {"eta": eta, "gamma": self.gamma}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class DelayedInformationEnvironment:
    early_length: int = 3
    name: str = "delayed_information"

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        if length <= self.early_length:
            raise ValueError("delayed-information length must exceed early_length")
        latent = int(rng.integers(0, 2))
        probabilities = np.full(length, 0.05)
        probabilities[self.early_length :] = 0.6 if latent else 0.03
        labels = rng.binomial(1, probabilities).astype(int)
        scores = _logit(probabilities) + rng.normal(0.0, 0.2, size=length)
        return Episode(labels, np.asarray(scores), {"z": latent}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class ClusteredEnvironment:
    persistence: float = 0.9
    name: str = "clustered"

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        states = np.empty(length, dtype=int)
        states[0] = int(rng.integers(0, 2))
        for index in range(1, length):
            states[index] = states[index - 1] if rng.random() < self.persistence else 1 - states[index - 1]
        probabilities = np.where(states == 1, 0.6, 0.01)
        labels = rng.binomial(1, probabilities).astype(int)
        scores = _logit(probabilities) + rng.normal(0.0, 0.75, size=length)
        return Episode(labels, np.asarray(scores), {"states": "held_out"}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class MixtureShiftEnvironment:
    weight_high: float = 0.3
    low_mu: float = -4.0
    high_mu: float = -0.5
    sigma: float = 0.4
    gamma: float = 0.8
    score_slope: float = 1.0
    score_noise: float = 0.5
    name: str = "mixture_shift"

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        component = int(rng.random() < self.weight_high)
        eta = float(rng.normal(self.high_mu if component else self.low_mu, self.sigma))
        ranks = np.arange(1, length + 1, dtype=float)
        logits = eta - self.gamma * np.log1p(ranks)
        probabilities = np.asarray(_expit(logits), dtype=float)
        labels = rng.binomial(1, probabilities).astype(int)
        scores = self.score_slope * logits + rng.normal(0.0, self.score_noise, size=length)
        return Episode(labels, np.asarray(scores), {"component": component, "eta": eta}, self.name, int(rng.integers(2**63)))


@dataclass(frozen=True)
class SparseRecallEnvironment:
    relevant_count: int
    gamma: float
    score_noise: float = 0.5
    name: str = "sparse_recall"

    def sample(self, rng: np.random.Generator, length: int) -> Episode:
        if not 0 < self.relevant_count <= length:
            raise ValueError("relevant_count must be between one and length")
        ranks = np.arange(length, dtype=float)
        weights = np.exp(-self.gamma * ranks / max(length, 1))
        positions = rng.choice(length, size=self.relevant_count, replace=False, p=weights / weights.sum())
        labels = np.zeros(length, dtype=int)
        labels[positions] = 1
        scores = np.log(weights) + rng.normal(0.0, self.score_noise, size=length)
        return Episode(labels, scores, {"relevant_count": self.relevant_count, "gamma": self.gamma}, self.name, int(rng.integers(2**63)))


def sample_episodes(environment: Environment, length: int, count: int, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [environment.sample(rng, length) for _ in range(count)]


def save_episodes(path: Path, episodes: list[Episode], parameters: Mapping[str, object]) -> None:
    """Persist fixed labels, scores, seeds, and generator parameters once for all methods."""
    if not episodes:
        raise ValueError("cannot save no episodes")
    lengths = {len(episode.labels) for episode in episodes}
    if len(lengths) != 1:
        raise ValueError("one dataset file must have a common horizon")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.stack([episode.labels for episode in episodes]),
        scores=np.stack([episode.scores for episode in episodes]),
        seeds=np.asarray([episode.seed for episode in episodes], dtype=np.int64),
        environment=np.asarray(episodes[0].environment),
        parameters=np.asarray(repr(dict(parameters))),
    )


def label_sequences(length: int) -> np.ndarray:
    if not 1 <= length <= 20:
        raise ValueError("exact enumeration is limited to horizons 1..20")
    numbers = np.arange(1 << length, dtype=np.uint32)
    shifts = np.arange(length - 1, -1, -1, dtype=np.uint32)
    return ((numbers[:, None] >> shifts) & 1).astype(int)


def independent_sequence_distribution(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    sequences = label_sequences(len(probabilities))
    log_probability = sequences * np.log(np.clip(probabilities, 1e-15, 1.0)) + (1 - sequences) * np.log(np.clip(1.0 - probabilities, 1e-15, 1.0))
    weights = np.exp(log_probability.sum(axis=1))
    return sequences, weights / weights.sum()


def beta_binomial_sequence_distribution(alpha: float, beta: float, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint label-sequence law for an exchangeable beta-binomial topic."""
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta-binomial parameters must be positive")
    sequences = label_sequences(length)
    hits = sequences.sum(axis=1)
    log_weights = np.asarray([
        lgamma(alpha + hit) + lgamma(beta + length - hit) - lgamma(alpha + beta + length) - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta))
        for hit in hits
    ])
    weights = np.exp(log_weights - log_weights.max())
    return sequences, weights / weights.sum()


def mixture_sequence_distribution(probability_rows: np.ndarray, mixture_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    probability_rows = np.asarray(probability_rows, dtype=float)
    if probability_rows.ndim != 2:
        raise ValueError("probability_rows must be (latent_state, rank)")
    weights = np.ones(len(probability_rows)) if mixture_weights is None else np.asarray(mixture_weights, dtype=float)
    weights = weights / weights.sum()
    sequences = label_sequences(probability_rows.shape[1])
    likelihood = np.prod(np.where(sequences[None, :, :] == 1, probability_rows[:, None, :], 1.0 - probability_rows[:, None, :]), axis=2)
    probabilities = weights @ likelihood
    return sequences, probabilities / probabilities.sum()


def delayed_sequence_distribution(length: int, early_length: int = 3) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(length, 0.05)
    high = low.copy()
    high[early_length:] = 0.6
    low[early_length:] = 0.03
    return mixture_sequence_distribution(np.stack((low, high)), np.array((0.5, 0.5)))


def clustered_sequence_distribution(length: int, persistence: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    sequences = label_sequences(length)
    weights = np.empty(len(sequences), dtype=float)
    transition = np.array(((persistence, 1.0 - persistence), (1.0 - persistence, persistence)))
    emissions = np.array((0.01, 0.6))
    for index, labels in enumerate(sequences):
        forward = np.array((0.5, 0.5)) * np.where(labels[0], emissions, 1.0 - emissions)
        for label in labels[1:]:
            forward = (forward @ transition) * np.where(label, emissions, 1.0 - emissions)
        weights[index] = forward.sum()
    return sequences, weights / weights.sum()


@dataclass(eq=False)
class ExactOracle:
    """Exact Bayes-optimal utility planner from a finite distribution over label rankings."""

    sequences: np.ndarray
    probabilities: np.ndarray
    threshold: int
    q_star: float
    cost: float

    def __post_init__(self) -> None:
        self.sequences = np.asarray(self.sequences, dtype=int)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if self.sequences.ndim != 2 or len(self.sequences) != len(self.probabilities):
            raise ValueError("sequences and probabilities disagree")
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("sequence probabilities must sum to one")
        self.length = int(self.sequences.shape[1])
        self._world = self.sequences.sum(axis=1) >= self.threshold
        self.q_values: dict[tuple[int, ...], np.ndarray] = {}
        self.values: dict[tuple[int, ...], float] = {}
        self._solve(())

    def _conditional(self, history: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if not history:
            mask = np.ones(len(self.sequences), dtype=bool)
        else:
            mask = np.all(self.sequences[:, : len(history)] == np.asarray(history), axis=1)
        probabilities = self.probabilities[mask]
        if probabilities.sum() <= 0.0:
            raise ValueError("history has zero probability under the oracle")
        return self.sequences[mask], probabilities / probabilities.sum()

    def world_probability(self, history: tuple[int, ...]) -> float:
        sequences, probabilities = self._conditional(history)
        return float(probabilities[sequences.sum(axis=1) >= self.threshold].sum())

    @lru_cache(maxsize=None)
    def _solve(self, history: tuple[int, ...]) -> float:
        depth, hits = len(history), sum(history)
        forced = certificate_action(hits, depth, self.length, self.threshold)
        values = np.full(3, -np.inf)
        if forced is not None:
            values[forced] = (1.0 - self.q_star) / 2.0 if forced == ACTION_ACCEPT else self.q_star / 2.0
        else:
            belief = self.world_probability(history)
            values[ACTION_ACCEPT] = (belief - self.q_star) / 2.0
            values[ACTION_REJECT] = (self.q_star - belief) / 2.0
            if depth < self.length:
                sequences, probabilities = self._conditional(history)
                p_hit = float((probabilities * sequences[:, depth]).sum())
                continuation = -self.cost
                if p_hit:
                    continuation += p_hit * self._solve(history + (1,))
                if p_hit < 1.0:
                    continuation += (1.0 - p_hit) * self._solve(history + (0,))
                values[ACTION_CONTINUE] = continuation
        self.q_values[history] = values
        self.values[history] = float(values.max())
        return self.values[history]

    def action(self, history: tuple[int, ...]) -> int:
        return int(np.argmax(self.q_values[history]))

    def action_gap(self, history: tuple[int, ...]) -> float:
        values = np.sort(self.q_values[history])
        return float(values[-1] - values[-2])

    def action_with_lookahead(self, history: tuple[int, ...], lookahead: int) -> int:
        """Receding-horizon action using the true terminal belief at the truncated leaf."""
        if lookahead < 1:
            raise ValueError("lookahead must be positive")
        limit = min(self.length, len(history) + lookahead)

        @lru_cache(maxsize=None)
        def value(prefix: tuple[int, ...]) -> tuple[float, np.ndarray]:
            depth, hits = len(prefix), sum(prefix)
            forced = certificate_action(hits, depth, self.length, self.threshold)
            values = np.full(3, -np.inf)
            if forced is not None:
                values[forced] = (1.0 - self.q_star) / 2.0 if forced == ACTION_ACCEPT else self.q_star / 2.0
            else:
                belief = self.world_probability(prefix)
                values[ACTION_ACCEPT] = (belief - self.q_star) / 2.0
                values[ACTION_REJECT] = (self.q_star - belief) / 2.0
                if depth < limit:
                    sequences, probabilities = self._conditional(prefix)
                    p_hit = float((probabilities * sequences[:, depth]).sum())
                    continuation = -self.cost
                    if p_hit:
                        continuation += p_hit * value(prefix + (1,))[0]
                    if p_hit < 1.0:
                        continuation += (1.0 - p_hit) * value(prefix + (0,))[0]
                    values[ACTION_CONTINUE] = continuation
            return float(values.max()), values

        return int(np.argmax(value(history)[1]))

    def rollout(self, episode: Episode) -> tuple[int, int, float]:
        history: tuple[int, ...] = ()
        for depth in range(self.length + 1):
            observation = episode.observation(depth, sum(history), self.threshold)
            assert not hasattr(observation, "latent") and not hasattr(observation, "labels")
            action = self.action(history)
            if action != ACTION_CONTINUE:
                return depth, action, self.world_probability(history)
            history += (int(episode.labels[depth]),)
        raise AssertionError("exact oracle did not terminate")


def hindsight_flow_action(realised_world: int, q_star: float) -> int:
    """The invalid per-topic target: terminal reward conditioned on the answer."""
    return ACTION_ACCEPT if realised_world else ACTION_REJECT


def utility(labels: np.ndarray, threshold: int, depth: int, action: int, q_star: float, cost: float) -> float:
    world = int(np.asarray(labels).sum() >= threshold)
    terminal = (1.0 - q_star) / 2.0 if action == ACTION_ACCEPT and world else 0.0
    if action == ACTION_ACCEPT and not world:
        terminal = -q_star / 2.0
    elif action == ACTION_REJECT and world:
        terminal = -(1.0 - q_star) / 2.0
    elif action == ACTION_REJECT:
        terminal = q_star / 2.0
    return terminal - cost * depth


def recall_at(labels: np.ndarray, reviewed: np.ndarray | list[int]) -> float:
    labels = np.asarray(labels, dtype=int)
    relevant = int(labels.sum())
    return 1.0 if relevant == 0 else float(labels[np.asarray(reviewed, dtype=int)].sum() / relevant)


def recall_oracle_depth(labels: np.ndarray, target: float) -> int:
    labels = np.asarray(labels, dtype=int)
    required = int(np.ceil(target * labels.sum()))
    if required == 0:
        return 0
    return int(np.flatnonzero(np.cumsum(labels) >= required)[0] + 1)


def calibration_metrics(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    outcomes = np.asarray(outcomes, dtype=float)
    if len(probabilities) != len(outcomes) or not len(probabilities):
        raise ValueError("calibration inputs must have matching non-zero lengths")
    brier = float(np.mean((probabilities - outcomes) ** 2))
    log_loss = float(-np.mean(outcomes * np.log(probabilities) + (1.0 - outcomes) * np.log(1.0 - probabilities)))
    ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    for identifier in range(bins):
        mask = ids == identifier
        if mask.any():
            ece += mask.mean() * abs(float(probabilities[mask].mean() - outcomes[mask].mean()))
    return {"brier": brier, "log_loss": log_loss, "ece": float(ece)}


def _decreasing_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators projection onto a decreasing sequence."""
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            right, left = blocks.pop(), blocks.pop()
            total_weight = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / total_weight, total_weight, left[2] + right[2]])
    expanded = []
    for value, _weight, count in blocks:
        expanded.extend([value] * int(count))
    return np.asarray(expanded, dtype=float)


@dataclass(frozen=True)
class RankBeliefState:
    weights: np.ndarray


@dataclass
class RankLatentBelief:
    """Monotone rank plus optional score model with a discrete topic-intensity posterior."""

    rank_edges: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0)
    use_scores: bool = True
    use_score_mean: bool = True
    use_local_scores: bool = True
    eta_points: int = 21

    def _bin(self, fraction: np.ndarray | float) -> np.ndarray:
        return np.minimum(np.searchsorted(self.rank_edges[1:], fraction, side="right"), len(self.rank_edges) - 2)

    @staticmethod
    def _score_features(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=float)
        scale = max(float(scores.std()), 1e-6)
        normalised = (scores - float(scores.mean())) / scale
        slope = np.r_[normalised[:-1] - normalised[1:], 0.0]
        return normalised, slope

    def fit(self, episodes: list[Episode]) -> "RankLatentBelief":
        if not episodes:
            raise ValueError("at least one training episode is required")
        self._probability_cache = None
        self._tail_cache = None
        bins = len(self.rank_edges) - 1
        hit_counts, totals = np.zeros(bins), np.zeros(bins)
        design, labels, offsets = [], [], []
        topic_residuals, score_means = [], []
        for episode in episodes:
            fractions = (np.arange(len(episode.labels)) + 0.5) / len(episode.labels)
            identifiers = self._bin(fractions)
            hit_counts += np.bincount(identifiers, weights=episode.labels, minlength=bins)
            totals += np.bincount(identifiers, minlength=bins)
        rates = _decreasing_isotonic((hit_counts + 0.5) / (totals + 1.0), totals + 1.0)
        self.rank_logits = np.asarray(_logit(rates), dtype=float)
        for episode in episodes:
            fractions = (np.arange(len(episode.labels)) + 0.5) / len(episode.labels)
            base = self.rank_logits[self._bin(fractions)]
            score, slope = self._score_features(episode.scores)
            design.append(np.c_[np.ones(len(score)), score, slope, score * fractions])
            labels.append(episode.labels)
            offsets.append(base)
            topic_residuals.append(float(_logit((episode.labels.sum() + 0.5) / (len(episode.labels) + 1.0)) - np.mean(base)))
            score_means.append(float(episode.scores.mean()))
        x = np.vstack(design)
        y = np.concatenate(labels).astype(float)
        offset = np.concatenate(offsets)
        self.score_weights = np.zeros(x.shape[1])
        if self.use_scores and self.use_local_scores:
            for _ in range(600):
                probabilities = np.asarray(_expit(offset + x @ self.score_weights), dtype=float)
                gradient = x.T @ (probabilities - y) / len(y)
                self.score_weights -= 0.15 * gradient
        self.eta_scale = float(np.clip(np.std(topic_residuals, ddof=1) if len(topic_residuals) > 1 else 0.5, 0.15, 2.0))
        self.score_mean_center = float(np.mean(score_means))
        self.score_mean_scale = max(float(np.std(score_means)), 1e-6)
        scaled_means = (np.asarray(score_means) - self.score_mean_center) / self.score_mean_scale
        self.score_eta_slope = float(np.clip(np.dot(scaled_means, topic_residuals) / max(np.dot(scaled_means, scaled_means), 1e-9), -2.5, 2.5))
        self.etas = np.linspace(-2.5 * self.eta_scale, 2.5 * self.eta_scale, self.eta_points)
        prior = np.exp(-0.5 * (self.etas / self.eta_scale) ** 2)
        self.prior = prior / prior.sum()
        return self

    def _observation_key(self, observation: PolicyObservation) -> tuple[int, bytes]:
        return observation.length, observation.scores.tobytes() if self.use_scores and self.use_local_scores else b""

    def _probability_matrix(self, observation: PolicyObservation) -> np.ndarray:
        if not hasattr(self, "rank_logits"):
            raise RuntimeError("fit the belief model before using it")
        key = self._observation_key(observation)
        cached = self._probability_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        fractions = (np.arange(observation.length) + 0.5) / observation.length
        score, slope = self._score_features(observation.scores)
        x = np.c_[np.ones(observation.length), score, slope, score * fractions]
        fixed = self.rank_logits[self._bin(fractions)]
        if self.use_scores and self.use_local_scores:
            fixed = fixed + x @ self.score_weights
        matrix = np.asarray(_expit(fixed[None, :] + self.etas[:, None]), dtype=float)
        self._probability_cache = key, matrix
        return matrix

    def _tail_distributions(self, observation: PolicyObservation) -> tuple[np.ndarray, ...]:
        key = self._observation_key(observation)
        cached = self._tail_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        matrix = self._probability_matrix(observation)
        tails = []
        for depth in range(observation.length + 1):
            tail = np.empty((len(matrix), observation.length - depth + 1))
            for index, probabilities in enumerate(matrix):
                distribution = np.array([1.0])
                for probability in probabilities[depth:]:
                    distribution = np.convolve(distribution, np.array((1.0 - probability, probability)))
                tail[index] = distribution
            tails.append(tail)
        result = tuple(tails)
        self._tail_cache = key, result
        return result

    def initial_state(self, observation: PolicyObservation) -> RankBeliefState:
        assert not hasattr(observation, "latent") and not hasattr(observation, "labels")
        self._probability_matrix(observation)
        if not self.use_scores or not self.use_score_mean:
            return RankBeliefState(self.prior.copy())
        predicted_eta = self.score_eta_slope * ((float(observation.scores.mean()) - self.score_mean_center) / self.score_mean_scale)
        weights = self.prior * np.exp(-0.5 * ((self.etas - predicted_eta) / self.eta_scale) ** 2)
        return RankBeliefState(weights / weights.sum())

    def update(self, state: RankBeliefState, observation: PolicyObservation, observed_label: int) -> RankBeliefState:
        if observed_label not in (0, 1) or observation.depth >= observation.length:
            raise ValueError("invalid observed label or depth")
        probabilities = self._probability_matrix(observation)[:, observation.depth]
        weights = state.weights * (probabilities if observed_label else 1.0 - probabilities)
        weights /= weights.sum()
        return RankBeliefState(weights)

    def next_hit_probability(self, state: RankBeliefState, observation: PolicyObservation) -> float:
        if observation.depth >= observation.length:
            return 0.0
        probabilities = self._probability_matrix(observation)[:, observation.depth]
        return float(state.weights @ probabilities)

    def remaining_distribution(self, state: RankBeliefState, observation: PolicyObservation) -> np.ndarray:
        return state.weights @ self._tail_distributions(observation)[observation.depth]

    def world_probability(self, state: RankBeliefState, observation: PolicyObservation) -> float:
        needed = observation.threshold - observation.hits
        if needed <= 0:
            return 1.0
        distribution = self.remaining_distribution(state, observation)
        return 0.0 if needed >= len(distribution) else float(distribution[needed:].sum())

    def expected_remaining(self, state: RankBeliefState, observation: PolicyObservation) -> float:
        distribution = self.remaining_distribution(state, observation)
        return float(np.arange(len(distribution)) @ distribution)

    def remaining_variance(self, state: RankBeliefState, observation: PolicyObservation) -> float:
        distribution = self.remaining_distribution(state, observation)
        mean = float(np.arange(len(distribution)) @ distribution)
        return float(((np.arange(len(distribution)) - mean) ** 2) @ distribution)


@dataclass(frozen=True)
class K1RankPlan:
    """Exact policy values on the only non-terminal K=1 path: consecutive misses."""

    actions: np.ndarray
    values: np.ndarray
    beliefs: np.ndarray


def k1_rank_bellman(
    model: RankLatentBelief, observation: PolicyObservation, q_star: float, cost: float,
) -> K1RankPlan:
    """Solve a full K=1 rank-aware Bellman problem in linear time.

    Before the first relevant document every reachable history is a run of
    misses; after a hit, acceptance is a logical certificate.  There is no
    need to enumerate a binary history tree.
    """
    if observation.threshold != 1 or observation.depth != 0 or observation.hits != 0:
        raise ValueError("K=1 planning starts before screening with no observed hits")
    if not 0.0 < q_star < 1.0 or cost < 0.0:
        raise ValueError("invalid utility parameters")
    return _k1_plan(model._probability_matrix(observation), model.initial_state(observation).weights, q_star, cost)


def zero_inflated_k1_rank_bellman(
    model: RankLatentBelief, observation: PolicyObservation, zero_mass: float, q_star: float, cost: float,
) -> K1RankPlan:
    """Exact K=1 planner with a structural all-zero-topic mixture component."""
    if not 0.0 <= zero_mass < 1.0:
        raise ValueError("zero_mass must be in [0, 1)")
    if observation.threshold != 1 or observation.depth != 0 or observation.hits != 0:
        raise ValueError("K=1 planning starts before screening with no observed hits")
    positive_probabilities = model._probability_matrix(observation)
    probabilities = np.vstack((np.zeros((1, observation.length)), positive_probabilities))
    weights = np.r_[zero_mass, (1.0 - zero_mass) * model.initial_state(observation).weights]
    return _k1_plan(probabilities, weights, q_star, cost)


def _k1_plan(probabilities: np.ndarray, initial_weights: np.ndarray, q_star: float, cost: float) -> K1RankPlan:
    probabilities = np.asarray(probabilities, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    if probabilities.ndim != 2 or len(initial_weights) != len(probabilities):
        raise ValueError("K=1 probabilities and weights disagree")
    if not np.isclose(initial_weights.sum(), 1.0):
        raise ValueError("K=1 initial weights must sum to one")
    length, states = probabilities.shape[1], len(probabilities)
    if not 0.0 < q_star < 1.0 or cost < 0.0:
        raise ValueError("invalid utility parameters")
    weights = np.empty((length + 1, states))
    weights[0] = initial_weights
    for depth in range(length):
        weights[depth + 1] = weights[depth] * (1.0 - probabilities[:, depth])
        weights[depth + 1] /= weights[depth + 1].sum()
    no_hit = np.ones((states, length + 1))
    no_hit[:, :-1] = np.cumprod((1.0 - probabilities)[:, ::-1], axis=1)[:, ::-1]
    beliefs = 1.0 - np.einsum("ij,ji->i", weights, no_hit)
    values = np.full((length + 1, 3), -np.inf)
    values[length, ACTION_REJECT] = q_star / 2.0
    for depth in range(length - 1, -1, -1):
        belief = beliefs[depth]
        values[depth, ACTION_ACCEPT] = (belief - q_star) / 2.0
        values[depth, ACTION_REJECT] = (q_star - belief) / 2.0
        hit_probability = float(weights[depth] @ probabilities[:, depth])
        values[depth, ACTION_CONTINUE] = (
            -cost
            + hit_probability * (1.0 - q_star) / 2.0
            + (1.0 - hit_probability) * values[depth + 1].max()
        )
    return K1RankPlan(np.argmax(values, axis=1), values, beliefs)


def _synthetic_observation(observation: PolicyObservation, observed_label: int) -> PolicyObservation:
    if observed_label not in (0, 1):
        raise ValueError("observed_label must be binary")
    return PolicyObservation(
        length=observation.length, depth=observation.depth + 1, hits=observation.hits + observed_label,
        threshold=observation.threshold, scores=observation.scores, history=observation.history + (observed_label,),
    )


def rank_belief_action(
    model: RankLatentBelief, state: RankBeliefState, observation: PolicyObservation, q_star: float, cost: float, lookahead: int,
    cache: dict[tuple[int, tuple[int, ...], bytes], tuple[float, np.ndarray]] | None = None,
) -> tuple[int, np.ndarray]:
    """Receding-horizon Bellman action over the belief state, with exact count tails at leaves."""
    if lookahead < 1:
        raise ValueError("lookahead must be positive")
    limit = min(observation.length, observation.depth + lookahead)
    root_key = (limit, observation.history, state.weights.tobytes())
    if cache is not None and root_key in cache:
        values = cache[root_key][1]
        return int(np.argmax(values)), values
    probabilities = model._probability_matrix(observation)
    tails = model._tail_distributions(observation)

    def world_probability(current: RankBeliefState, current_observation: PolicyObservation) -> float:
        needed = current_observation.threshold - current_observation.hits
        if needed <= 0:
            return 1.0
        distribution = current.weights @ tails[current_observation.depth]
        return 0.0 if needed >= len(distribution) else float(distribution[needed:].sum())

    def update(current: RankBeliefState, current_observation: PolicyObservation, observed_label: int) -> RankBeliefState:
        probability = probabilities[:, current_observation.depth]
        weights = current.weights * (probability if observed_label else 1.0 - probability)
        return RankBeliefState(weights / weights.sum())

    def terminal_values(current: RankBeliefState, current_observation: PolicyObservation) -> np.ndarray:
        values = np.full(3, -np.inf)
        forced = certificate_action(current_observation.hits, current_observation.depth, current_observation.length, current_observation.threshold)
        if forced is not None:
            values[forced] = (1.0 - q_star) / 2.0 if forced == ACTION_ACCEPT else q_star / 2.0
            return values
        belief = world_probability(current, current_observation)
        values[ACTION_ACCEPT] = (belief - q_star) / 2.0
        values[ACTION_REJECT] = (q_star - belief) / 2.0
        return values

    def recurse(current: RankBeliefState, current_observation: PolicyObservation) -> tuple[float, np.ndarray]:
        key = (limit, current_observation.history, current.weights.tobytes())
        if cache is not None and key in cache:
            return cache[key]
        values = terminal_values(current, current_observation)
        forced = certificate_action(current_observation.hits, current_observation.depth, current_observation.length, current_observation.threshold)
        if forced is None and current_observation.depth < limit:
            probability = float(current.weights @ probabilities[:, current_observation.depth])
            hit_observation = _synthetic_observation(current_observation, 1)
            miss_observation = _synthetic_observation(current_observation, 0)
            hit_state = update(current, current_observation, 1)
            miss_state = update(current, current_observation, 0)
            values[ACTION_CONTINUE] = -cost + probability * recurse(hit_state, hit_observation)[0] + (1.0 - probability) * recurse(miss_state, miss_observation)[0]
        result = float(values.max()), values
        if cache is not None:
            cache[key] = result
        return result

    values = recurse(state, observation)[1]
    return int(np.argmax(values)), values


def rollout_rank_belief(
    episode: Episode, model: RankLatentBelief, threshold: int, q_star: float, cost: float, lookahead: int,
) -> tuple[int, int, float]:
    state = model.initial_state(episode.observation(0, 0, threshold))
    cache: dict[tuple[int, tuple[int, ...], bytes], tuple[float, np.ndarray]] = {}
    hits = 0
    for depth in range(len(episode.labels) + 1):
        observation = episode.observation(depth, hits, threshold)
        action, _values = rank_belief_action(model, state, observation, q_star, cost, lookahead, cache)
        belief = model.world_probability(state, observation)
        if action != ACTION_CONTINUE:
            return depth, action, belief
        state = model.update(state, observation, int(episode.labels[depth]))
        hits += int(episode.labels[depth])
    raise AssertionError("rank-aware planner did not terminate")


def sequential_predictions(episodes: list[Episode], model: RankLatentBelief, threshold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-step next-hit and world predictions; caller decides how to aggregate by topic."""
    next_predictions, next_outcomes, world_predictions, worlds = [], [], [], []
    for episode in episodes:
        state = model.initial_state(episode.observation(0, 0, threshold))
        hits = 0
        for depth in range(len(episode.labels)):
            observation = episode.observation(depth, hits, threshold)
            next_predictions.append(model.next_hit_probability(state, observation))
            next_outcomes.append(int(episode.labels[depth]))
            world_predictions.append(model.world_probability(state, observation))
            worlds.append(int(episode.labels.sum() >= threshold))
            state = model.update(state, observation, int(episode.labels[depth]))
            hits += int(episode.labels[depth])
    return tuple(np.asarray(values) for values in (next_predictions, next_outcomes, world_predictions, worlds))
