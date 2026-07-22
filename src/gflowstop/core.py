from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import comb, lgamma, log

import numpy as np


ACTION_CONTINUE, ACTION_ACCEPT, ACTION_REJECT = range(3)


@dataclass(frozen=True)
class Topic:
    qid: str
    labels: np.ndarray
    world: int
    scores: np.ndarray | None = None


@dataclass(frozen=True)
class CountModel:
    """Exchangeable beta-binomial model for a topic's remaining relevant count."""

    alpha: float
    beta: float

    def posterior(self, depth: int, hits: int) -> tuple[float, float]:
        if not 0 <= hits <= depth:
            raise ValueError("hits must be between zero and depth")
        return self.alpha + hits, self.beta + depth - hits

    def p_hit(self, depth: int, hits: int) -> float:
        alpha, beta = self.posterior(depth, hits)
        return alpha / (alpha + beta)

    def remaining_variance(self, length: int, depth: int, hits: int) -> float:
        remaining = length - depth
        alpha, beta = self.posterior(depth, hits)
        total = alpha + beta
        return remaining * alpha * beta * (total + remaining) / (total * total * (total + 1.0))

    def _remaining_pmf(self, length: int, depth: int, hits: int) -> np.ndarray:
        remaining = length - depth
        alpha, beta = self.posterior(depth, hits)
        log_pmf = np.empty(remaining + 1, dtype=float)
        log_pmf[0] = lgamma(beta + remaining) - lgamma(beta) + lgamma(alpha + beta) - lgamma(alpha + beta + remaining)
        for count in range(remaining):
            log_pmf[count + 1] = log_pmf[count] + log(remaining - count) - log(count + 1) + log(alpha + count) - log(beta + remaining - count - 1)
        log_pmf -= log_pmf.max()
        pmf = np.exp(log_pmf)
        return pmf / pmf.sum()

    def high_yield_probability(self, length: int, threshold: int, depth: int, hits: int) -> float:
        needed = threshold - hits
        remaining = length - depth
        if needed <= 0:
            return 1.0
        if needed > remaining:
            return 0.0
        return float(self._remaining_pmf(length, depth, hits)[needed:].sum())


def fit_count_model(topics: list[Topic]) -> CountModel:
    """Fit a topic-level beta prior by moments; documents are not pseudo-replicates."""
    if not topics:
        raise ValueError("at least one training topic is required")
    lengths = np.asarray([len(topic.labels) for topic in topics], dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("topics must have at least one document")
    counts = np.asarray([topic.labels.sum() for topic in topics], dtype=float)
    mean = float((counts.sum() + 0.5) / (lengths.sum() + 1.0))
    rates = counts / lengths
    variance = float(np.var(rates, ddof=1)) if len(rates) > 1 else 0.0
    binomial_fraction = float(np.mean(1.0 / lengths))
    base = mean * (1.0 - mean)
    ratio = variance / base if base else 0.0
    if ratio <= binomial_fraction + 1e-12:
        concentration = 10_000.0
    else:
        concentration = (1.0 - binomial_fraction) / (ratio - binomial_fraction) - 1.0
    concentration = float(np.clip(concentration, 1e-3, 10_000.0))
    return CountModel(mean * concentration, (1.0 - mean) * concentration)


def certificate_action(hits: int, depth: int, length: int, threshold: int) -> int | None:
    if hits >= threshold:
        return ACTION_ACCEPT
    if hits + length - depth < threshold:
        return ACTION_REJECT
    return None


@dataclass(frozen=True)
class CountState:
    """Exact predictive quantities for every valid (depth, observed-hit) state."""

    model: CountModel
    length: int
    threshold: int
    belief: np.ndarray
    p_hit: np.ndarray
    remaining_variance: np.ndarray


def count_state(model: CountModel, length: int, threshold: int) -> CountState:
    if length < 1 or threshold < 1:
        raise ValueError("length and threshold must be positive")
    belief = np.full((length + 1, length + 1), np.nan)
    p_hit = np.full_like(belief, np.nan)
    variance = np.full_like(belief, np.nan)
    for depth in range(length, -1, -1):
        hits = np.arange(depth + 1)
        alpha = model.alpha + hits
        beta = model.beta + depth - hits
        total = alpha + beta
        p_hit[depth, : depth + 1] = alpha / total if depth < length else 0.0
        remaining = length - depth
        variance[depth, : depth + 1] = remaining * alpha * beta * (total + remaining) / (total * total * (total + 1.0))
        accept = hits >= threshold
        reject = hits + remaining < threshold
        belief[depth, hits[accept]] = 1.0
        belief[depth, hits[reject]] = 0.0
        middle = ~(accept | reject)
        if middle.any():
            probability = p_hit[depth, : depth + 1][middle]
            belief[depth, hits[middle]] = probability * belief[depth + 1, hits[middle] + 1] + (1.0 - probability) * belief[depth + 1, hits[middle]]
    return CountState(model, length, threshold, belief, p_hit, variance)


def payoff(world: int, action: int, q_star: float) -> float:
    """Normalised asymmetric utility for accept/reject at the yield threshold."""
    if action == ACTION_ACCEPT:
        return (1.0 - q_star) / 2.0 if world else -q_star / 2.0
    if action == ACTION_REJECT:
        return -(1.0 - q_star) / 2.0 if world else q_star / 2.0
    raise ValueError("continue is not terminal")


def terminal_values(
    model: CountModel, length: int, threshold: int, depth: int, hits: int, q_star: float, state: CountState | None = None,
) -> np.ndarray:
    belief = state.belief[depth, hits] if state is not None else model.high_yield_probability(length, threshold, depth, hits)
    return np.array([-np.inf, (belief - q_star) / 2.0, (q_star - belief) / 2.0])


def terminal_action(
    model: CountModel, length: int, threshold: int, depth: int, hits: int, q_star: float, state: CountState | None = None,
) -> tuple[int, float]:
    belief = state.belief[depth, hits] if state is not None else model.high_yield_probability(length, threshold, depth, hits)
    forced = certificate_action(hits, depth, length, threshold)
    if forced is not None:
        return forced, belief
    return (ACTION_ACCEPT if belief >= q_star else ACTION_REJECT), belief


@dataclass(frozen=True)
class CountBellmanTeacher:
    model: CountModel
    length: int
    threshold: int
    q_star: float
    cost: float
    state: CountState
    q_values: np.ndarray
    values: np.ndarray
    targets: np.ndarray

    def action_values(self, depth: int, hits: int) -> np.ndarray:
        return self.q_values[depth, hits]

    def target(self, depth: int, hits: int) -> np.ndarray:
        return self.targets[depth, hits]


def count_bellman_teacher(
    model: CountModel, length: int, threshold: int, q_star: float, cost: float, temperature: float, state: CountState | None = None,
) -> CountBellmanTeacher:
    """Exact finite-horizon Bellman teacher for the fitted count model."""
    if length < 1 or threshold < 1:
        raise ValueError("length and threshold must be positive")
    if not 0.0 < q_star < 1.0 or cost < 0.0 or temperature <= 0.0:
        raise ValueError("q_star and temperature must be in (0, 1); cost must be non-negative")
    state = state or count_state(model, length, threshold)
    q_values = np.full((length + 1, length + 1, 3), -np.inf)
    values = np.full((length + 1, length + 1), -np.inf)
    for depth in range(length, -1, -1):
        for hits in range(depth + 1):
            forced = certificate_action(hits, depth, length, threshold)
            if forced is not None:
                belief = float(forced == ACTION_ACCEPT)
                q_values[depth, hits, forced] = (belief - q_star) / 2.0 if forced == ACTION_ACCEPT else (q_star - belief) / 2.0
            else:
                q_values[depth, hits] = terminal_values(model, length, threshold, depth, hits, q_star, state)
                if depth < length:
                    p_hit = state.p_hit[depth, hits]
                    q_values[depth, hits, ACTION_CONTINUE] = -cost + p_hit * values[depth + 1, hits + 1] + (1.0 - p_hit) * values[depth + 1, hits]
            values[depth, hits] = q_values[depth, hits].max()
    targets = np.zeros_like(q_values)
    for depth in range(length + 1):
        for hits in range(depth + 1):
            logits = q_values[depth, hits] / temperature
            logits -= np.max(logits)
            targets[depth, hits] = np.exp(logits)
            targets[depth, hits] /= targets[depth, hits].sum()
    return CountBellmanTeacher(model, length, threshold, q_star, cost, state, q_values, values, targets)


def _score_features(topic: Topic, depth: int) -> tuple[float, float]:
    if topic.scores is None or depth >= len(topic.scores):
        return 0.0, 0.0
    scale = max(abs(float(topic.scores[0])), 1e-12)
    score = float(topic.scores[depth]) / scale
    next_score = float(topic.scores[depth + 1]) / scale if depth + 1 < len(topic.scores) else score
    return score, score - next_score


def features(
    topic: Topic, model: CountModel, threshold: int, depth: int, hits: int, q_star: float, cost: float, state: CountState | None = None,
) -> np.ndarray:
    """Coherent count state, posterior uncertainty, and observable ranking features."""
    length = len(topic.labels)
    remaining = length - depth
    needed = threshold - hits
    belief = state.belief[depth, hits] if state is not None else model.high_yield_probability(length, threshold, depth, hits)
    p_hit = state.p_hit[depth, hits] if state is not None else (model.p_hit(depth, hits) if remaining else 0.0)
    score, slope = _score_features(topic, depth)
    return np.array(
        [
            depth / length,
            hits,
            needed,
            needed / max(remaining, 1),
            belief,
            p_hit,
            (state.remaining_variance[depth, hits] if state is not None else model.remaining_variance(length, depth, hits)) / (length * length),
            q_star,
            cost,
            score,
            slope,
        ],
        dtype=float,
    )


class ConditionalStoppingPolicy:
    """A log-linear policy distilled from a coherent finite-horizon teacher."""

    def fit(self, x: np.ndarray, target: np.ndarray, steps: int = 1_000, learning_rate: float = 0.2) -> "ConditionalStoppingPolicy":
        if len(x) == 0 or x.shape[0] != target.shape[0]:
            raise ValueError("features and targets must have the same non-zero length")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        x = np.c_[np.ones(len(x)), (x - self.mean) / self.scale]
        self.weights = np.zeros((x.shape[1], 3), dtype=float)
        for _ in range(steps):
            logits = x @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            prediction = np.exp(logits)
            prediction /= prediction.sum(axis=1, keepdims=True)
            self.weights -= learning_rate * (x.T @ (prediction - target)) / len(x)
        return self

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = np.r_[1.0, (x - self.mean) / self.scale] @ self.weights
        logits -= logits.max()
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()


# Compatibility name for earlier callers; this project no longer claims a GFlowNet objective.
ConditionalFlowPolicy = ConditionalStoppingPolicy


def _forced_or_terminal(
    model: CountModel, length: int, threshold: int, depth: int, hits: int, q_star: float, state: CountState | None = None,
) -> tuple[int | None, float]:
    belief = state.belief[depth, hits] if state is not None else model.high_yield_probability(length, threshold, depth, hits)
    forced = certificate_action(hits, depth, length, threshold)
    if forced is not None:
        return forced, belief
    if depth == length:
        return terminal_action(model, length, threshold, depth, hits, q_star, state)
    return None, belief


def _finish(
    model: CountModel, length: int, threshold: int, depth: int, hits: int, q_star: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    action, belief = terminal_action(model, length, threshold, depth, hits, q_star, state)
    return depth, action, belief


def decide_amortized(
    topic: Topic, model: CountModel, policy: ConditionalStoppingPolicy, threshold: int, q_star: float, cost: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    hits, length = 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(length + 1):
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        action = int(np.argmax(policy.probabilities(features(topic, model, threshold, depth, hits, q_star, cost, state))))
        if action != ACTION_CONTINUE:
            return depth, action, belief
        hits += int(topic.labels[depth])
    raise AssertionError("terminal state was not reached")


def decide_bellman(topic: Topic, teacher: CountBellmanTeacher) -> tuple[int, int, float]:
    """Execute the exact count-state teacher using only revealed labels."""
    hits = 0
    for depth in range(teacher.length + 1):
        action, belief = _forced_or_terminal(teacher.model, teacher.length, teacher.threshold, depth, hits, teacher.q_star, teacher.state)
        if action is not None:
            return depth, action, belief
        action = int(np.argmax(teacher.action_values(depth, hits)))
        if action != ACTION_CONTINUE:
            return depth, action, belief
        hits += int(topic.labels[depth])
    raise AssertionError("terminal state was not reached")


def decide_one_step_evsi(
    topic: Topic, model: CountModel, threshold: int, q_star: float, cost: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    """Exact one-step expected value of sample information baseline."""
    hits, length = 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(length + 1):
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        stop_values = terminal_values(model, length, threshold, depth, hits, q_star, state)
        p_hit = state.p_hit[depth, hits]
        hit_value = terminal_values(model, length, threshold, depth + 1, hits + 1, q_star, state).max()
        miss_value = terminal_values(model, length, threshold, depth + 1, hits, q_star, state).max()
        if -cost + p_hit * hit_value + (1.0 - p_hit) * miss_value <= stop_values.max():
            return depth, int(np.argmax(stop_values)), belief
        hits += int(topic.labels[depth])
    raise AssertionError("terminal state was not reached")


def decide_evpi_bound(
    topic: Topic, model: CountModel, threshold: int, q_star: float, cost: float, batch: int = 1, state: CountState | None = None,
) -> tuple[int, int, float]:
    """Cheap EVPI-bound heuristic; unlike EVSI it does not price the miss branch exactly."""
    hits, depth, length = 0, 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    while depth < length:
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        width = min(batch, length - depth)
        probabilities = [state.p_hit[depth + offset, hits] for offset in range(width)]
        p_any = 1.0 - float(np.prod(1.0 - np.asarray(probabilities)))
        evpi = min(belief * (1.0 - q_star), (1.0 - belief) * q_star)
        if p_any * evpi <= width * cost:
            return _finish(model, length, threshold, depth, hits, q_star, state)
        for _ in range(width):
            hits += int(topic.labels[depth])
            depth += 1
    return _finish(model, length, threshold, depth, hits, q_star, state)


def decide_fixed(
    topic: Topic, model: CountModel, threshold: int, q_star: float, budget: int, certify: bool = True, state: CountState | None = None,
) -> tuple[int, int, float]:
    hits, length = 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(min(budget, length)):
        if certify:
            action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
            if action is not None:
                return depth, action, belief
        hits += int(topic.labels[depth])
    depth = min(budget, length)
    action, belief = terminal_action(model, length, threshold, depth, hits, q_star, state)
    return depth, action, belief


def decide_consecutive_misses(
    topic: Topic, model: CountModel, threshold: int, q_star: float, misses: int, state: CountState | None = None,
) -> tuple[int, int, float]:
    """Stop after a tuned run of observed non-relevant documents."""
    hits, run, length = 0, 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(length):
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        hit = int(topic.labels[depth])
        hits += hit
        run = 0 if hit else run + 1
        if run >= misses:
            return _finish(model, length, threshold, depth + 1, hits, q_star, state)
    return _finish(model, length, threshold, length, hits, q_star, state)


def decide_hit_probability_cutoff(
    topic: Topic, model: CountModel, threshold: int, q_star: float, cutoff: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    hits, length = 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(length):
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        if state.p_hit[depth, hits] <= cutoff:
            return _finish(model, length, threshold, depth, hits, q_star, state)
        hits += int(topic.labels[depth])
    return _finish(model, length, threshold, length, hits, q_star, state)


def decide_score_cutoff(
    topic: Topic, model: CountModel, threshold: int, q_star: float, cutoff: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    """Actual BM25-score cutoff on score normalised by the topic's top score."""
    if topic.scores is None:
        raise ValueError("score cutoff requires topic scores")
    hits, length = 0, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth in range(length):
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        score, _ = _score_features(topic, depth)
        if score <= cutoff:
            return _finish(model, length, threshold, depth, hits, q_star, state)
        hits += int(topic.labels[depth])
    return _finish(model, length, threshold, length, hits, q_star, state)


def decide_kneedle(
    topic: Topic, model: CountModel, threshold: int, q_star: float, first_batch: int, rho_threshold: float, state: CountState | None = None,
) -> tuple[int, int, float]:
    """Online Kneedle on exponentially spaced observed ranking prefixes."""
    hits, next_check, length = 0, first_batch, len(topic.labels)
    state = state or count_state(model, length, threshold)
    for depth, hit in enumerate(topic.labels, start=1):
        hits += int(hit)
        action, belief = _forced_or_terminal(model, length, threshold, depth, hits, q_star, state)
        if action is not None:
            return depth, action, belief
        if depth < next_check and depth < length:
            continue
        observed = topic.labels[:depth]
        total = int(observed.sum())
        if total:
            cumulative = np.r_[0, np.cumsum(observed)]
            curve = cumulative / total
            knee = int(np.argmax(curve - np.arange(depth + 1) / depth))
            pre = cumulative[knee] / max(knee, 1)
            post = (1 + cumulative[-1] - cumulative[knee]) / max(depth - knee, 1)
            if pre / post >= rho_threshold:
                return _finish(model, length, threshold, depth, hits, q_star, state)
        next_check *= 2
    return _finish(model, length, threshold, length, hits, q_star, state)


def _qbcb_stop_positive(r: int, target_recall: float, alpha: float) -> int:
    """Small-r exact Binomial CDF; returns r+1 when certification is impossible."""
    for j in range(1, r + 1):
        cdf = sum(comb(r, k) * target_recall**k * (1.0 - target_recall) ** (r - k) for k in range(j))
        if cdf >= 1.0 - alpha:
            return j
    return r + 1


def decide_qbcb(
    topic: Topic, model: CountModel, threshold: int, q_star: float, sample_positives: int, target_recall: float, alpha: float, seed: int,
) -> tuple[int, int, float]:
    """Adapted QBCB kept for a separate high-recall track, not the default triage comparison."""
    digest = blake2b(f"{seed}|{topic.qid}|{sample_positives}|{target_recall}|{alpha}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    sample, positives = [], []
    for index in rng.permutation(len(topic.labels)):
        sample.append(int(index))
        if topic.labels[index]:
            positives.append(int(index))
            if len(positives) == sample_positives:
                break
    j = _qbcb_stop_positive(sample_positives, target_recall, alpha)
    prefix = len(topic.labels) if len(positives) < sample_positives or j > sample_positives else sorted(positives)[j - 1] + 1
    screened = sorted(set(sample).union(range(prefix)))
    hits = int(topic.labels[screened].sum())
    action, belief = terminal_action(model, len(topic.labels), threshold, len(screened), hits, q_star)
    return len(screened), action, belief


def realised_utility(topic: Topic, depth: int, action: int, q_star: float, cost: float) -> float:
    return payoff(topic.world, action, q_star) - cost * depth


def full_information_utility(topic: Topic, q_star: float) -> float:
    return payoff(topic.world, ACTION_ACCEPT if topic.world else ACTION_REJECT, q_star)


# Compatibility alias for output consumers of the original experiment.
oracle_utility = full_information_utility
