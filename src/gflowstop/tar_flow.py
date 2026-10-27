"""Target-recall stopping (Track R) with a learned stopping flow network.

A ranking is reviewed in order and the reviewer may stop at a fixed grid of
checkpoints. Stopping after ``n`` of ``N`` documents is terminal, with utility

    U(n) = -(n / N + penalty * [recall(n) < target]) / (1 + penalty),

the review fraction plus a penalty for missing the target recall, rescaled so
every penalty spans the same range. The recall term depends on labels the
reviewer has not seen, so the terminal edge passes through a chance node; the
flow network's geometric aggregation makes its stop value the posterior
expected utility, never the realised one. Both the target and the penalty are
network inputs, so one network covers every operating point.

Every feature at a checkpoint is computed from the labels already reviewed;
``prefix_features`` takes only that prefix, and the replay rule calls it with
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flow import ChainData, FlowStoppingNetwork

CHECKPOINTS = 200
RATE_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1)
RATE_WINDOWS = (10, 25, 50, 100, 200)
REMAINING_FRACTIONS = (0.02, 0.05, 0.1)
MINIMUM_WINDOW = 5
FIT_BINS = 20


def checkpoints(length: int, count: int = CHECKPOINTS) -> np.ndarray:
    """Review depths at which a stop may be taken: ``ceil(j N / count)``, without repeats, ending at ``N``."""
    if length < 1:
        raise ValueError("a ranking needs at least one document")
    return np.unique(np.ceil(np.arange(1, count + 1) * length / count).astype(int))


def _window_rates(cumulative: np.ndarray, depths: np.ndarray, windows: np.ndarray) -> np.ndarray:
    windows = np.minimum(depths, windows)
    return (cumulative[depths] - cumulative[depths - windows]) / windows


def _exponential_tail(cumulative: np.ndarray, depths: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected relevant documents after each depth under a log-linear rate fitted to the reviewed prefix.

    The prefix is cut into ``FIT_BINS`` equal bins and their counts fitted by
    Poisson regression of ``log rate = a + b * position / length`` (a few
    Newton steps, ridge-stabilised). Returns the remaining-count estimate,
    capped at the unreviewed documents, and the fitted decay ``b``.
    """
    edges = np.rint(depths[:, None] * np.arange(FIT_BINS + 1)[None, :] / FIT_BINS).astype(int)
    counts = (cumulative[edges[:, 1:]] - cumulative[edges[:, :-1]]).astype(float)
    widths = np.maximum(edges[:, 1:] - edges[:, :-1], 1).astype(float)
    centres = (edges[:, 1:] + edges[:, :-1]) / 2.0 / length
    valid = edges[:, 1:] > edges[:, :-1]
    total = counts.sum(axis=1)
    a = np.log((total + 0.5) / depths)
    b = np.zeros(len(depths))
    for _ in range(12):
        mu = np.where(valid, widths * np.exp(np.clip(a[:, None] + b[:, None] * centres, -50, 50)), 0.0)
        residual = np.where(valid, counts - mu, 0.0)
        grad_a, grad_b = residual.sum(axis=1), (residual * centres).sum(axis=1) - 1e-3 * b
        h_aa = mu.sum(axis=1) + 1e-6
        h_ab = (mu * centres).sum(axis=1)
        h_bb = (mu * centres * centres).sum(axis=1) + 1e-3
        det = h_aa * h_bb - h_ab * h_ab
        a = a + (h_bb * grad_a - h_ab * grad_b) / det
        b = np.clip(b + (h_aa * grad_b - h_ab * grad_a) / det, -500.0, 50.0)
    start, stop = depths / length, 1.0
    # integral of exp(a + b x) over [start, 1], in documents
    small = np.abs(b) < 1e-6
    span = np.where(small, stop - start, (np.exp(np.clip(b * stop, -50, 50)) - np.exp(np.clip(b * start, -50, 50))) / np.where(small, 1.0, b))
    remaining = np.minimum(np.exp(np.clip(a, -50, 50)) * length * span, length - depths)
    return np.maximum(remaining, 0.0), b


def base_features(labels: np.ndarray, length: int, depths: np.ndarray) -> np.ndarray:
    """Target-independent observable state at each depth, reading only ``labels[:depth]``.

    ``labels`` may be a prefix of the ranking; ``length`` is the full ranking's
    size, which a reviewer knows before starting.
    """
    labels = np.asarray(labels) > 0
    depths = np.asarray(depths, dtype=int)
    if np.any(depths < 1) or np.any(depths > len(labels)) or len(labels) > length:
        raise ValueError("depths must lie within the reviewed labels")
    cumulative = np.r_[0, np.cumsum(labels)]
    found = cumulative[depths].astype(float)
    remaining = (length - depths).astype(float)
    fraction_windows = np.maximum(MINIMUM_WINDOW, np.rint(np.asarray(RATE_FRACTIONS) * length)).astype(int)
    fraction_rates = [_window_rates(cumulative, depths, window) for window in fraction_windows]
    absolute_rates = [_window_rates(cumulative, depths, window) for window in RATE_WINDOWS]
    positions = np.flatnonzero(labels)
    last = np.searchsorted(positions, depths) - 1
    since_last = np.where(last >= 0, depths - (positions[np.maximum(last, 0)] + 1) if len(positions) else depths, depths).astype(float)
    precision = found / depths
    estimates = []
    for fraction in REMAINING_FRACTIONS:
        window = max(MINIMUM_WINDOW, int(round(fraction * length)))
        expected = _window_rates(cumulative, depths, window) * remaining
        estimates += [np.log1p(expected) - np.log1p(found), np.where(found + expected > 0, found / np.maximum(found + expected, 1e-12), 0.0)]
    tail, decay = _exponential_tail(cumulative, depths, length)
    half = cumulative[depths] - cumulative[depths // 2]
    tenth = cumulative[depths] - cumulative[depths - np.maximum(1, depths // 10)]
    return np.column_stack(
        [
            depths / length,
            np.full(len(depths), np.log(length) / 10.0),
            np.log1p(found) / 5.0,
            precision,
            *fraction_rates,
            *absolute_rates,
            np.log1p(since_last) / 10.0,
            since_last / length,
            np.log1p(since_last * precision),
            *estimates,
            np.log1p(tail) - np.log1p(found),
            np.where(found + tail > 0, found / np.maximum(found + tail, 1e-12), 0.0),
            np.tanh(decay / 50.0),
            half / np.maximum(found, 1.0),
            tenth / np.maximum(found, 1.0),
        ]
    ).astype(np.float32)


ESTIMATED_RECALL_COLUMNS = (18, 20, 22, 24)  # window-rate and fitted-decay recall estimates in base_features


def with_operating_point(base: np.ndarray, target: float, penalty: float) -> np.ndarray:
    """Append how far each estimated recall is from the target (log-odds), then the target and penalty."""
    estimated = np.clip(base[:, ESTIMATED_RECALL_COLUMNS], 1e-4, 1 - 1e-4)
    margin = np.log(estimated / (1 - estimated)) - np.log(target / (1 - target))
    return np.c_[base, margin / 4.0, np.full((len(base), 2), (target, np.log(penalty) / 4.0))].astype(np.float32)


def prefix_features(prefix: np.ndarray, length: int, target: float, penalty: float) -> np.ndarray:
    """Observable state after reviewing ``prefix`` (labels in review order) of a ``length``-document ranking."""
    if not 1 <= len(prefix) <= length:
        raise ValueError("the prefix must hold between one and length documents")
    return with_operating_point(base_features(prefix, length, np.array([len(prefix)])), target, penalty)[0]


def ranking_features(labels: np.ndarray, target: float, penalty: float, count: int = CHECKPOINTS) -> tuple[np.ndarray, np.ndarray]:
    """Features at every checkpoint of one ranking; each row reads only its own reviewed prefix."""
    depths = checkpoints(len(labels), count)
    return depths, with_operating_point(base_features(labels, len(labels), depths), target, penalty)


def stop_utilities(labels: np.ndarray, depths: np.ndarray, target: float, penalty: float) -> np.ndarray:
    labels = np.asarray(labels) > 0
    relevant = int(labels.sum())
    found = np.cumsum(labels)[depths - 1]
    recall = found / relevant if relevant else np.ones(len(depths))
    return -(depths / len(labels) + penalty * (recall < target - 1e-12)) / (1.0 + penalty)


def ranking_chain(
    labels: np.ndarray, target: float, penalty: float, count: int = CHECKPOINTS, weight: float = 1.0,
    base: tuple[np.ndarray, np.ndarray] | None = None,
) -> ChainData:
    depths, base_rows = base if base is not None else (checkpoints(len(labels), count), None)
    features = with_operating_point(base_features(labels, len(labels), depths) if base_rows is None else base_rows, target, penalty)
    utilities = stop_utilities(labels, depths, target, penalty)
    next_row = np.r_[np.arange(1, len(depths)), -1]
    return ChainData(features, utilities[:, None], next_row, np.full(len(depths), weight / len(depths)))


def training_chains(
    rankings: list[np.ndarray], targets: tuple[float, ...] = (0.8, 0.9, 0.95), penalty_range: tuple[float, float] = (0.25, 64.0),
    copies: int = 3, count: int = CHECKPOINTS, seed: int = 0, weights: list[float] | None = None,
) -> ChainData:
    """Every ranking with ``copies`` random (target, penalty) pairs; each ranking weighs ``weight`` in total."""
    rng = np.random.default_rng(seed)
    weights = [1.0] * len(rankings) if weights is None else weights
    parts = []
    for labels, weight in zip(rankings, weights):
        if not np.any(labels):
            continue
        depths = checkpoints(len(labels), count)
        base = (depths, base_features(labels, len(labels), depths))
        for _ in range(copies):
            target = float(rng.choice(targets))
            penalty = float(np.exp(rng.uniform(*np.log(penalty_range))))
            parts.append(ranking_chain(labels, target, penalty, count, weight / copies, base))
    return ChainData.concatenate(parts)


def _stop_probability(stop: np.ndarray, cont: np.ndarray, can_continue: np.ndarray, beta: float) -> np.ndarray:
    if np.isinf(beta):
        probability = (stop >= cont).astype(float)
    else:
        probability = 0.5 * (1.0 + np.tanh(0.5 * beta * (stop - cont)))
    return np.where(can_continue, probability, 1.0)


class RecallFlowEnsemble:
    """Several independently trained networks whose flows are averaged in log space."""

    def __init__(self, members: list["RecallFlowNetwork"]) -> None:
        if not members or len({member.beta for member in members}) != 1:
            raise ValueError("an ensemble needs members with one beta")
        self.members, self.beta = members, members[0].beta

    def values(self, features, review, review_next, penalty):
        outputs = [member.values(features, review, review_next, penalty) for member in self.members]
        return np.mean([stop for stop, _ in outputs], axis=0), np.mean([cont for _, cont in outputs], axis=0)

    def stop_probabilities(self, features, review, review_next, penalty, can_continue):
        return _stop_probability(*self.values(features, review, review_next, penalty), can_continue, self.beta)

    def miss_probabilities(self, features):
        return np.mean([member.miss_probabilities(features) for member in self.members], axis=0)


@dataclass
class FlowStopPolicy:
    """A fitted network used as a stopping rule at one target and penalty."""

    network: "FlowStoppingNetwork | RecallFlowNetwork | RecallFlowEnsemble"
    target: float
    penalty: float
    threshold: float = 0.5
    count: int = CHECKPOINTS

    def stop_probabilities(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``P_F(stop)`` at every checkpoint, the reviewer's stopping indicator."""
        depths, features = ranking_features(labels, self.target, self.penalty, self.count)
        if isinstance(self.network, (RecallFlowNetwork, RecallFlowEnsemble)):
            review_next = np.r_[depths[1:], len(labels)] / len(labels)
            return depths, self.network.stop_probabilities(features, depths / len(labels), review_next, self.penalty, depths < len(labels))
        available = np.c_[depths < len(labels), np.ones(len(depths), dtype=bool)]
        return depths, self.network.forward_policy(features, available)[:, 1]

    def stop(self, labels: np.ndarray) -> int:
        depths, probabilities = self.stop_probabilities(labels)
        fired = np.flatnonzero(probabilities >= self.threshold)
        return int(depths[fired[0]]) if len(fired) else len(labels)


@dataclass
class RecallChains:
    """Track-R chains with the utility's parts kept apart: review fraction and realised miss."""

    features: np.ndarray
    review: np.ndarray
    review_next: np.ndarray
    miss: np.ndarray
    penalty: np.ndarray
    next_row: np.ndarray
    weights: np.ndarray

    @staticmethod
    def build(
        rankings: list[np.ndarray], targets: tuple[float, ...] = (0.8, 0.9, 0.95), penalty_range: tuple[float, float] = (0.25, 64.0),
        copies: int = 3, count: int = CHECKPOINTS, seed: int = 0, weights: list[float] | None = None,
    ) -> "RecallChains":
        rng = np.random.default_rng(seed)
        weights = [1.0] * len(rankings) if weights is None else weights
        parts: dict[str, list[np.ndarray]] = {name: [] for name in ("features", "review", "review_next", "miss", "penalty", "next_row", "weights")}
        offset = 0
        for labels, weight in zip(rankings, weights):
            labels = np.asarray(labels) > 0
            if not labels.any():
                continue
            depths = checkpoints(len(labels), count)
            base = base_features(labels, len(labels), depths)
            recall = np.cumsum(labels)[depths - 1] / labels.sum()
            for _ in range(copies):
                target = float(rng.choice(targets))
                penalty = float(np.exp(rng.uniform(*np.log(penalty_range))))
                parts["features"].append(with_operating_point(base, target, penalty))
                parts["review"].append(depths / len(labels))
                parts["review_next"].append(np.r_[depths[1:], len(labels)] / len(labels))
                parts["miss"].append((recall < target - 1e-12).astype(float))
                parts["penalty"].append(np.full(len(depths), penalty))
                parts["next_row"].append(np.r_[np.arange(1, len(depths)), -1] + np.r_[np.full(len(depths) - 1, offset), 0])
                parts["weights"].append(np.full(len(depths), weight / copies / len(depths)))
                offset += len(depths)
        return RecallChains(**{name: np.concatenate(values) for name, values in parts.items()})


class RecallFlowNetwork:
    """Stopping flow network for target recall, parameterised by the reward's structure.

    With ``U = -(review + penalty * miss) / (1 + penalty)`` the flows are

        log F(x, stop)     / beta = -(n/N + penalty * p_miss(x)) / (1 + penalty)
        log F(x, continue) / beta = -(c(x) + penalty * m(x)) / (1 + penalty) + e(x)

    The future review fraction ``c`` is constrained to lie between the next
    checkpoint's review fraction and one, so continuing is never priced below
    its own certain cost.

    ``p_miss`` is the posterior probability that the target recall has not yet
    been reached: the geometric chance node after ``stop``, learned as a
    classifier of realised misses and independent of the penalty. ``c``, ``m``
    and ``e`` are the expected review fraction, miss probability and entropy
    bonus ``-(1/beta) log P_F`` still to come after continuing, under the
    current forward policy ``P_F = softmax(beta Q)``. They are learned by soft
    policy iteration: every round evaluates the frozen copy's policy exactly
    along each realised chain and regresses on the value at the next
    checkpoint. Only the policy's own choices are averaged, and those depend
    only on observed features.

    Two comparators keep this parameterisation and change only the continue
    targets. ``aggregator='hindsight'`` uses the best stop further along each
    realised chain, chosen with that chain's future labels in hand: the
    original fixed-chain flow target at ``beta = inf``. ``aggregator='myopic'``
    uses a stop at the next checkpoint, a one-step lookahead.
    """

    def __init__(
        self, beta: float = 1e4, hidden: tuple[int, ...] = (64, 64), rounds: int = 120, steps: int = 25, batch_size: int = 4096,
        learning_rate: float = 1e-3, weight_decay: float = 1e-5, seed: int = 0, aggregator: str = "geometric",
    ) -> None:
        if not beta > 0.0:
            raise ValueError("beta must be positive")
        if aggregator not in ("geometric", "hindsight", "myopic"):
            raise ValueError("aggregator must be geometric, hindsight or myopic")
        self.aggregator = aggregator
        self.beta, self.hidden, self.rounds, self.steps, self.batch_size = float(beta), tuple(hidden), rounds, steps, batch_size
        self.learning_rate, self.weight_decay, self.seed = learning_rate, weight_decay, seed

    def _mlp(self, inputs: int, outputs: int):
        import torch

        layers, width = [], inputs
        for size in self.hidden:
            layers += [torch.nn.Linear(width, size), torch.nn.SiLU()]
            width = size
        layers.append(torch.nn.Linear(width, outputs))
        return torch.nn.Sequential(*layers)

    def _heads(self, x):
        import torch

        miss_now = self.miss_model_(x[:, :-1])[:, 0]  # the miss classifier never sees the penalty
        rest = self.continue_model_(x)
        return miss_now, rest[:, 0], rest[:, 1], rest[:, 2]

    @staticmethod
    def _future_review(cost, review_next):
        import torch

        return review_next + (1.0 - review_next) * torch.sigmoid(cost)

    def _values(self, x, review, review_next, penalty):
        import torch

        with torch.no_grad():
            outputs = [torch.cat(chunk) for chunk in zip(*(self._heads(part) for part in torch.split(x, 65536)))]
        miss_now, cost, miss_later, entropy = outputs
        stop = -(review + penalty * torch.sigmoid(miss_now)) / (1.0 + penalty)
        cont = -(self._future_review(cost, review_next) + penalty * torch.sigmoid(miss_later)) / (1.0 + penalty) + entropy
        return stop, cont

    def _stop_probability(self, stop, cont, can_continue):
        import torch

        if np.isinf(self.beta):
            p = (stop >= cont).float()
        else:
            p = torch.sigmoid(self.beta * (stop - cont))
        return torch.where(can_continue, p, torch.ones_like(p))

    def fit(self, data: RecallChains) -> "RecallFlowNetwork":
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mean_ = data.features.mean(axis=0)
        self.scale_ = data.features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        x = torch.as_tensor((data.features - self.mean_) / self.scale_, dtype=torch.float32)
        self.miss_model_ = self._mlp(x.shape[1] - 1, 1)
        self.continue_model_ = self._mlp(x.shape[1], 3)
        parameters = list(self.miss_model_.parameters()) + list(self.continue_model_.parameters())
        optimiser = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        review = torch.as_tensor(data.review, dtype=torch.float32)
        review_next = torch.as_tensor(data.review_next, dtype=torch.float32)
        miss = torch.as_tensor(data.miss, dtype=torch.float32)
        penalty = torch.as_tensor(data.penalty, dtype=torch.float32)
        can_continue = torch.as_tensor(data.next_row >= 0)
        next_row = np.maximum(data.next_row, 0)
        weights = torch.as_tensor(data.weights / data.weights.mean(), dtype=torch.float32)
        chain = ChainData(data.features[:, :1], np.zeros((len(data.review), 1)), data.next_row)
        heights = chain.heights()
        order = np.argsort(heights, kind="stable")
        boundaries = np.searchsorted(heights[order], np.arange(heights.max() + 2))
        review_np, miss_np = data.review, data.miss
        fixed = None
        if self.aggregator == "myopic":
            fixed = [data.review[next_row], data.miss[next_row], np.zeros(len(next_row))]
        elif self.aggregator == "hindsight":
            # best realised stop at or after each row, then read at the next row
            utility = -(data.review + data.penalty * data.miss) / (1.0 + data.penalty)
            best = np.arange(len(utility))
            for height in range(1, heights.max() + 1):
                rows = order[boundaries[height] : boundaries[height + 1]]
                following = best[data.next_row[rows]]
                best[rows] = np.where(utility[following] > utility[rows], following, rows)
            fixed = [data.review[best[next_row]], data.miss[best[next_row]], np.zeros(len(next_row))]
        self.history_ = []
        for _ in range(self.rounds):
            if fixed is not None:
                targets = [torch.as_tensor(values, dtype=torch.float32) for values in fixed]
                self._train_round(x, miss, targets, review_next, can_continue, weights, optimiser, rng)
                continue
            stop, cont = self._values(x, review, review_next, penalty)
            p_stop = self._stop_probability(stop, cont, can_continue).numpy().astype(float)
            p_cont = 1.0 - p_stop
            if np.isfinite(self.beta):
                with np.errstate(divide="ignore", invalid="ignore"):
                    bonus = -(np.where(p_stop > 0, p_stop * np.log(p_stop), 0.0) + np.where(p_cont > 0, p_cont * np.log(p_cont), 0.0)) / self.beta
            else:
                bonus = np.zeros(len(p_stop))
            cost_after, miss_after, entropy_after = np.zeros(len(p_stop)), np.zeros(len(p_stop)), np.zeros(len(p_stop))
            for height in range(heights.max() + 1):
                rows = order[boundaries[height] : boundaries[height + 1]]
                cost_after[rows] = p_stop[rows] * review_np[rows]
                miss_after[rows] = p_stop[rows] * miss_np[rows]
                entropy_after[rows] = bonus[rows]
                if height:
                    following = data.next_row[rows]
                    cost_after[rows] += p_cont[rows] * cost_after[following]
                    miss_after[rows] += p_cont[rows] * miss_after[following]
                    entropy_after[rows] += p_cont[rows] * entropy_after[following]
            targets = [torch.as_tensor(values[next_row], dtype=torch.float32) for values in (cost_after, miss_after, entropy_after)]
            self._train_round(x, miss, targets, review_next, can_continue, weights, optimiser, rng)
        return self

    def _train_round(self, x, miss, targets, review_next, can_continue, weights, optimiser, rng) -> None:
        import torch

        total = 0.0
        for _ in range(self.steps):
            batch = torch.as_tensor(rng.integers(0, len(x), size=min(self.batch_size, len(x))))
            miss_now, cost, miss_later, entropy = self._heads(x[batch])
            live = can_continue[batch].float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(miss_now, miss[batch], reduction="none")
            loss = loss + live * (
                torch.nn.functional.binary_cross_entropy_with_logits(miss_later, targets[1][batch], reduction="none")
                + 100.0 * (self._future_review(cost, review_next[batch]) - targets[0][batch]) ** 2
                + 1e4 * (entropy - targets[2][batch]) ** 2
            )
            loss = (loss * weights[batch]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        self.history_.append(total / self.steps)

    def values(self, features: np.ndarray, review: np.ndarray, review_next: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
        """``log F / beta`` of stop and continue at each state."""
        import torch

        x = torch.as_tensor((np.atleast_2d(features) - self.mean_) / self.scale_, dtype=torch.float32)
        stop, cont = self._values(
            x, torch.as_tensor(review, dtype=torch.float32), torch.as_tensor(review_next, dtype=torch.float32), torch.full((len(x),), float(penalty)),
        )
        return stop.numpy().astype(float), cont.numpy().astype(float)

    def stop_probabilities(
        self, features: np.ndarray, review: np.ndarray, review_next: np.ndarray, penalty: float, can_continue: np.ndarray,
    ) -> np.ndarray:
        """The forward policy's stop probability ``P_F(stop | x)``, the stopping indicator."""
        return _stop_probability(*self.values(features, review, review_next, penalty), can_continue, self.beta)

    def miss_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Posterior probability that the target recall has not been reached at each state."""
        import torch

        x = torch.as_tensor((np.atleast_2d(features) - self.mean_) / self.scale_, dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.miss_model_(x[:, :-1])[:, 0]).numpy().astype(float)
