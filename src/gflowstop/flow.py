"""Flow networks for stopping in a stochastic environment.

A stopping problem is a tree with two kinds of node. At an *agent* node the
reviewer chooses ``continue`` or a terminal action; at a *chance* node the
environment reveals something the reviewer did not choose: the next label after
``continue``, or the latent world after a terminal action. A GFlowNet assumes
every edge is a choice, so it sums child flows everywhere. Summing over chance
edges is the hindsight error recorded in CLAIMS_AND_APPROACH.md: the realised
labels are treated as if the reviewer had picked them.

The networks here keep the GFlowNet constraints at agent nodes,

    F(h) = sum_a F(h, a),        P_F(a | h) = F(h, a) / F(h),

with terminal reward ``R = exp(beta * utility)``, and choose how a chance node
aggregates its children:

``geometric``
    log F(h, a) = E[log F(child) | h]. This is the risk-neutral choice: with
    ``Q = log F / beta`` it is the soft Bellman equation, ``V(h) =
    (1/beta) logsumexp(beta Q(h, .))`` and ``Q(h, continue) = E[V(h') | h]``,
    so as ``beta`` grows the greedy policy is the Bayes-optimal one. With no
    chance node at all (a known world), it is an ordinary tree GFlowNet whose
    terminal distribution is proportional to the reward.
``arithmetic``
    F(h, a) = E[F(child) | h], the expected-flow constraint of expected flow
    networks (Jiralerspong et al., ICLR 2024). In log space it is
    ``(1/beta) log E exp(beta V)``, a risk-seeking certainty equivalent that
    tends to the best outcome, not the expected one, as ``beta`` grows.
``hindsight``
    The original fixed-chain construction: each realised ranking is a chain
    whose continue flow sums every terminal reward further down that chain,
    and the shared policy averages those log flows over rankings consistent
    with the observed prefix. It prices continuing as if the reviewer could
    pick its best future stop with the future labels in hand.

``ExactFlowNetwork`` solves all three on a finite law over label sequences,
so they can be compared with ``toys.ExactOracle``. ``FlowStoppingNetwork`` learns
the geometric (or hindsight) network from realised rankings alone, for use
where no law is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .core import ACTION_ACCEPT, ACTION_CONTINUE, ACTION_REJECT, certificate_action

AGGREGATORS = ("geometric", "arithmetic", "hindsight")


def _logsumexp(values: np.ndarray, axis: int | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    top = np.max(values, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    scaled = np.exp(values - top)
    if weights is not None:
        scaled = scaled * weights
    with np.errstate(divide="ignore"):
        result = np.log(np.sum(scaled, axis=axis, keepdims=True)) + top
    return np.squeeze(result, axis=axis) if axis is not None else float(result.squeeze())


def _payoff(world: np.ndarray, action: int, q_star: float) -> np.ndarray:
    if action == ACTION_ACCEPT:
        return np.where(world, (1.0 - q_star) / 2.0, -q_star / 2.0)
    return np.where(world, -(1.0 - q_star) / 2.0, q_star / 2.0)


@dataclass(eq=False)
class ExactFlowNetwork:
    """Exact flows for the threshold-decision task (Track U) on a finite sequence law.

    ``q_values[h]`` is ``log F(h, .) / beta`` for continue, accept and reject;
    unavailable actions are ``-inf``. Certificates force the only legal action,
    exactly as ``ExactOracle`` does.
    """

    sequences: np.ndarray
    probabilities: np.ndarray
    threshold: int
    q_star: float
    cost: float
    beta: float
    aggregator: str = "geometric"
    q_values: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        self.sequences = np.asarray(self.sequences, dtype=int)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("sequence probabilities must sum to one")
        self.length = int(self.sequences.shape[1])
        self._world = self.sequences.sum(axis=1) >= self.threshold
        self._cumulative = np.c_[np.zeros(len(self.sequences), dtype=int), np.cumsum(self.sequences, axis=1)]
        self._solve(())

    def _conditional(self, history: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        mask = np.all(self.sequences[:, : len(history)] == np.asarray(history, dtype=int), axis=1) if history else np.ones(len(self.sequences), bool)
        weights = self.probabilities[mask]
        if weights.sum() <= 0.0:
            raise ValueError("history has zero probability")
        return np.flatnonzero(mask), weights / weights.sum()

    def _terminal_utilities(self, rows: np.ndarray, depth: int) -> np.ndarray:
        """Realised utility of continue (nan), accept and reject at ``depth`` for each sequence row."""
        utilities = np.full((len(rows), 3), np.nan)
        hits = self._cumulative[rows, depth]
        world = self._world[rows]
        for row, (hit, observed_world) in enumerate(zip(hits, world)):
            forced = certificate_action(int(hit), depth, self.length, self.threshold)
            for action in (ACTION_ACCEPT, ACTION_REJECT):
                if forced is None or forced == action:
                    utilities[row, action] = float(_payoff(np.array(observed_world), action, self.q_star)) - self.cost * depth
        return utilities

    @lru_cache(maxsize=None)
    def _hindsight_return(self, row: int, depth: int) -> float:
        """(1/beta) log of every terminal reward reachable along one realised chain from ``depth``."""
        utilities = self._terminal_utilities(np.array([row]), depth)[0, 1:]
        logs = list(self.beta * utilities[np.isfinite(utilities)])
        hits = int(self._cumulative[row, depth])
        if depth < self.length and certificate_action(hits, depth, self.length, self.threshold) is None:
            logs.append(self.beta * self._hindsight_return(row, depth + 1))
        return _logsumexp(np.asarray(logs)) / self.beta

    @lru_cache(maxsize=None)
    def _solve(self, history: tuple[int, ...]) -> float:
        depth, hits = len(history), sum(history)
        rows, weights = self._conditional(history)
        values = np.full(3, -np.inf)
        forced = certificate_action(hits, depth, self.length, self.threshold)
        terminal = self._terminal_utilities(rows, depth)
        for action in (ACTION_ACCEPT, ACTION_REJECT):
            if forced is not None and forced != action:
                continue
            realised = terminal[:, action]
            if self.aggregator == "arithmetic":
                values[action] = _logsumexp(self.beta * realised, weights=weights) / self.beta
            else:
                values[action] = float(weights @ realised)
        if forced is None and depth < self.length:
            p_hit = float(weights @ self.sequences[rows, depth])
            children = [(label, probability) for label, probability in ((1, p_hit), (0, 1.0 - p_hit)) if probability > 0.0]
            child_values = np.array([self._solve(history + (label,)) for label, _ in children])
            child_weights = np.array([probability for _, probability in children])
            if self.aggregator == "hindsight":
                values[ACTION_CONTINUE] = float(weights @ np.array([self._hindsight_return(int(row), depth + 1) for row in rows]))
            else:
                if self.aggregator == "arithmetic":
                    values[ACTION_CONTINUE] = _logsumexp(self.beta * child_values, weights=child_weights) / self.beta
                else:
                    values[ACTION_CONTINUE] = float(child_weights @ child_values)
        self.q_values[history] = values
        return _logsumexp(self.beta * values) / self.beta

    def policy(self, history: tuple[int, ...]) -> np.ndarray:
        """Forward policy ``P_F(. | h) = F(h, .) / F(h)`` over continue, accept, reject."""
        logits = self.beta * self.q_values[history]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum()

    def greedy_action(self, history: tuple[int, ...]) -> int:
        return int(np.argmax(self.q_values[history]))

    def expected_utility(self, greedy: bool = True) -> float:
        """True expected utility of the greedy or sampled forward policy under the sequence law."""

        @lru_cache(maxsize=None)
        def value(history: tuple[int, ...]) -> float:
            depth = len(history)
            rows, weights = self._conditional(history)
            terminal = self._terminal_utilities(rows, depth)
            probabilities = np.eye(3)[self.greedy_action(history)] if greedy else self.policy(history)
            total = 0.0
            for action in (ACTION_ACCEPT, ACTION_REJECT):
                if probabilities[action] > 0.0:
                    total += probabilities[action] * float(weights @ terminal[:, action])
            if probabilities[ACTION_CONTINUE] > 0.0:
                p_hit = float(weights @ self.sequences[rows, depth])
                continuation = sum(probability * value(history + (label,)) for label, probability in ((1, p_hit), (0, 1.0 - p_hit)) if probability > 0.0)
                total += probabilities[ACTION_CONTINUE] * continuation
            return total

        return value(())

    def terminal_distribution(self) -> dict[tuple[tuple[int, ...], int], float]:
        """Probability of ending at each (history, terminal action) under the forward policy and the law."""
        result: dict[tuple[tuple[int, ...], int], float] = {}

        def walk(history: tuple[int, ...], mass: float) -> None:
            probabilities = self.policy(history)
            for action in (ACTION_ACCEPT, ACTION_REJECT):
                if probabilities[action] > 0.0:
                    result[(history, action)] = result.get((history, action), 0.0) + mass * probabilities[action]
            if probabilities[ACTION_CONTINUE] > 0.0:
                rows, weights = self._conditional(history)
                p_hit = float(weights @ self.sequences[rows, len(history)])
                for label, probability in ((1, p_hit), (0, 1.0 - p_hit)):
                    if probability > 0.0:
                        walk(history + (label,), mass * probabilities[ACTION_CONTINUE] * probability)

        walk((), 1.0)
        return result


@dataclass
class ChainData:
    """Realised review chains for model-free training, one row per checkpoint.

    ``features`` holds only what the reviewer has observed at that checkpoint.
    ``utilities[i, j]`` is the realised utility of terminal action ``j`` at row
    ``i`` (``nan`` where the action is unavailable); it may depend on the
    unobserved future, which is what the chance node after a terminal action
    averages over. ``next_row[i]`` is the row the same ranking reaches after
    ``continue`` (``-1`` when continuing is impossible) and must exceed ``i``.
    """

    features: np.ndarray
    utilities: np.ndarray
    next_row: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.utilities = np.asarray(self.utilities, dtype=np.float64)
        if self.utilities.ndim == 1:
            self.utilities = self.utilities[:, None]
        self.next_row = np.asarray(self.next_row, dtype=np.int64)
        rows = len(self.features)
        if self.utilities.shape[0] != rows or self.next_row.shape != (rows,):
            raise ValueError("features, utilities and next_row must have one row per checkpoint")
        has_next = self.next_row >= 0
        if np.any(self.next_row[has_next] <= np.flatnonzero(has_next)) or np.any(self.next_row >= rows):
            raise ValueError("next_row must point forward within the data")
        if not np.all(has_next | np.isfinite(self.utilities).any(axis=1)):
            raise ValueError("every row needs continue or a terminal action")
        self.weights = np.ones(rows) if self.weights is None else np.asarray(self.weights, dtype=np.float64)

    @staticmethod
    def concatenate(parts: list["ChainData"]) -> "ChainData":
        offsets = np.cumsum([0] + [len(part.features) for part in parts[:-1]])
        return ChainData(
            np.concatenate([part.features for part in parts]),
            np.concatenate([part.utilities for part in parts]),
            np.concatenate([np.where(part.next_row >= 0, part.next_row + offset, -1) for part, offset in zip(parts, offsets)]),
            np.concatenate([part.weights for part in parts]),
        )

    def heights(self) -> np.ndarray:
        """Checkpoints remaining after each row along its chain (0 where continuing is impossible)."""
        heights = np.zeros(len(self.features), dtype=np.int64)
        for row in range(len(self.features) - 1, -1, -1):
            if self.next_row[row] >= 0:
                heights[row] = heights[self.next_row[row]] + 1
        return heights

    def policy_returns(self, probabilities: np.ndarray, beta: float, heights: np.ndarray | None = None) -> np.ndarray:
        """Expected soft return of a forward policy along each realised chain, from each row.

        ``probabilities`` is ``P_F`` over continue and the terminal actions at
        every row. Only the policy's own choices are averaged, exactly; the
        chain's labels are those observed, so the policy cannot use labels it
        has not reached. Entropy terms ``-(1/beta) log P_F`` make this the soft
        value of the policy, whose fixed point under ``P_F = softmax(beta Q)``
        is the geometric flow network.
        """
        heights = self.heights() if heights is None else heights
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = np.where(probabilities > 0.0, np.log(probabilities) / beta if np.isfinite(beta) else 0.0, 0.0)
        terminal = np.where(probabilities[:, 1:] > 0.0, probabilities[:, 1:] * (np.nan_to_num(self.utilities) - entropy[:, 1:]), 0.0).sum(axis=1)
        returns = np.zeros(len(self.features))
        order = np.argsort(heights, kind="stable")
        boundaries = np.searchsorted(heights[order], np.arange(heights.max() + 2))
        for height in range(heights.max() + 1):
            rows = order[boundaries[height] : boundaries[height + 1]]
            value = terminal[rows]
            if height:
                cont = probabilities[rows, 0]
                value = value + np.where(cont > 0.0, cont * (returns[self.next_row[rows]] - entropy[rows, 0]), 0.0)
            returns[rows] = value
        return returns

    def hindsight_returns(self, beta: float) -> np.ndarray:
        """(1/beta) log of every terminal reward reachable along each realised chain from each row."""
        returns = np.full(len(self.features), -np.inf)
        for row in range(len(self.features) - 1, -1, -1):
            logs = beta * self.utilities[row][np.isfinite(self.utilities[row])]
            if self.next_row[row] >= 0:
                logs = np.r_[logs, beta * returns[self.next_row[row]]]
            returns[row] = _logsumexp(logs) / beta
        return returns


class FlowStoppingNetwork:
    """A stopping flow network learned from realised chains (see the module docstring).

    The network outputs ``Q(x, .) = log F(x, .) / beta`` for continue (column 0)
    and each terminal action. Terminal columns regress on realised utilities:
    the squared loss makes them estimate the posterior expected utility, which
    is the geometric chance node. The continue column regresses on
    ``(1/beta) logsumexp(beta Q(x', .))`` at the next checkpoint, computed by a
    frozen copy refreshed every round (fitted soft value iteration), so the
    expectation over the next labels is again taken by the regression rather
    than inside one ranking. ``aggregator='hindsight'`` instead regresses
    continue on each chain's own future soft return, the original fixed-chain
    target.

    ``solver='bootstrap'`` is the one-step soft backup above. Over long chains it
    inherits the maximisation bias of fitted Q iteration: noise in the frozen
    copy's values is maximised at every step and compounds. ``solver='policy'``
    (the default) is soft policy iteration instead: each round evaluates the
    frozen copy's forward policy exactly along every realised chain
    (``ChainData.policy_returns``) and regresses continue on that return at the
    next checkpoint. It averages only over the policy's own choices, which
    depend on observed features alone, so it adds neither the maximisation bias
    nor the hindsight bias, and its fixed point satisfies the same geometric
    flow constraints. ``aggregator='myopic'`` regresses continue on the best realised
    terminal utility one checkpoint later, a one-step-lookahead comparator
    without any flow beyond the next checkpoint. ``beta=inf`` is hard fitted
    Q iteration.
    """

    def __init__(
        self, beta: float = 100.0, aggregator: str = "geometric", hidden: tuple[int, ...] = (64, 64), rounds: int = 200,
        steps: int = 25, batch_size: int = 4096, learning_rate: float = 1e-3, weight_decay: float = 1e-5, seed: int = 0,
        solver: str = "policy",
    ) -> None:
        if solver not in ("policy", "bootstrap"):
            raise ValueError("solver must be 'policy' or 'bootstrap'")
        if aggregator not in ("geometric", "hindsight", "myopic"):
            raise ValueError("learned networks support the geometric, hindsight and myopic aggregators")
        if not beta > 0.0:
            raise ValueError("beta must be positive")
        self.beta, self.aggregator, self.hidden = float(beta), aggregator, tuple(hidden)
        self.rounds, self.steps, self.batch_size = rounds, steps, batch_size
        self.learning_rate, self.weight_decay, self.seed, self.solver = learning_rate, weight_decay, seed, solver

    def _soft_value(self, q, available):
        import torch

        masked = torch.where(available, q, torch.full_like(q, -torch.inf))
        if np.isinf(self.beta):
            return masked.max(dim=1).values
        return torch.logsumexp(self.beta * masked, dim=1) / self.beta

    def fit(self, data: ChainData) -> "FlowStoppingNetwork":
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mean_ = data.features.mean(axis=0)
        self.scale_ = data.features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        x = torch.as_tensor((data.features - self.mean_) / self.scale_, dtype=torch.float32)
        terminal = torch.as_tensor(np.nan_to_num(data.utilities, nan=0.0), dtype=torch.float32)
        terminal_mask = torch.as_tensor(np.isfinite(data.utilities))
        next_row = torch.as_tensor(np.maximum(data.next_row, 0))
        has_next = torch.as_tensor(data.next_row >= 0)
        available = torch.cat([has_next[:, None], terminal_mask], dim=1)
        weights = torch.as_tensor(data.weights / data.weights.mean(), dtype=torch.float32)
        layers, width = [], x.shape[1]
        for size in self.hidden:
            layers += [torch.nn.Linear(width, size), torch.nn.SiLU()]
            width = size
        layers.append(torch.nn.Linear(width, 1 + data.utilities.shape[1]))
        self.model_ = torch.nn.Sequential(*layers)
        optimiser = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        fixed_target = None
        if self.aggregator == "hindsight":
            returns = data.hindsight_returns(self.beta if np.isfinite(self.beta) else 1e9)
            fixed_target = torch.as_tensor(np.where(data.next_row >= 0, returns[np.maximum(data.next_row, 0)], 0.0), dtype=torch.float32)
        elif self.aggregator == "myopic":
            best_next = np.nanmax(np.where(np.isfinite(data.utilities), data.utilities, -np.inf), axis=1)[np.maximum(data.next_row, 0)]
            fixed_target = torch.as_tensor(np.where(data.next_row >= 0, best_next, 0.0), dtype=torch.float32)
        heights = data.heights() if self.solver == "policy" else None
        available_array = available.numpy()
        self.history_ = []
        for round_index in range(self.rounds):
            if fixed_target is None and self.solver == "policy":
                probabilities = self._policy(self._predict_tensor(x).numpy().astype(float), available_array)
                returns = data.policy_returns(probabilities, self.beta, heights)
                continue_target = torch.as_tensor(np.where(data.next_row >= 0, returns[np.maximum(data.next_row, 0)], 0.0), dtype=torch.float32)
            elif fixed_target is None:
                with torch.no_grad():
                    values = self._soft_value(self._predict_tensor(x), available)
                    continue_target = torch.where(has_next, values[next_row], torch.zeros_like(values))
            else:
                continue_target = fixed_target
            target = torch.cat([continue_target[:, None], terminal], dim=1)
            total = 0.0
            for _ in range(self.steps):
                batch = torch.as_tensor(rng.integers(0, len(x), size=min(self.batch_size, len(x))))
                error = (self.model_(x[batch]) - target[batch]) ** 2 * available[batch]
                loss = (error.sum(dim=1) * weights[batch]).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.detach())
            self.history_.append(total / self.steps)
        self.model_.eval()
        return self

    def _predict_tensor(self, x):
        import torch

        with torch.no_grad():
            return torch.cat([self.model_(chunk) for chunk in torch.split(x, 65536)])

    def action_values(self, features: np.ndarray) -> np.ndarray:
        """``Q = log F / beta`` for continue and each terminal action, in utility units."""
        import torch

        x = torch.as_tensor((np.atleast_2d(features) - self.mean_) / self.scale_, dtype=torch.float32)
        return self._predict_tensor(x).numpy().astype(float)

    def forward_policy(self, features: np.ndarray, available: np.ndarray) -> np.ndarray:
        """``P_F(a | x) = F(x, a) / F(x)`` over the available actions (continue first)."""
        return self._policy(self.action_values(features), np.atleast_2d(available))

    def _policy(self, q: np.ndarray, available: np.ndarray) -> np.ndarray:
        q = np.where(available, q, -np.inf)
        if np.isinf(self.beta):
            best = q == q.max(axis=1, keepdims=True)
            return best / best.sum(axis=1, keepdims=True)
        logits = self.beta * q
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)
