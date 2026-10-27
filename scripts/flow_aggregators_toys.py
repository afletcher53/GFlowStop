"""Exact expected utility of greedy and sampled forward policies for each chance-node aggregator.

Writes results/flow/toy_aggregators.csv: for five fixed toy laws and a range of
beta, the true expected utility (under the law) of the flow network's greedy
and sampled policies, next to the Bayes value of ``ExactOracle``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gflowstop.flow import AGGREGATORS, ExactFlowNetwork
from gflowstop.toys import (
    ExactOracle,
    RankDecayEnvironment,
    beta_binomial_sequence_distribution,
    clustered_sequence_distribution,
    delayed_sequence_distribution,
    independent_sequence_distribution,
    mixture_sequence_distribution,
)

BETAS = (3.0, 10.0, 30.0, 100.0, 1_000.0, 100_000.0)


def laws():
    environment = RankDecayEnvironment(1.5, 1.0, 0.5, mu=1.0)
    etas = np.linspace(-2.0, 4.0, 13)
    return {
        "perfect_information (L=1, K=1, q*=0.5, c=0.01)": (independent_sequence_distribution(np.array([0.5])), 1, 0.5, 0.01),
        "beta_binomial(1.5, 3) (L=8, K=3, q*=0.4, c=0.02)": (beta_binomial_sequence_distribution(1.5, 3.0, 8), 3, 0.4, 0.02),
        "delayed_information (L=10, K=2, q*=0.5, c=0.01)": (delayed_sequence_distribution(10, 3), 2, 0.5, 0.01),
        "clustered (L=10, K=3, q*=0.5, c=0.01)": (clustered_sequence_distribution(10), 3, 0.5, 0.01),
        "rank_decay mixture (L=10, K=3, q*=0.5, c=0.01)": (
            mixture_sequence_distribution(np.stack([environment.probabilities(eta, 10) for eta in etas]), np.exp(-0.5 * (etas - 1.0) ** 2)),
            3, 0.5, 0.01,
        ),
    }


def main() -> None:
    rows = []
    for name, ((sequences, probabilities), threshold, q_star, cost) in laws().items():
        bayes = ExactOracle(sequences, probabilities, threshold, q_star, cost).values[()]
        for aggregator in AGGREGATORS:
            for beta in BETAS:
                network = ExactFlowNetwork(sequences, probabilities, threshold, q_star, cost, beta, aggregator)
                rows.append({"law": name, "aggregator": aggregator, "beta": beta, "bayes_value": bayes,
                             "greedy_utility": network.expected_utility(greedy=True), "sampled_utility": network.expected_utility(greedy=False)})
    frame = pd.DataFrame(rows)
    output = Path("results/flow/toy_aggregators.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.pivot_table(index=["law", "aggregator"], columns="beta", values="greedy_utility").round(4).to_string())


if __name__ == "__main__":
    main()
