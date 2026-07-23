# Amortized Bayesian Stopping

This is cost-sensitive Bayesian finite-horizon stopping with policy distillation. It is **not** a GFlowNet and does not claim soft-optimality.

For a fixed ranked pool, the fitted beta-binomial model maintains a posterior over the number of relevant documents remaining. The high-yield probability is derived from that count distribution. The exact count-state Bellman teacher evaluates accept, reject, and the expected value of the next label; hard certificates force accept once the yield threshold is reached and reject once it is no longer attainable. A compact log-linear student is trained from teacher and student-induced states.

The default Vaswani evaluation uses all 93 topics, five repeated yield-stratified outer folds, nested baseline tuning, topic-level bootstrap intervals, and fixed split IDs written to the results directory. It treats unjudged documents as non-relevant; the collection is a smoke test, not external validation.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest
.venv/bin/amortized-stopping
```

Outputs are `per_topic.csv`, `summary.csv`, `statistics.json`, `splits.json`, `metadata.json`, and `report.md`. Direct net utility is primary. The reported full-information gap is only an upper-bound comparison, not a hindsight stopping oracle.

## Synthetic benchmark

The toy benchmark is the primary validation path. It separates two non-interchangeable tasks:

- Track U: terminal threshold-decision utility minus per-document cost.
- Track R: minimise unique reviewed documents subject to a target-recall failure bound.

```bash
.venv/bin/python -m gflowstop.toy_experiment
```

This writes fixed episodes (labels, scores, generator parameters, and seeds) to `data/toys/`, and condition-level utility, planner, calibration, and recall results to `results/toys/`. It runs the perfect-information hindsight-target regression, beta-binomial oracle/learning curve, rank decay and score-shuffle ablations, delayed-information planning ladder, clustered and shifted stress tests, and target-recall safety/effort comparisons. QBCB and the alpha-spending hypergeometric rule are kept in the recall track; estimator and point-process rules are reported as empirical comparators, not certificates.

The bounded interpretation of these results, including what is and is not a
GFlowNet claim, is in [CLAIMS_AND_APPROACH.md](CLAIMS_AND_APPROACH.md).

## External benchmark protocol

The first external adapter consumes Repke et al.'s fixed public ranking
simulations without re-ranking them. It is a target-recall track only; its
posterior stopping statement is evaluated empirically and is not presented as
a frequentist certificate.

```bash
.venv/bin/amortized-stopping-repke /path/to/repke/data/rankings \
  --framework-revision "$(git -C /path/to/stopping-methods rev-parse HEAD)"
```

See [EXTERNAL_BENCHMARK.md](EXTERNAL_BENCHMARK.md) for the two-track protocol,
data provenance, and the claim gates before any external comparison.

## Scope

The exact beta-binomial teacher is the coherent chain baseline. The toy suite adds a rank-aware latent belief planner; its results must pass the listed synthetic gates before Vaswani or other real collections are used as evidence. A fixed chain still has no reward-proportional flow objective: an evidence-set GFlowNet remains a separate research branch, not a relabeling of hindsight targets.
