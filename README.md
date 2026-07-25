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

The Repke adapter is a frozen, negative target-recall screen. The primary
Track-U follow-up is the explicit RCV1 imposed-world horizon protocol: it fits
only the supplied 94 training rankings, reports held-out next-hit loss and
root diagnostics on the 45 test rankings, then evaluates `W = 1[sum(y) >= K]`
without presenting it as a natural decision world. CLEF-IP, CLEF eHealth, and
DTA are excluded from new external claims because they are owned artifacts.
See `EXTERNAL_BENCHMARK.md` for the required root preflight and source hashes.

```bash
.venv/bin/amortized-stopping-repke /path/to/repke/data/rankings \
  --framework-revision "$(git -C /path/to/stopping-methods rev-parse HEAD)" \
  --check-interval 15
```

See [EXTERNAL_BENCHMARK.md](EXTERNAL_BENCHMARK.md) for the two-track protocol,
data provenance, and the claim gates before any external comparison.

### First RCV1 horizon result

The included `results/external/rcv1_horizon_patched/` ledger is a controlled,
in-house AutoTAR-style RCV1 reconstruction, not the authors' unpublished
GRLStop rankings or a natural RCV1 decision task. It covers 6,480 held-out
topic/cell rows over the fixed 94/45 category split. Held-out next-hit log loss
is 0.01951--0.02117; 95.6--97.8% of root decisions continue and have positive
fitted-model information value.

On the imposed world, horizon 2 has mean utility -0.05314 versus -0.07738 for
horizon 1 (paired difference +0.02424 across 1,620 topic/cell pairs). The
mean is driven by rare large gains: 1,050 pairs favour horizon 1, 485 tie, and
85 favour horizon 2. Thus `H*` is descriptive only, not a selected deployment
rule or a claim of external superiority. The ledger records the ranking hashes,
fixed seed, implementation revision, and its pre-commit dirty-tree provenance.

## Scope

The exact beta-binomial teacher is the coherent chain baseline. The toy suite adds a rank-aware latent belief planner; its results must pass the listed synthetic gates before Vaswani or other real collections are used as evidence. A fixed chain still has no reward-proportional flow objective: an evidence-set GFlowNet remains a separate research branch, not a relabeling of hindsight targets.
