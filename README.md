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

## Scope

The exact beta-binomial teacher is the coherent chain baseline. A rank-aware hierarchical count model, additional fully judged collections, and any evidence-set GFlowNet belong in separate follow-up experiments; the current fixed chain has no reward-proportional flow objective.
