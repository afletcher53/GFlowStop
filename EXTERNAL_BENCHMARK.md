# External benchmark protocol

## Status

The repository now has a target-recall adapter for the public Repke et al.
stopping-method framework. It has passed a miniature schema/cross-fitting test
and one predeclared external Rank-only screen. No comparison with published
baselines, CLEF-IP, or DTA result is currently claimed.

The external benchmark has two separate leaderboards. This is deliberate:
target-recall methods are assessed by safety and effort, while downstream
decision methods are assessed by their own utility. Neither is graded on the
other objective as its primary score.

## Track R: fixed-ranking target recall

The initial target is the public framework accompanying Repke et al.,
`destiny-evidence/stopping-methods`. The adapter was checked against revision
`f880b63d29f1922783d55f564be68e29d5238fa5`; every actual run must record the
framework revision that generated or evaluated its supplied rankings. The
paper links its raw ranking simulations at Zenodo record `17497378`. The
archive is large and is not committed to this repository; it must remain under
`data/external/`.

The adapter reads its paired ranking metadata JSON and Feather files. By
default, it selects the paper's 500-initial-record, 25--200 adaptive-batch
protocol, yielding 81 dataset IDs from the public archive. The release has
217 matching ranking files, rather than the paper's reported 223 simulations;
the metadata records the actual selected count. It keeps the exact supplied
review order, uses the framework's published `--batch-size 15` rounded batch
boundaries, and emits
`repke_trace.csv` fields compatible with the framework's method logs. Existing
CMH, TM-QBCB, QUANT_CI, S-CAL, SAL, IPP, Knee, and heuristic rows remain
outputs of the external framework; they are not reimplemented here.

Run after obtaining the external ranking directory:

```bash
.venv/bin/amortized-stopping-repke /path/to/repke/data/rankings \
  --framework-revision "$(git -C /path/to/stopping-methods rev-parse HEAD)" \
  --output-dir results/external/repke \
  --targets 0.8,0.9,0.95 --alphas 0.05,0.10 --check-interval 15
```

The command writes:

- `repke_trace.csv`: every permitted stopping check, to merge with baseline
  traces on the same rankings;
- `repke_first_stops.csv`: first posterior stop (or full screen) per ranking;
- `repke_summary.csv`: failure rate, one-sided Clopper-Pearson upper bounds,
  worst-dataset upper bound, effort, and realised recall;
- `metadata.json`: source hashes, framework revision, GFlowStop revision, and
  all declared settings.

### GFlowStop-R method

For observed positives `h_d` and target recall `tau`, the residual-count cap
is `floor(h_d (1-tau) / tau)`. The method stops at a check point only when its
posterior probability of remaining count at or below that cap is at least
`1-alpha`.

The full Poisson-binomial residual distribution is exact in the short toy
planner but unsuitable for rankings of thousands of documents. This external
adapter uses the normal approximation for each discrete latent-intensity state
and mixes those CDFs by the current posterior intensity weights. Its method
name contains `normal-posterior` for that reason.

The primary external variant is rank-only. The synthetic score advantage was
mainly a raw topic-score-mean effect, which should not be trusted across real
queries. `--include-local-scores` adds only within-ranking normalised local
score features as a secondary ablation. No raw score-mean result belongs on
the main leaderboard without a separate calibration study.

Beliefs are fitted by deterministic dataset-level cross-fitting: all simulated
rankings of a held-out dataset are excluded from its belief-model fit. The
default has five dataset folds. A ranking's labels are used only after that
rank has been selected for simulated review; unrevealed labels never enter the
belief update.

The threshold is a posterior statement, not a recall certificate. External
failure rate and the worst-cell one-sided upper confidence bound are the
headline safety measures.

### Initial external screen

The initial run used the archived ranking files with SHA-256
`27395ee71116c629c7aba516c0c70aab782f39e90488405c180de8d798634e40`, the
published-protocol filter, 15-document checks, five dataset-level folds, and
GFlowStop source `3ec09d14106ac420e64bf94c417d7b90a4f8b41c`. It contains 217
rankings from 81 dataset IDs. The checked archive is therefore short of the
paper's stated 223 simulations; it is an external screen on the released
files, not an exact paper reproduction.

| Target recall | Confidence | Failure rate | Global 95% upper bound | Review fraction |
| --- | ---: | ---: | ---: | ---: |
| 0.80 | 0.95 | 18.89% | 23.81% | 44.85% |
| 0.80 | 0.90 | 25.35% | 30.67% | 42.38% |
| 0.90 | 0.95 | 1.38% | 3.53% | 74.96% |
| 0.90 | 0.90 | 1.38% | 3.53% | 72.12% |
| 0.95 | 0.95 | 0.00% | 1.37% | 90.08% |
| 0.95 | 0.90 | 0.00% | 1.37% | 88.49% |

This rejects any claim that the rank-only posterior rule is safe at 80%
recall. The 90% and 95% aggregate rows are encouraging but do not pass the
worst-dataset gate: each dataset has only one to three ranking simulations, so
a zero-failure cell can still have a 95% upper bound between 63% and 95%.
There is also no matched published-baseline result in this bundle yet.

The committed first-stop rows, summary, and metadata are sufficient to audit
the table. The 132 MB checkpoint trace remains ignored because it is fully
regenerable from the archived ranking hashes and the recorded source revision.

## Track U: downstream decision utility

Repke's collections are appropriate for Track R but do not define a common
downstream decision world. They must not be coerced into the count-threshold
utility score for a headline comparison.

The first Track-U external tasks are instead:

1. CLEF-IP, with a fixed `(K=1)` prior-art decision and the published payoff,
   cost, ranking, calibration, and topic split;
2. DTA outcomes, with the adoption/reject decision defined by the cumulative
   evidence posterior and its published outcome split.

Those adapters are intentionally not present yet: each needs its task-specific
observation model and a frozen source-data/ranking manifest. The generic
Repke loader is reusable for fixed ranking provenance, but it is not a
substitute for the task definition.

## Promotion gates

No external claim is promoted until all of these hold:

1. The public archive and framework revision are recorded and all source file
   hashes are written with the results.
2. At least one baseline row is reproduced from the supplied framework before
   GFlowStop is compared with it.
3. The exact same fixed ranking simulations and check intervals are used for
   every Track-R method.
4. Safety is reported by target, confidence, source collection, and ranking
   condition; the maximum condition-specific one-sided upper bound is used,
   not a grand mean.
5. Effort comparisons exclude methods failing the declared safety constraint.
6. Any Track-U result uses the task's own downstream utility and is reported
   separately from recall safety.

The next executable experiment is to reproduce selected framework baselines
on these exact 217 files, then merge their traces with GFlowStop before any
efficiency comparison. CLEF, SYNERGY, and RCV1 are then reported as source or
hold-out groups within that protocol; CLEF-IP and DTA begin only after their
task-specific inputs are frozen.
