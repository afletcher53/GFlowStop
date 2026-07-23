# External benchmark protocol

## Status

The repository now has a target-recall adapter for the public Repke et al.
stopping-method framework. It has passed a miniature schema/cross-fitting test,
one external Rank-only screen, and a matched direct CMH baseline reproduction.
No comparison with the remaining published baselines, CLEF-IP, or DTA result
is currently claimed.

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
`repke_trace.csv` fields compatible with the framework's method logs. The
included `scripts/run_repke_cmh.py` calls the framework's CMH implementation
directly (`methods.buscar.Buscar` with `bias=1`); other baselines remain
outputs of the external framework and are not reimplemented here.

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

To produce the matched CMH baseline from the same framework checkout:

```bash
.venv/bin/python scripts/run_repke_cmh.py /path/to/repke/data/rankings \
  --framework-root /path/to/stopping-methods \
  --framework-revision "$(git -C /path/to/stopping-methods rev-parse HEAD)" \
  --output-dir results/external/repke_cmh
```

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

### Matched external screen

The matched runs used the archived ranking files with SHA-256
`27395ee71116c629c7aba516c0c70aab782f39e90488405c180de8d798634e40`, the
published-protocol filter, the framework's rounded `--batch-size 15` boundaries,
and 217 rankings from 81 dataset IDs. The GFlowStop run was made from clean
source `7da7072405c34ddaba8d0580bec6c0e661b1e7ae`; CMH was called directly
from framework revision `f880b63d29f1922783d55f564be68e29d5238fa5` by source
`fc37da869fc29064925b824cafefaeb2d838a4dc`. The checked archive is short of
the paper's stated 223 simulations, so this is an external screen on the
released files, not an exact paper reproduction.

Both methods use a requested 95% confidence level. CMH denotes the framework's
`BUSCAR` implementation with its documented `bias=1.0` setting.

| Target recall | Method | Failure rate | Global 95% upper bound | Review fraction |
| --- | --- | ---: | ---: | ---: |
| 0.80 | GFlowStop-R rank only | 19.82% | 24.80% | 44.81% |
| 0.80 | CMH | 0.46% | 2.17% | 55.02% |
| 0.90 | GFlowStop-R rank only | 1.38% | 3.53% | 74.90% |
| 0.90 | CMH | 0.00% | 1.37% | 65.24% |
| 0.95 | GFlowStop-R rank only | 0.00% | 1.37% | 90.08% |
| 0.95 | CMH | 0.00% | 1.37% | 76.02% |

This is a negative result for the current recall method. At 80% recall, the
rank-only posterior rule fails the 5% safety gate while CMH passes it. At 90%
and 95%, its global bound is below 5%, but CMH reviews fewer records at the
same aggregate bound. Both methods fail the predeclared worst-dataset gate:
each dataset has only one to three simulations, so even a zero-failure cell
can have a 95% upper bound between 63% and 95%.

The optional local-score ablation is worse still: its failure rates are 22.58%,
5.99%, and 1.38% at the three targets, while its review fractions are 40.22%,
59.11%, and 81.62%. It therefore trades safety for early stopping and does not
alter the conclusion.

The committed first-stop rows, summary, and metadata are sufficient to audit
the table. The 132 MB checkpoint trace remains ignored because it is fully
regenerable from the archived ranking hashes and the recorded source revision.

## Track U: downstream decision utility

Repke's collections are appropriate for Track R but do not define a common
downstream decision world. They must not be coerced into the count-threshold
utility score for a headline comparison.

### Constructed CLEF-IP K=1 diagnostic

The repository includes one deliberately narrow, non-promoted Track-U screen
from the local `PatentSearch` checkout. Its `omega_combined` split has 1,000
topics in each partition: 500 original positive CLEF-IP topics and 500
`OMEGAN_` corpus-sampled structural-zero controls. It is a constructed mixed
screening split, not natural external negative-world evidence.

`python -m gflowstop.external_utility` fits the rank-latent model only on the
positive training topics and adds a structural-zero component. Its exact K=1
Bellman solver is valid along the only nonterminal history (a run of misses).
The optional score-mean variant learns a one-feature world prior solely from
training topics. When `--greedy-script` is supplied, it calls the external
project's `simulate_topic(..., policy="greedy")` directly and records a
per-topic paired comparison.

The first matched run uses `q*=0.5`, `c=0.01`, 1,000 fixed test topics, and
the source CSV hashes in
`results/external/clefip_k1_matched/metadata.json`. GFlowStop source was clean
at `bc9b1c68899fa1b473be793e0d6fe5a214304227`. The external checkout was at
`9e1320fcb60217a233d7272170396251408af0f3` but dirty; the directly called
Greedy source has SHA-256
`02dfd36c209eb9cade6da22705f4e6166b8c8713bd97ccd658120198aca404f7`.

| Method | Mean utility | SE | Mean depth | Paired utility difference vs Greedy | SE | Decision accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct external Greedy | 0.17137 | 0.00529 | 1.363 | -- | -- | 0.870 |
| Zero-inflated rank | 0.13581 | 0.00483 | 7.669 | -0.03556 | 0.00616 | 0.925 |
| Zero-inflated score mean | 0.24183 | 0.00171 | 0.267 | +0.07046 | 0.00566 | 0.989 |

The score-mean result is **not a breakthrough claim**. Its root-world AUC is
`0.99907`, and only 4% of test topics continue: raw mean BM25 scores nearly
identify which half of this constructed split consists of corpus-sampled
zero controls. The rank-only model is the relevant model-repair result here,
and it loses to direct Greedy. This experiment is retained as a leakage and
external-validity diagnostic, not as evidence of non-myopic planning or
real-data generalisation.

The next valid Track-U tasks remain CLEF-IP with a published natural
prior-art split and DTA outcomes with their cumulative-evidence decision.
Each needs a frozen task-specific source-data/ranking manifest. The generic
Repke loader is reusable for fixed ranking provenance, but it is not a
substitute for the decision world.

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

The recall branch is frozen at this negative result. The constructed CLEF-IP
screen is also frozen as a diagnostic. The next experimental branch is a
natural Track-U task with a task-specific input manifest. A new recall-policy
variant requires a predeclared calibration mechanism and a fresh held-out
benchmark, not another threshold sweep on these 217 runs.
