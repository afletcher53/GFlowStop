# External benchmark protocol

## Status

The repository has a target-recall adapter for the public Repke et al.
stopping-method framework and a separate, controlled Track-U horizon protocol.
The existing Repke screen is negative. The owned CLEF collections and the
SIGIR'26 DTA artifact set are excluded from new external claims: they may
remain as historical diagnostics, but are not valid benchmark rows.

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

## Track U: imposed threshold-decision utility

RCV1 is the primary Track-U collection because its fixed 94/45 category split
can support a belief-quality ladder. This is explicitly a controlled decision
task, not a claim that RCV1 has a natural downstream decision: for each test
ranking the protocol sets `W = 1[sum(y) >= K]`. `K` is selected from the
training-fitted root belief to make its predicted world probability closest to
the declared `q*`; test labels are used only after the policy acts.

The runner accepts the in-house fixed AutoTAR train/test ranking CSVs
(`Topic`, `Rank`, `Score`, `label`) and retains the supplied source revision.
Run its root preflight first, inspect `root_diagnostics.csv`, then run
the horizon sweep in a separate output directory:

```bash
.venv/bin/python -m gflowstop.external_horizon /path/to/rcv1_train.csv /path/to/rcv1_test.csv \
  --source-root /path/to/autotar --source-revision "$(git -C /path/to/autotar rev-parse HEAD)" \
  --train-sizes 10,25,50,94 --horizons 1,2,4,8 \
  --q-stars 0.3,0.5,0.7 --kappas 0.05,0.20,0.50 \
  --expected-train-topics 94 --expected-test-topics 45 --require-common-length \
  --root-only --output-dir results/external/rcv1_root_preflight

.venv/bin/python -m gflowstop.external_horizon /path/to/rcv1_train.csv /path/to/rcv1_test.csv \
  --source-root /path/to/autotar --source-revision "$(git -C /path/to/autotar rev-parse HEAD)" \
  --train-sizes 10,25,50,94 --horizons 1,2,4,8 \
  --q-stars 0.3,0.5,0.7 --kappas 0.05,0.20,0.50 \
  --expected-train-topics 94 --expected-test-topics 45 --require-common-length \
  --output-dir results/external/rcv1_horizon
```

For each training size, the rank-belief model is fit on a deterministic nested
subset of the 94 training categories and evaluated only on the held-out 45.
Its features are rank, within-topic score z-scores, and score slopes; raw
topic-score means are disabled. The planner uses a normal mixture for the
remaining-count tail and emits all horizon cells, so `H=1` is the internal
one-step/Greedy analogue and fixed larger `H` values cover the corresponding
block-lookahead family without treating any as a published baseline.

`calibration.csv` records held-out next-hit log loss. The root calculation in
the preflight uses only the train-fitted belief and observable test rankings;
the accompanying calibration report evaluates that belief against held-out
labels. It records root continuation fraction and information value at the
largest requested horizon before a utility curve is generated. These
diagnostics are not a real-data oracle; a flat curve when they show zero
continuation or zero information is non-evidence. `utility_summary.csv`
reports the descriptive `H*` for every training-size, `q*`, and cost cell.
Horizon selection is not deployed or tuned on the held-out curve.

### First RCV1 horizon run: decision-live, not promoted

The local RCV1 AutoTAR-style reproduction completed the root preflight and
the 6,480-row horizon sweep in
`results/external/rcv1_root_preflight_patched/` and
`results/external/rcv1_horizon_patched/`. Across 10, 25, 50, and 94 training
topics, held-out next-hit log loss was `0.01951` to `0.02117`; root
continuation was `95.6%` to `97.8%`, with mean root information value
`0.0781` to `0.1110`. The curve is therefore decision-live rather than a
flat/no-information diagnostic.

On the imposed world, `H=2` has mean utility `-0.05314` versus `-0.07738`
for `H=1` (paired mean difference `+0.02424` across 1,620 topic/cell pairs).
It is the descriptive `H*` in 27 of 36 cells; `H=4` and `H=8` are selected in
four and five cells. This mean advantage is not uniform: 1,050 pairs favour
`H=1`, 485 tie, and 85 favour `H=2`, so rare large gains offset many small
losses. These rows establish neither a deployable horizon choice nor a
natural RCV1 decision task.

The run used a dirty local GFlowStop tree at base revision
`7ce492a1c4f2b52ff8fb457e9425f8d19e88862a` and has no supplied external
source revision. It remains a provenance-marked internal diagnostic until
the builder and result-producing source are committed and the public-source
gate below is met.

SYNERGY and non-CLEF Repke/TREC rankings are secondary out-of-domain checks
after the RCV1 curve is decision-live. CLEF-IP, CLEF eHealth, and DTA are
excluded from this external robustness set because they are owned data/method
artifacts. An intervention-review evidence-state decision is a separate
future task, not a substitute for this imposed-world experiment.

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
6. Track-U reports the imposed world, train-only threshold selection,
   held-out next-hit loss, and root diagnostics separately from recall safety.
7. The full RCV1 horizon curve is reported before any SYNERGY, TREC, or Repke
   robustness result is interpreted.

The recall branch remains frozen at its negative result. The CLEF-IP and DTA
artifacts are retained only as historical diagnostics; neither can be revived
as an external comparison. A natural intervention-review decision model would
require an evidence-state formulation and separate data extraction.
