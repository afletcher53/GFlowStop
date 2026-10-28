# A stopping flow network that is a GFlowNet where it can be

## Status

This document was written in two stages. The protocol and claim gates below
were committed before any leave-one-collection-out result existed. The results
section was added afterwards and must be read against them.

## The flaw in the original construction, stated as flow

A GFlowNet assumes that every edge of its DAG is chosen by the sampler. It sums
the flow of a state's children, so the policy can put mass on whichever child
is best. In stopping, two kinds of edge are not chosen by the reviewer:

* after `continue`, the environment reveals the next label;
* after a terminal action, the latent world (the topic's yield, or whether the
  target recall was reached) decides the payoff.

The fixed-chain construction summed over those edges too. Its continue flow is
the sum of every terminal reward further along one realised ranking, so the
reviewer is priced as if it could choose its best future stop with the future
labels in hand. `CLAIMS_AND_APPROACH.md` records why that is not
value-of-information training.

## The construction used here

`src/gflowstop/flow.py` keeps the GFlowNet constraints at the reviewer's
(agent) nodes:

    F(h) = sum_a F(h, a),     P_F(a | h) = F(h, a) / F(h),     F(terminal) = R = exp(beta * U),

and makes the aggregation at chance nodes explicit:

| Aggregator | Chance-node constraint | In value units `Q = log F / beta` | Large-`beta` limit |
| --- | --- | --- | --- |
| geometric | `log F(h,a) = E[log F(child) | h]` | `Q(h, continue) = E[V(h') | h]`, `V = (1/beta) logsumexp(beta Q)` | the Bayes-optimal policy |
| arithmetic | `F(h,a) = E[F(child) | h]` (the expected-flow constraint of EFlowNets) | `(1/beta) log E exp(beta V)` | the best outcome, not the expected one |
| hindsight | each realised chain sums its own future terminal rewards | `E[(1/beta) log sum_future exp(beta U)]` | the best future stop with future labels known |

With the geometric aggregator the flows are exactly the soft Bellman equation
of the belief MDP. When no chance node is left (a known world) all three
coincide with an ordinary tree GFlowNet whose terminal distribution is
proportional to `R`. The stopping indicator is the forward policy's
`P_F(stop | h)`.

`tests/test_flow.py` makes these properties executable:

* the exact geometric network attains the Bayes value of `ExactOracle` on the
  beta-binomial and delayed-information toys;
* with a point-mass law, every aggregator samples terminals in proportion to
  reward;
* the arithmetic network stops immediately in the perfect-information toy,
  where the Bayes policy continues;
* the hindsight continue flow is never below the geometric one, and is
  strictly above it somewhere;
* a network learned from realised chains alone reaches the Bayes value on an
  exchangeable toy within 0.002.

### Learning from realised rankings

Nothing about a real collection's law is known, so the network is learned from
realised rankings. A terminal head regresses on realised utilities; the
squared or cross-entropy loss makes it the posterior expectation, which is the
geometric chance node. The continue head cannot regress on a ranking's own
future (that is the hindsight target). Two solvers are implemented:

* `solver='bootstrap'`: one-step soft backup through a frozen copy (fitted soft
  value iteration). Over 200 checkpoints it showed the maximisation bias of
  fitted Q iteration: continue values above every attainable utility.
* `solver='policy'` (used): soft policy iteration. Each round evaluates the
  frozen copy's forward policy exactly along every realised ranking and
  regresses on the value at the next checkpoint. Only the policy's own choices
  are averaged, and they depend only on observed features, so the target
  carries neither bias. Its fixed point satisfies the geometric constraints.

### Target-recall stopping (Track R)

`src/gflowstop/tar_flow.py` applies the network to the TAR task. Stopping
after `n` of `N` documents has utility

    U = -(n/N + penalty * [recall(n) < target]) / (1 + penalty)

The flows are parameterised by that structure:

    log F(x, stop)     / beta = -(n/N + penalty * p_miss(x)) / (1 + penalty)
    log F(x, continue) / beta = -(c(x) + penalty * m(x)) / (1 + penalty) + e(x)

`p_miss` is a classifier of whether the target is still unmet. It never sees
the penalty. `c`, `m` and `e` are the review fraction, miss probability and
entropy bonus still to come under the current forward policy, and `c` is bounded
below by the next checkpoint's review. The target and the penalty are inputs, so
one network covers every operating point. Every feature is computed from the
reviewed prefix only; a test flips every later label and checks that no feature
changes. Stops are allowed at 200 evenly spaced checkpoints (every document for
rankings shorter than 200).

## Data

`scripts/build_clef_trajectories.py` rebuilds the CLEF 2017–2019 AutoTAR
trajectories without the CLEF archives, which the session's network policy
blocked:

* rankings: the hash-pinned AutoTAR files of Bin-Hezam and Stevenson
  (`TAR_Stopping_Point_Processes` at `709f808`, the same SHA-256 values
  pyterrier-tar pins);
* labels for 2018 and 2019: CLEF-TAR abstract qrels at `dbc13d0`;
* labels for 2017: the per-topic qrels of `dli1/auto-stop-tar` at `7e72795`.
  The 2017 ranking file mixes CLEF's 2017 test and training topics, and was
  produced with these labels. With CLEF-TAR's training qrels, seven 2017 topics
  put their relevant documents at the very end of the ranking and the oracle's
  mean review at 0.9 becomes 28%, not 5.7%.

The oracle's mean review at 0.9 is 5.7%, 6.7% and 7.1% for 2017, 2018 and
2019, which is pyterrier-tar's published value for each year. Its baseline replay
(`scripts/clef_baselines.py`) reproduces the estimator and heuristic rows of
pyterrier-tar's `RESULTS.md`.

Extra training rankings: the twelve further 2017 AutoTAR topics
(`clef2017_training_autotar_ranking.txt`) and 48 of the 2017 and 2018
participant runs that Bin-Hezam and Stevenson curated. These are labelled with
the CLEF-TAR test qrels. Runs that omit relevant documents are skipped, as are
byte-identical duplicates.

## Protocol (fixed before the evaluation)

* Leave one collection out. When 2017, 2018 or 2019 is the test collection,
  training uses the AutoTAR rankings of the other two, the twelve extra 2017
  topics, and participant runs of other years (run topics weigh 0.25 each). Any
  ranking whose topic id is in the test collection is dropped (CD011686 is
  judged in both 2018 and 2019).
* The penalty for each target is chosen by inner leave-one-collection-out over
  the two training collections: the cheapest grid penalty whose pooled
  validation reliability is at least 0.95. Test labels are used only to score
  stops.
* Each network is an ensemble of three seeds (flows averaged in log space),
  with `beta = 1e4`, two 64-unit layers and 120 rounds of 25 steps.
* Every rule, learned or not, is scored by pyterrier-tar's `run_experiment` and
  `stopping_metrics` on the same trajectories.

Disclosure: the features, the structured parameterisation, and the switch from
the bootstrap solver to policy iteration were developed while inspecting
failures on the 2019 fold with its labels. Penalty selection never used test
labels. The 2019 row is therefore not a clean held-out estimate; 2017 and 2018
are.

### Claim gates

At each target, a learned rule **beats** a baseline only if both hold:

1. its mean review fraction is lower, with the 95% topic-bootstrap interval of
   the paired difference entirely below zero; and
2. its reliability (share of the 91 topics reaching the target) is at least the
   baseline's.

The primary comparisons are with IP-H (`tail='model'`), the most reliable of
the cheap estimators, with IP-P, which is reliable on every topic, with
ConsecutiveIrrelevant, and with RLStop trained on the same topics. A result that
beats a baseline on review but loses reliability is reported as a trade-off,
not a win. The grid-penalty frontier is descriptive only, because choosing a
point on it uses test labels.

Ablations under the same protocol, with no gate: hindsight continue targets
(the original fixed-chain flow), one-step myopic targets, and `beta = inf`
(hard policy iteration).

## Results of the declared evaluation (revision 1)

Written after the run, against the gates above. Configuration
`results/flow/v1/config.json` (28 base features, the `cheapest` penalty rule,
penalties up to 64). Pooled over the 91 test topics, at the penalty chosen by
inner validation:

| Target | Reliability | Mean review | IP-H (`tail='model'`) reliability / review |
| --- | --- | --- | --- |
| 0.8 | 96.7% | 13.1% | 100.0% / 17.6% |
| 0.9 | 95.6% | 19.7% | 97.8% / 18.5% |
| 0.95 | 95.6% | 32.2% | 97.8% / 20.0% |

It does not beat IP-H (`tail='model'`) at any target. At 0.8 it reviews less but
is less reliable, which the gates count as a trade-off. The chosen penalties
varied erratically between folds: for 2018 the penalty was 64 at target 0.8 but
1 at 0.9. Inner validation picked a lucky low penalty from about sixty
topics.

## Revision 2 (declared before its run)

Changes, decided after seeing revision 1's test rows and recorded here before
revision 2 runs:

1. Tail-fit features: a Poisson power-law rate fitted to the reviewed prefix
   (the IP-P family) with a rough Poisson upper bound, and an exponential rate
   fitted to the last half of the prefix, each as an estimated recall and its
   log-odds margin from the target.
2. Penalty selection `stable`: the smallest penalty from which pooled
   validation reliability stays at or above 0.95 for every larger penalty.
3. Penalty grid and training range extended to 256.

Everything else, including the gates, is unchanged. Because these changes
followed revision 1's test results, revision 2 is a second look at the same
91 topics and not an independent confirmation; a held-out confirmation would
need another collection.

## Revision 3 (declared before its run)

Revision 2 was lost to a container restart before its first fold finished.
Partial single-seed ablation runs exposed the weak point. When 2017 is the test
collection, each inner-validation network trains on only 43 topics, because the
participant runs of both held-out years are excluded. Its miss classifier is then
confidently wrong on unfamiliar topics. Inner-validation reliability at target
0.9 plateaued near 85% however large the penalty.

Revision 3 is revision 2 plus sub-pool thinning of every training ranking: 8
random sub-pools per ranking, at weight 0.5 each. Each sub-pool keeps every
relevant document with probability U(0.6, 1) and every non-relevant one with
probability exp(U(log 0.1, 0)), preserving review order. It is a smaller and
often denser ranking of the same topic, which covers the small, dense topics
that failed. The choice was made on one inner split: training on 2019 and the
extra 2017 topics, validating on 2018. Thinning raised validation reliability
from 96.7/90.0/86.7% to 100/93.3/93.3% at the three targets, for about one
point more review. 2018 is a test collection in its own fold, so revision 3 is
also a second look and not an independent confirmation.

## Results of revision 3

Configuration in `results/flow/v3/config.json`. Every row is pyterrier-tar's
`run_experiment` on the same 91 topics (`results/flow/compare/`). Each cell is
reliability (share of topics reaching the target) / mean review fraction.

| Rule | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| **GFlowStop r3** (penalty from inner validation) | 95.6% / 8.2% | 92.3% / 12.6% | 93.4% / 18.2% |
| GFlowStop r1 | 96.7% / 13.1% | 95.6% / 19.7% | 95.6% / 32.2% |
| IP-H (published, `tail='released'`) | 93.4% / 16.1% | 90.1% / 16.8% | 89.0% / 18.1% |
| IP-H, Cox (CX-H) | 93.4% / 17.8% | 91.2% / 19.0% | 87.9% / 19.9% |
| IP-E | 98.9% / 17.1% | 95.6% / 18.0% | 94.5% / 19.4% |
| IP-H (`tail='model'`) | 100% / 17.6% | 97.8% / 18.5% | 97.8% / 20.0% |
| ConsecutiveIrrelevant | 100% / 20.8% | 97.8% / 20.8% | 95.6% / 20.8% |
| IP-P | 100% / 27.7% | 100% / 28.6% | 100% / 28.9% |
| RLStop (PPO, same training topics) | 62.6% / 6.6% | 65.9% / 9.1% | 73.6% / 11.2% |
| Oracle (offline bound) | 100% / 5.1% | 100% / 6.5% | 100% / 7.8% |

Against the gates, with paired topic-bootstrap 95% intervals for the review
difference:

* **It beats the published IP-H at 0.8 and 0.9.** It reviews 7.9 points less
  [−10.3, −5.5] and 4.2 points less [−6.8, −1.3], with reliability +2.2 points
  both times. At 0.95 review is equal (+0.1 [−2.9, +3.4]), so it is not a win
  although reliability is 4.4 points higher.
* **It beats CX-H at 0.8 and 0.9.** It reviews 9.6 points less [−12.1, −7.2]
  and 6.4 points less [−9.2, −3.3], with reliability +2.2 and +1.1 points. At
  0.95 review is equal (−1.7 [−4.8, +1.8]), with reliability +5.5 points.
* **It does not beat IP-H (`tail='model'`), IP-E, IP-P or
  ConsecutiveIrrelevant at any target.** It is 5–20 points cheaper at 0.8 and
  0.9 than each of them, but misses the target on a net 1–7 more topics. That is a
  trade-off, not a win.
* **Against RLStop**, it reaches the target on 33, 26 and 20 points more topics,
  for 1.6, 3.6 and 7.1 points more review.

By collection at 0.9, reliability / review: 2017 96.7% / 9.8%, 2018 80.0% / 8.7%,
2019 100% / 19.1%. IP-H (`tail='model'`) gets 100% / 16.3%, 93.3% / 17.4% and
100% / 21.7%. The 2018 shortfall comes from penalty selection. Inner
validation on 2017 and 2019 reached 95% at a penalty of 1, but 2018 is harder.
Inner-validation reliability was above test reliability in every fold.

The penalty grid (descriptive only, because reading it uses test labels; see
`results/flow/clef_frontier_r3.png`) shows that the learned network itself is
competitive with the best estimators:

| Penalty | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| 4 | 100% / 11.2% | 95.6% / 13.2% | 93.4% / 16.6% |
| 8 | 100% / 13.3% | 96.7% / 16.3% | 97.8% / 20.2% |
| 16 | 100% / 15.4% | 97.8% / 20.8% | 97.8% / 25.4% |

At target 0.8, every penalty from 4 upwards reaches all 91 topics with 2–6
points less review than IP-H (`tail='model'`). At 0.9 and 0.95 the grid passes
through IP-H (`tail='model'`)'s point without dominating it. The bottleneck is
choosing the operating point from about sixty validation topics across
collections that differ, not the flow network.

A post-hoc sensitivity check, not a claim, changes only the reliability goal of
the penalty selection. A goal of 0.975 gives 98.9% / 10.1%, 93.4% / 18.3% and
94.5% / 36.9%. A goal of 0.99 gives 98.9% / 10.6%, 95.6% / 22.8% and 98.9% /
38.9%. A stricter goal helps at 0.8 and makes selection erratic at the higher
targets.

### What this does and does not show

* It shows that stopping can be posed as a flow network whose chance nodes
  are aggregated risk-neutrally. In the exact toys this recovers the Bayes
  policy, while the expected-flow and fixed-chain alternatives fail in the
  directions their algebra predicts.
* It shows that a learned version of that network, trained on about 70 CLEF
  topics plus participant runs and thinned sub-pools, beats the published
  IP-H estimator and its Cox variant at targets 0.8 and 0.9 under a
  leave-one-collection-out protocol. It is far more reliable than RLStop
  trained on the same topics.
* It does not show superiority to the strongest estimators (IP-H with the
  model tail, IP-E, IP-P) or to ConsecutiveIrrelevant.
* It is not a recall certificate.
* Revision 3 is a second look at the same 91 topics, and its 2019 fold also
  guided feature design. Confirmation needs another collection, such as
  SYNERGY with pyterrier-tar's rankings, or TREC Total Recall.

## Ablations (revision 3 settings, one seed, no penalty selection)

Each variant changes only the continue targets of the reward-structured network,
with the same features, thinning and data. One seed each and frontiers only:
inner validation is skipped, so these grids are descriptive. The table gives the
least mean review on the penalty grid at which pooled reliability reaches 95%,
97.8% (IP-H `tail='model'` at 0.9 and 0.95), and 100%, and the best reliability
reached at any penalty (`results/flow/ablations/`).

| Continue target | Target | Review at ≥95% | at ≥97.8% | at 100% | Best reliability |
| --- | --- | --- | --- | --- | --- |
| geometric, soft policy iteration (`beta = 1e4`) | 0.8 | 8.2% | 9.0% | 11.4% | 100% |
| | 0.9 | 17.1% | 25.3% | – | 98.9% |
| | 0.95 | 28.3% | 38.6% | 55.0% | 100% |
| geometric, `beta = inf` | 0.8 | 8.1% | 8.8% | 11.0% | 100% |
| | 0.9 | 15.2% | 26.1% | – | 98.9% |
| | 0.95 | 22.0% | 29.7% | 46.1% | 100% |
| hindsight (fixed chain, best future stop) | 0.8 | 10.8% | 10.8% | 10.8% | 100% |
| | 0.9 | 20.2% | 23.7% | 45.2% | 100% |
| | 0.95 | 22.7% | 37.3% | 51.0% | 100% |
| myopic (one checkpoint ahead) | 0.8 | – | – | – | 64.8% |
| | 0.9 | – | – | – | 50.5% |
| | 0.95 | – | – | – | 46.2% |

Reading:

* **Looking more than one checkpoint ahead is essential.** The one-step
  comparator stops early whatever the penalty, and never reaches more than
  46–65% of topics.
* **The entropy (soft) part of the flow contributes nothing measurable at
  `beta = 1e4`.** Hard policy iteration traces the same frontier, within
  single-seed noise. On CLEF the flow network's value is its risk-neutral chance
  nodes and multi-step evaluation, not the GFlowNet sampling temperature.
* **The fixed-chain hindsight target behaves as its algebra predicts.** At every
  penalty it continues longer than the geometric network, because it prices
  continuing as if the best future stop could be chosen with the future labels
  in hand. Since the penalty is a free knob, its frontier is not uniformly worse
  on CLEF: worse at the 95% level, better near 100% at target 0.9. What the
  risk-neutral construction buys there is a penalty with its decision-theoretic
  meaning (the cost of a miss in units of review), which the hindsight
  network's penalty does not have. The exact toys show the same bias as lost
  utility at a fixed reward.
* The three-seed ensemble of revision 3 traces a better frontier than the
  single-seed geometric run. At target 0.9, a penalty of 8 gives 96.7% / 16.3%
  against 94.5% / 13.9%.

## Revision 4 (declared before its run)

Revision 3's fixed-penalty frontier is competitive with the best estimators, but
the penalty chosen by inner leave-one-collection-out was too aggressive in 2018.
Its misses are confident early stops on unfamiliar topics. Revision 4 keeps
revision 3's network, features and thinning, and changes how the ensemble is
built and how the operating point is chosen:

1. **Subbagging.** Ten members per test collection, each trained on a random
   70% of the training topics. All of a topic's rankings (AutoTAR, participant
   runs and thinned sub-pools) go in or out together.
2. **Out-of-bag selection.** Every AutoTAR training topic is scored by the
   members that never trained on it. The penalty for each target is chosen with
   the `stable` rule on those roughly 70 topics. Revision 3 used two
   inner-collection networks validated on about 60 topics, each trained on much
   less data than the deployed network. No extra training is needed.
3. **Cautious aggregation as a candidate.** Besides averaging flows (`mean`),
   the ensemble may stop only when its least willing member's forward policy
   would stop (`min`), so disagreement keeps the reviewer going. For each
   target, the aggregation with the lower out-of-bag review among those meeting
   the goal is deployed.
4. **Reliability goal 0.975** (was 0.95), the level of IP-H (`tail='model'`) at
   0.9 and 0.95.

The gates are unchanged. Revision 4 is a third look at the same 91 topics.

## External test on SYNERGY (declared before any SYNERGY result)

All CLEF numbers above come from collections that also guided the design. The
external test uses a collection that played no part in development:

* **Data.** ASReview SYNERGY 1.0, 26 reviews and about 169,000 records, read
  through pyterrier-tar's pinned `SynergyDataset`, with final-inclusion labels.
  SYNERGY has no released rankings, so `scripts/build_synergy_rankings.py`
  ranks each review by simulating AutoTAR, the procedure behind the CLEF
  rankings: TF-IDF features, the review title as a synthetic seed,
  logistic regression with 100 random pseudo-negatives, and batches growing by
  a tenth. Seed 0, fixed before any stopping result.
* **Method.** Revision 4 exactly as declared above. Ten subbagged members are
  trained on every CLEF ranking (2017–2019 AutoTAR, the extra 2017 topics,
  participant runs and thinned sub-pools). Penalties and aggregation are chosen
  out-of-bag on CLEF topics only. SYNERGY labels are used only to score stops.
  The method is applied whatever revision 4 scores on CLEF.
* **Comparisons.** The same pyterrier-tar rules as on CLEF, replayed on the
  same SYNERGY rankings, and RLStop trained on every CLEF AutoTAR topic.
* **Gates.** As for CLEF, over the 26 reviews: lower mean review with the
  paired 95% interval below zero, and reliability at least the baseline's.
  With 26 reviews, one review is 3.8 points of reliability.

## Results of revision 4

The 2019 fold first crashed, apparently out of memory (four members training in
parallel on the largest training set). It was rerun with two workers, resuming
2017 and 2018 from their saved members, which reproduced them exactly.
Artifacts are in `results/flow/v4/` and `results/flow/compare/`.

| Rule | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| **GFlowStop r4** | 96.7% / 8.8% | 95.6% / 12.3% | 94.5% / 19.0% |
| GFlowStop r3 | 95.6% / 8.2% | 92.3% / 12.6% | 93.4% / 18.2% |
| IP-H (published) | 93.4% / 16.1% | 90.1% / 16.8% | 89.0% / 18.1% |
| IP-E | 98.9% / 17.1% | 95.6% / 18.0% | 94.5% / 19.4% |
| IP-H (`tail='model'`) | 100% / 17.6% | 97.8% / 18.5% | 97.8% / 20.0% |
| ConsecutiveIrrelevant | 100% / 20.8% | 97.8% / 20.8% | 95.6% / 20.8% |

Out-of-bag selection chose penalties of 0.5–4 and averaged flows (`mean`) at
every target except 2017 at 0.95. Against the gates:

* **It beats the published IP-H at 0.8 and 0.9.** It reviews 7.3 points less
  [−9.7, −4.9] and 4.5 points less [−7.1, −1.7], with reliability +3.3 and
  +5.5 points.
* **It beats CX-H at 0.8 and 0.9.** It reviews 9.0 points less [−11.5, −6.6]
  and 6.7 points less [−9.4, −3.9], with reliability +3.3 and +4.4 points.
* **It beats IP-E at 0.9.** It reviews 5.7 points less [−8.4, −3.0] at equal
  reliability, 95.6%.
* **It does not beat IP-H (`tail='model'`), IP-P or ConsecutiveIrrelevant at any
  target.** Against IP-H (`tail='model'`) it is 8.8, 6.2 and 1.0 points cheaper
  and misses 3, 2 and 3 more topics.
* At 0.95 no review difference excludes zero.

Revision 4 is more reliable than revision 3 at every target (+1.1, +3.3, +1.1
points) for the same review. By collection at 0.9: 2017 96.7% / 12.8%, 2018
90.0% / 9.7%, 2019 100% / 14.4%. IP-H (`tail='model'`) gets 100% / 16.3%,
93.3% / 17.4% and 100% / 21.7%.

The ten-member ensemble's penalty grid (descriptive; reading it uses test
labels; `results/flow/clef_frontier_r4.png`):

| Penalty, aggregation | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| 2, mean | 100% / 10.6% | 95.6% / 12.3% | 93.4% / 14.4% |
| 4, mean | 100% / 12.0% | 95.6% / 13.7% | 96.7% / 16.8% |
| 8, mean | 100% / 14.2% | 100% / 16.5% | 97.8% / 20.5% |
| 16, mean | 100% / 16.9% | 100% / 18.9% | 98.9% / 27.1% |
| 2, min | 100% / 17.7% | 100% / 20.8% | 98.9% / 28.8% |

At a penalty of 8, the averaged ensemble would beat IP-H (`tail='model'`) at
0.8 and 0.9: equal or higher reliability for 3.4 and 2.0 points less review.
The deployed rule cannot claim that, because that penalty was not chosen
without test labels. Out-of-bag reliability on CLEF training topics was again
optimistic about a new collection. The penalties it chose sit just below the
frontier's knee, so selection, not the network, still limits the declared
result. The cautious `min` aggregation is uniformly more reliable and more
expensive; out-of-bag selection rarely preferred it.

## Results of the external SYNERGY test

The rankings (`results/flow/synergy/synergy_manifest.json`) are much harder than
CLEF's. With final-inclusion labels and simulated AutoTAR rankings, the oracle
needs 27.7% review on average to reach recall 0.9, against 6.5% on CLEF.
Revision 4 was trained on every CLEF ranking. Out-of-bag selection on CLEF chose
penalties of 1, 2 and 2 with averaged flows. Results on the 26 reviews:

| Rule | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| **GFlowStop r4** | 80.8% / 23.2% | 80.8% / 43.8% | 88.5% / 56.4% |
| IP-H (published) | 84.6% / 31.0% | 73.1% / 31.4% | 69.2% / 33.5% |
| IP-E | 88.5% / 33.4% | 76.9% / 34.4% | 73.1% / 36.6% |
| IP-H (`tail='model'`) | 92.3% / 34.2% | 84.6% / 35.6% | 80.8% / 39.2% |
| IP-P | 92.3% / 47.6% | 88.5% / 50.6% | 84.6% / 52.7% |
| ConsecutiveIrrelevant | 73.1% / 27.6% | 57.7% / 27.6% | 53.8% / 27.6% |
| Budget | 100% / 53.3% | 88.5% / 53.3% | 88.5% / 53.3% |
| CMHHeuristic | 100% / 55.7% | 100% / 70.7% | 100% / 81.9% |
| RLStop (trained on all CLEF) | 15.4% / 1.7% | 3.8% / 2.4% | 11.5% / 4.3% |
| Oracle | 100% / 19.7% | 100% / 27.7% | 100% / 32.5% |

**It beats no baseline under the gates.**

* At 0.8 it is cheaper than every IP rule but less reliable. Against
  ConsecutiveIrrelevant it is +7.7 points more reliable and 4.5 points cheaper,
  but the interval [−16.5, +8.1] includes zero.
* At 0.9 and 0.95 it becomes conservative on these weak rankings. It reviews
  more than the IP-H family and is more reliable than them at 0.95, a
  trade-off.
* It never reaches CMH's 100%.
* RLStop trained on CLEF collapses on SYNERGY, stopping almost at once.

The penalty grid (descriptive) shows what a well-placed operating point would
have given:

| Penalty, mean | 0.8 | 0.9 | 0.95 |
| --- | --- | --- | --- |
| 1 | 80.8% / 23.2% | 65.4% / 27.7% | 73.1% / 39.8% |
| 2 | 88.5% / 31.5% | 80.8% / 43.8% | 88.5% / 56.4% |
| 4 | 96.2% / 41.7% | 96.2% / 61.0% | 96.2% / 70.2% |
| 8 | 96.2% / 54.8% | 96.2% / 68.4% | 96.2% / 75.6% |

Even at its best, the grid is not clearly ahead of the point-process rules
here. A penalty of 4 at 0.8 is more reliable and cheaper than IP-P (96.2% /
41.7% against 92.3% / 47.6%); elsewhere the grid trades reliability against
review.

### What this means

The CLEF gains of revisions 3 and 4 do not transfer to a collection whose
rankings are much weaker than any the network was trained on. The learned
policy moves in the right direction: it reviews more when the gain curve is
flat, unlike RLStop, which collapses. But its estimate of the remaining
relevant documents is not calibrated outside CLEF's AutoTAR regime, and the
penalty chosen on CLEF does not carry over. The point-process estimators, which
fit each ranking's own gain curve with no training, travel better. A learned
stopper of this kind needs training rankings from the regime it is deployed in,
for example leave-one-review-out within SYNERGY, or a calibration step on a few
reviews from the target collection.

## Revision 5 (declared before its runs)

The SYNERGY result points at three remedies, tried here:

1. **Few-review calibration.** Choose the penalty and aggregation on `k`
   labelled topics of the target collection; score the rest
   (`scripts/calibrate_on_few.py`, 200 random draws per `k`).
2. **IP-H as an input feature.** At each checkpoint the network also sees
   pyterrier-tar's IP-H (`tail='model'`) state from its own checkpoints (every
   2.5% of the pool), each computed from the reviewed prefix only (tested):
   * whether the latest fit was accepted;
   * the last accepted fit's estimated and Poisson-bound recall, with their
     log-odds margins from the target;
   * whether the latter windows were empty;
   * how long ago the last accepted fit was taken.
3. **Training in the target domain.** SYNERGY's 26 reviews are split into five
   folds. Each fold is scored by a ten-member ensemble trained on all of CLEF
   plus the other folds' reviews (weight 3 each, thinned like AutoTAR). Its
   penalty and aggregation are chosen out-of-bag on those in-domain reviews
   only.

To keep five-fold training affordable, revision 5 thins only the AutoTAR and
SYNERGY rankings (not the participant runs) and fixes each ranking's sub-pools
across members. A control with the same settings but without the IP-H feature
is run as well. The runs:

| Run | Training | IP-H feature | Selection |
| --- | --- | --- | --- |
| S0 | all CLEF | no | out-of-bag on CLEF |
| S3 | all CLEF | yes | out-of-bag on CLEF |
| S1 | CLEF + other SYNERGY folds | yes | out-of-bag on in-domain reviews |
| C0 | CLEF, leave one collection out | no | out-of-bag on CLEF |
| C3 | CLEF, leave one collection out | yes | out-of-bag on CLEF |

The gates are unchanged. S0, S3 and S1 are scored on the 26 SYNERGY reviews and
C0, C3 on the 91 CLEF topics. RLStop stays trained on CLEF only.

### Result of remedy 1: few-review calibration (revision 4 networks)

Each cell is reliability / review on the held-out topics, averaged over 200
draws; IP-H (`tail='model'`) is on the same held-out topics. The last column is
the share of draws in which calibrated GFlowStop was both cheaper and at least
as reliable as IP-H (`tail='model'`) (`results/flow/calibration/`).

| Collection | k | Target | Calibrated GFlowStop | IP-H (`tail='model'`) | Draws cheaper and as reliable |
| --- | --- | --- | --- | --- | --- |
| CLEF (k per collection) | 3 | 0.9 | 84.7% / 8.2% | 97.8% / 18.6% | 2% |
| | 5 | 0.9 | 90.7% / 9.8% | 97.8% / 18.5% | 16% |
| | 8 | 0.8 | 93.3% / 8.2% | 100% / 17.6% | 4% |
| | 8 | 0.9 | 93.7% / 10.7% | 97.8% / 18.6% | 24% |
| | 8 | 0.95 | 92.9% / 15.5% | 97.8% / 20.1% | 11% |
| SYNERGY | 3 | 0.9 | 78.6% / 46.0% | 84.6% / 35.3% | 0% |
| | 5 | 0.9 | 85.9% / 55.1% | 84.3% / 35.6% | 0% |
| | 8 | 0.8 | 92.1% / 45.0% | 92.3% / 33.9% | 0% |
| | 8 | 0.9 | 92.2% / 63.1% | 84.5% / 35.2% | 0% |
| | 8 | 0.95 | 90.9% / 69.1% | 80.9% / 39.0% | 0% |

**Negative on both collections.** On CLEF, a few topics cannot estimate the
rate of rare misses. Every calibration topic reaches the target at most
penalties, so the choice is too aggressive, and out-of-bag selection over about
70 training topics does better (96.7% / 95.6% / 94.5%). On SYNERGY,
calibration raises reliability as `k` grows, but only by moving up a frontier
that is no better than IP-H's. In no draw is it both cheaper and as reliable.
Choosing the operating point better cannot fix a model that is not better.
