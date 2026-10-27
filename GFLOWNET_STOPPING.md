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
